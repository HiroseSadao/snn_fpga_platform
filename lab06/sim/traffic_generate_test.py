# file: sim/traffic_generator_tb.py
import os
from pathlib import Path
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Event


# ========= helpers =========

CLK_PERIOD_NS = 12  # ~83.33 MHz controller clk（サイト説明の想定クロックに合わせる）
# ref: "controller clock 83.333MHz" in assignment text.  :contentReference[oaicite:2]{index=2}

async def tick(dut, n=1):
    await ClockCycles(dut.clk, n)

def set_default_inputs(dut):
    # AXIS write (into DRAM)
    if hasattr(dut, "write_axis_tdata"):
        dut.write_axis_tdata.value  = 0
        dut.write_axis_tvalid.value = 0
        dut.write_axis_tlast.value  = 0
    # AXIS read (out to unstacker/consumer)
    if hasattr(dut, "read_axis_tready"):
        dut.read_axis_tready.value  = 0

    # UberDDR3-like interface (inputs to DUT)
    if hasattr(dut, "memrequest_resp_data"):
        dut.memrequest_resp_data.value = 0
    if hasattr(dut, "memrequest_complete"):
        dut.memrequest_complete.value = 0
    if hasattr(dut, "memrequest_busy"):
        dut.memrequest_busy.value = 0


async def reset(dut):
    dut.rst.value = 1
    set_default_inputs(dut)
    await tick(dut, 8)
    dut.rst.value = 0
    await tick(dut, 2)


# ========= simple DRAM controller model =========
class MockMemCtrl:
    """
    Very simple model behind traffic_generator:
    - When DUT asserts memrequest_en:
        * if memrequest_write_enable=1: write memrequest_write_data into memory[address]
          and after a small delay, pulse memrequest_complete.
        * else (read): after a small delay, place memory[address] on memrequest_resp_data
          and pulse memrequest_complete.
    - Can optionally inject 'busy' cycles.
    """
    def __init__(self, dut):
        self.dut = dut
        self.mem = {}  # address -> 128-bit int
        self.pending = []  # queue of (delay, is_write, addr, data)
        self.random_busy = True

    async def run(self):
        while True:
            await RisingEdge(self.dut.clk)

            # (1) Occasionally assert busy to stress DUT arbitration
            if hasattr(self.dut, "memrequest_busy"):
                if self.random_busy and random.random() < 0.06:
                    self.dut.memrequest_busy.value = 1
                else:
                    self.dut.memrequest_busy.value = 0

            # (2) Accept new request from DUT
            if int(self.dut.memrequest_en.value) == 1 and int(self.dut.memrequest_busy.value) == 0:
                addr = int(self.dut.memrequest_addr.value)
                is_write = int(self.dut.memrequest_write_enable.value)
                data = int(self.dut.memrequest_write_data.value)

                # enqueue completion after a small random latency
                latency = random.randint(2, 7)
                self.pending.append([latency, is_write, addr, data])

            # (3) Service oldest pending when delay expires: drive complete (+ resp_data if read)
            if self.pending:
                self.pending[0][0] -= 1
                if self.pending[0][0] <= 0:
                    _, is_write, addr, data = self.pending.pop(0)
                    if is_write:
                        self.mem[addr] = data
                        # write complete
                        self.dut.memrequest_complete.value = 1
                        await RisingEdge(self.dut.clk)
                        self.dut.memrequest_complete.value = 0
                    else:
                        # read: drive resp_data, pulse complete
                        val = self.mem.get(addr, 0)
                        self.dut.memrequest_resp_data.value = val
                        self.dut.memrequest_complete.value = 1
                        await RisingEdge(self.dut.clk)
                        self.dut.memrequest_complete.value = 0


# ========= drivers/monitors =========

async def drive_write_axis(dut, words):
    """
    Send a frame worth of 128-bit words into the DUT via write AXIS.
    words: list[int] length N. TLAST high on the final element.
    """
    assert hasattr(dut, "write_axis_tdata"), "Missing write_axis_tdata on DUT"
    assert hasattr(dut, "write_axis_tvalid"), "Missing write_axis_tvalid on DUT"
    assert hasattr(dut, "write_axis_tready"), "Missing write_axis_tready on DUT"
    assert hasattr(dut, "write_axis_tlast"), "Missing write_axis_tlast on DUT"

    dut.write_axis_tvalid.value = 0
    dut.write_axis_tlast.value = 0
    await tick(dut, 2)

    for i, w in enumerate(words):
        last = (i == len(words) - 1)
        dut.write_axis_tdata.value = w
        dut.write_axis_tvalid.value = 1
        dut.write_axis_tlast.value = 1 if last else 0

        # wait handshake
        while int(dut.write_axis_tready.value) == 0:
            await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)

    # deassert
    dut.write_axis_tvalid.value = 0
    dut.write_axis_tlast.value = 0
    await tick(dut, 2)


async def collect_read_axis(dut, tlast_event: Event, sink_ready_mode="random"):
    """
    Collect data from read AXIS until TLAST is seen; signal via tlast_event.
    Returns the captured list of words (as attribute on event).
    """
    assert hasattr(dut, "read_axis_tdata"), "Missing read_axis_tdata on DUT"
    assert hasattr(dut, "read_axis_tvalid"), "Missing read_axis_tvalid on DUT"
    assert hasattr(dut, "read_axis_tready"), "Missing read_axis_tready on DUT"
    assert hasattr(dut, "read_axis_tlast"), "Missing read_axis_tlast on DUT"

    captured = []
    while True:
        # drive tready
        if sink_ready_mode == "always":
            dut.read_axis_tready.value = 1
        else:
            dut.read_axis_tready.value = 1 if random.random() < 0.8 else 0

        await RisingEdge(dut.clk)

        if int(dut.read_axis_tvalid.value) == 1 and int(dut.read_axis_tready.value) == 1:
            word = int(dut.read_axis_tdata.value)
            last = int(dut.read_axis_tlast.value)
            captured.append(word)
            if last == 1:
                # latch result for waiter
                tlast_event.data = captured
                tlast_event.set()
                return


# ========= tests =========

@cocotb.test()
async def traffic_generator_basic_frame_test(dut):
    """
    Scenario:
      1) Reset DUT
      2) Drive a short "frame" (e.g., 16 words) into write AXIS; TLAST on the last.
      3) MockMemCtrl returns write completions; upon DUT issuing reads, the model returns stored words.
      4) Verify read AXIS reproduces the same sequence with TLAST, honoring tready back-pressure.
    The exact address schedule is not asserted (DUT's evt_counters are free to choose),
    but the round-trip content and TLAST framing must match.
    """
    # clock
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    await reset(dut)

    # spin up mock controller
    mem = MockMemCtrl(dut)
    cocotb.start_soon(mem.run())

    # prepare a tiny "frame" of 128-bit words
    # keep it small so this runs quickly in sim; content is unique & checkable
    N = 16
    words = []
    base = 0xA5A5_0000_0000_0000_0000_0000_0000_0001
    for i in range(N):
        words.append((base + i) & ((1 << 128) - 1))

    # collect readout concurrently
    got_frame_event = Event()
    cocotb.start_soon(collect_read_axis(dut, got_frame_event, sink_ready_mode="random"))

    # drive a frame into write axis
    await drive_write_axis(dut, words)

    # Wait until TLAST observed on read axis
    await got_frame_event.wait()
    got = got_frame_event.data

    assert len(got) == len(words), f"read length mismatch: exp {len(words)} got {len(got)}"
    assert got == words, f"read data mismatch:\nexp={list(map(hex, words))}\ngot={list(map(hex, got))}"


@cocotb.test()
async def traffic_generator_backpressure_busy_test(dut):
    """
    Stress test with heavier controller busy/back-pressure:
      - Heavier random busy on memrequest_busy
      - Sink drops tready more often
    Must still deliver identical frame back.
    """
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset(dut)

    mem = MockMemCtrl(dut)
    mem.random_busy = True
    cocotb.start_soon(mem.run())

    N = 20
    words = []
    base = 0x5AA5_1234_0000_0000_0000_0000_0000_0010
    for i in range(N):
        words.append((base ^ (i * 0x1234_5678)) & ((1 << 128) - 1))

    got_frame_event = Event()
    cocotb.start_soon(collect_read_axis(dut, got_frame_event, sink_ready_mode="random"))
    await drive_write_axis(dut, words)
    await got_frame_event.wait()
    got = got_frame_event.data

    assert got == words, "Data must be preserved round-trip even with busy/back-pressure"


# ========= runner =========

def traffic_generator_runner():
    """
    Build & run using cocotb runner. We compile all hdl/*.sv to include dependencies
    (e.g., command_fifo, evt_counter, etc.). Top-level must be 'traffic_generator'.
    """
    from cocotb.runner import get_runner
    sim = os.getenv("SIM", "icarus")  # or "verilator"
    repo_root = Path(__file__).resolve().parents[1]
    hdl_dir = repo_root / "hdl"

    # Collect all SystemVerilog sources under hdl/
    sv_sources = [str(p) for p in hdl_dir.glob("**/*.sv")]

    runner = get_runner(sim)
    runner.build(
        sources=sv_sources,
        hdl_toplevel="traffic_generator",
        parameters={},  # add if your DUT has parameters (e.g., FRAME sizes)
        timescale=("1ns", "1ps"),
        waves=True,
        always=True,
        build_dir=str(repo_root / "sim" / "sim_build_traffic_gen"),
    )
    runner.test(
        hdl_toplevel="traffic_generator",
        test_module=Path(__file__).stem,  # this file
        seed=None,
    )


if __name__ == "__main__":
    traffic_generator_runner()
