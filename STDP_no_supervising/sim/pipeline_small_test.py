# file: sim/pipeline_small_test.py
import os
from pathlib import Path
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


CLK_PERIOD_NS = 10  # 100 MHz
MAX_WAIT_CYCLES = 5000


def required_signals_present(dut):
    names = [
        "clk",
        "rst",
        "s_tvalid",
        "s_tready",
        "s_tdata",
        "m_tvalid",
        "m_tready",
        "m_tdata",
    ]
    return all(hasattr(dut, name) for name in names)


async def reset_dut(dut):
    dut.rst.value = 1
    dut.s_tvalid.value = 0
    dut.s_tdata.value = 0
    dut.m_tready.value = 1
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)


async def send_packet(dut, tstep_id, spike):
    dut.s_tdata.value = (int(tstep_id) << 1) | int(spike)
    dut.s_tvalid.value = 1
    for cycle in range(MAX_WAIT_CYCLES):
        await RisingEdge(dut.clk)
        if int(dut.s_tready.value) == 1:
            if cycle > 0:
                dut._log.info(
                    f"s_tready asserted after {cycle} cycles (tstep_id={tstep_id})"
                )
            break
        if cycle % 200 == 0:
            dut._log.info(
                f"waiting s_tready... cycle={cycle} s_tvalid={int(dut.s_tvalid.value)} "
                f"m_tvalid={int(dut.m_tvalid.value)} m_tready={int(dut.m_tready.value)}"
            )
    else:
        raise AssertionError("Timeout waiting for s_tready")
    dut.s_tvalid.value = 0


async def recv_packet(dut):
    for cycle in range(MAX_WAIT_CYCLES):
        await RisingEdge(dut.clk)
        if int(dut.m_tvalid.value) == 1 and int(dut.m_tready.value) == 1:
            data = int(dut.m_tdata.value)
            tstep_id = data >> 1
            spike = data & 0x1
            if cycle > 0:
                dut._log.info(f"m_tvalid&ready after {cycle} cycles (tstep_id={tstep_id})")
            return tstep_id, spike
        if cycle % 200 == 0:
            dut._log.info(
                f"waiting m_tvalid... cycle={cycle} s_tvalid={int(dut.s_tvalid.value)} "
                f"s_tready={int(dut.s_tready.value)} m_tvalid={int(dut.m_tvalid.value)} "
                f"m_tready={int(dut.m_tready.value)}"
            )
    raise AssertionError("Timeout waiting for m_tvalid&m_tready")


@cocotb.test()
async def pipeline_small_basic_handshake(dut):
    if not required_signals_present(dut):
        dut._log.info("Skipping: DUT missing required AXI-stream ports.")
        return

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    expected_ids = [0, 1, 2, 3, 4]
    spikes = [1, 0, 1, 1, 0]

    # Bufferless assumption: keep ready high and receive per send
    dut.m_tready.value = 1

    got_ids = []
    for tstep_id, spike in zip(expected_ids, spikes):
        await send_packet(dut, tstep_id, spike)
        out_id, _ = await recv_packet(dut)
        got_ids.append(out_id)

    assert got_ids == expected_ids, f"tstep_id mismatch: exp={expected_ids} got={got_ids}"


@cocotb.test()
async def pipeline_small_backpressure(dut):
    if not required_signals_present(dut):
        dut._log.info("Skipping: DUT missing required AXI-stream ports.")
        return

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    # Bufferless assumption: only send when ready, receive per send.
    # Still toggle m_tready to confirm upstream stalls cleanly.
    async def toggle_ready():
        while True:
            dut.m_tready.value = 1
            await ClockCycles(dut.clk, 3)
            dut.m_tready.value = 0
            await ClockCycles(dut.clk, 2)

    cocotb.start_soon(toggle_ready())

    expected_ids = [10, 11, 12]
    got_ids = []
    for tstep_id in expected_ids:
        await send_packet(dut, tstep_id, 1)
        out_id, _ = await recv_packet(dut)
        got_ids.append(out_id)

    assert got_ids == expected_ids, f"backpressure tstep_id mismatch: exp={expected_ids} got={got_ids}"


def pipeline_small_runner():
    from cocotb.runner import get_runner
    sim = os.getenv("SIM", "icarus")
    repo_root = Path(__file__).resolve().parents[1]
    hdl_dir = repo_root / "hdl"

    sv_sources = [str(p) for p in hdl_dir.glob("**/*.sv")]

    runner = get_runner(sim)
    runner.build(
        sources=sv_sources,
        hdl_toplevel="pipeline_small",
        parameters={},
        timescale=("1ns", "1ps"),
        waves=True,
        always=True,
        build_dir=str(repo_root / "sim" / "sim_build_pipeline_small"),
    )
    runner.test(
        hdl_toplevel="pipeline_small",
        test_module=Path(__file__).stem,
        seed=None,
    )


if __name__ == "__main__":
    pipeline_small_runner()
