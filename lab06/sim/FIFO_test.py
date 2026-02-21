import os
from pathlib import Path
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

async def tick(dut, n=1):
    await ClockCycles(dut.clk, n)

async def fifo_write(dut, value: int):
    dut.command_in.value = value
    dut.write.value = 1
    await tick(dut, 1)
    dut.write.value = 0

async def fifo_pop_and_get(dut) -> int:
    await tick(dut, 1)
    data = int(dut.command_out.value)

    dut.read.value = 1
    await tick(dut, 1)
    dut.read.value = 0
    return data

@cocotb.test()
async def command_fifo_spec_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    dut.rst.value = 1
    dut.write.value = 0
    dut.read.value = 0
    dut.command_in.value = 0
    await tick(dut, 5)
    dut.rst.value = 0
    await tick(dut, 1)

    assert int(dut.empty.value) == 1, "After reset, empty must be 1"
    assert int(dut.full.value) == 0, "After reset, full must be 0"

    seq1 = [0x11, 0x22, 0x33, 0x44]
    for w in seq1:
        await fifo_write(dut, w)

    await tick(dut, 1)
    assert int(dut.empty.value) == 0, "After writes, empty should be 0"

    for i, exp in enumerate(seq1):
        got = await fifo_pop_and_get(dut)
        assert got == exp, f"[basic rd/wr] idx={i}: exp {hex(exp)} got {hex(got)}"

    await tick(dut, 1)
    assert int(dut.empty.value) == 1, "After draining, empty should be 1"
    assert int(dut.full.value) == 0, "After draining, full should be 0"

    DEPTH = 16

    values_written = []
    next_val = 0xA000
    safety = DEPTH
    while int(dut.full.value) == 0 and safety > 0:
        await fifo_write(dut, next_val)
        values_written.append(next_val)
        next_val += 1
        safety -= 1

    await tick(dut, 1)
    assert int(dut.full.value) == 1, "FIFO should assert full after enough writes"
    assert int(dut.empty.value) == 0, "FIFO should not be empty when full"
    n_written = len(values_written)
    assert n_written > 0, "No data written before full asserted"

    # Read several values and verify that full deasserts (while also checking order)
    k = 3
    popped = []
    for _ in range(k):
        popped.append(await fifo_pop_and_get(dut))

    await tick(dut, 1)
    assert int(dut.full.value) == 0, "After several reads, full should deassert"

    for idx, val in enumerate(popped):
        exp = values_written[idx]
        assert val == exp, f"[wrap read] idx={idx} exp {hex(exp)} got {hex(val)}"

    for _ in range(n_written - k):
        _ = await fifo_pop_and_get(dut)

    await tick(dut, 1)
    assert int(dut.empty.value) == 1, "After draining all entries, empty must be 1"

    for _ in range(2):
        _ = await fifo_pop_and_get(dut)
    await tick(dut, 1)
    assert int(dut.empty.value) == 1, "Empty should remain 1 when over-reading"

    # wrapping around
    # prefill
    await tick(dut, 1)
    prefill = DEPTH - 2
    A = []
    val = 0xC100
    for _ in range(prefill):
        await fifo_write(dut, val)
        A.append(val)
        val += 1

    await tick(dut, 1)
    assert int(dut.empty.value) == 0, "Prefill failed: FIFO should not be empty"

    r = 5
    # pop
    popped_A_head = []
    for _ in range(r):
        popped_A_head.append(await fifo_pop_and_get(dut))
    assert popped_A_head == A[:r], f"Head pops mismatch: exp {list(map(hex, A[:r]))} got {list(map(hex, popped_A_head))}"

    await tick(dut, 1)

    B = []
    val = 0xD200
    for _ in range(r):
        await fifo_write(dut, val)
        B.append(val)
        val += 1

    await tick(dut, 1)

    #    [ A[r], A[r+1], ..., A[prefill-1], B[0], B[1], ..., B[r-1] ]
    expected = A[r:] + B[:]
    got_all = []
    for _ in range(len(expected)):
        got_all.append(await fifo_pop_and_get(dut))

    assert got_all == expected, f"[wrap verify] order mismatch:\nexp={list(map(hex, expected))}\ngot={list(map(hex, got_all))}"

    await tick(dut, 1)



def command_fifo_runner():
    from cocotb.runner import get_runner

    sim = os.getenv("SIM", "icarus")  # "icarus" or "verilator"
    proj_root = Path(__file__).resolve().parents[1]  # .../lab06/
    hdl_file = proj_root / "hdl" / "command_fifo.sv"

    runner = get_runner(sim)
    runner.build(
        sources=[str(hdl_file)],
        hdl_toplevel="command_fifo",
        parameters={
            "DEPTH": 16,
            "WIDTH": 16,
        },
        timescale=("1ns", "1ps"),
        waves=True,
        always=True,
        build_dir=str(proj_root / "sim" / "sim_build"),
    )
    runner.test(
        hdl_toplevel="command_fifo",
        test_module=Path(__file__).stem,
        seed=None,
    )


if __name__ == "__main__":
    command_fifo_runner()
