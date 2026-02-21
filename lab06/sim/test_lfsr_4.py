# sim/tests/test_lfsr_4.py
import cocotb
import os
import sys
from pathlib import Path
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles

EXPECTED_FROM_0001 = [
    0b0010, 0b0100, 0b1000, 0b0011, 0b0110,
    0b1100, 0b1011, 0b0101, 0b1010, 0b0111,
    0b1110, 0b1111, 0b1101, 0b1001, 0b0001,
]

async def sync_reset_and_seed(dut, seed_val: int):
    dut.seed.value = seed_val & 0xF
    dut.rst.value = 1
    await ClockCycles(dut.clk, 1)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

async def step_and_read(dut, cycles=1):
    for _ in range(cycles):
        await ClockCycles(dut.clk, 1)
    return int(dut.q.value) & 0xF

@cocotb.test()
async def lfsr_stuck_at_zero_when_seed_zero(dut):
    cocotb.start_soon(Clock(dut.clk, 5, units="ns").start())
    await sync_reset_and_seed(dut, 0b0000)
    for _ in range(20):
        q = await step_and_read(dut, 1)
        assert q == 0, f"Expected 0000 while seeded with 0000, got {q:04b}"

@cocotb.test()
async def lfsr_maximal_length_for_nonzero_seeds(dut):
    cocotb.start_soon(Clock(dut.clk, 5, units="ns").start())

    for seed in [0x1, 0x2, 0x4, 0x8, 0xF]:
        await sync_reset_and_seed(dut, seed)
        first = int(dut.q.value) & 0xF
        assert first == seed, f"After reset expected q==seed ({seed:04b}), got {first:04b}"

        seen = set([first])
        for i in range(1, 16):
            q = await step_and_read(dut, 1)
            if i < 15:
                assert q not in seen, (
                    f"State repeated too early for seed {seed:04b}: {q:04b} at step {i}"
                )
                seen.add(q)
            else:
                assert q == first, (
                    f"Did not return to seed after 15 steps for seed {seed:04b}: got {q:04b}"
                )

@cocotb.test()
async def lfsr_sequence_matches_spec_from_0001(dut):
    cocotb.start_soon(Clock(dut.clk, 5, units="ns").start())
    await sync_reset_and_seed(dut, 0b0001)

    for idx, expect in enumerate(EXPECTED_FROM_0001, start=1):
        q = await step_and_read(dut, 1)
        assert q == expect, (
            f"Step {idx}: expected {expect:04b}, got {q:04b}"
        )

from cocotb.runner import get_runner

def lfsr_runner():
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent
    sources = [proj_path / "hdl" / "lfsr_4.sv"]
    hdl_toplevel = "lfsr_4"

    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel=hdl_toplevel,
        always=True,
        timescale=("1ns", "1ps"),
        waves=True,
    )
    runner.test(hdl_toplevel=hdl_toplevel, test_module=os.path.basename(__file__).replace(".py",""))

if __name__ == "__main__":
    lfsr_runner()
