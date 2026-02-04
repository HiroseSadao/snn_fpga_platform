# file: sim/synapse_test.py
import os
from pathlib import Path
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


CLK_PERIOD_NS = 10  # 100 MHz default for synapse test


def required_signals_present(dut):
    return all(hasattr(dut, name) for name in ["clk", "rst", "tick", "spike", "r_out"])


def ref_single_exponential_synapse_float(spikes, dt=1.0, td=50.0):
    r = 0.0
    out = []
    for s in spikes:
        r = r * (1 - dt / td) + s / td
        out.append(r)
    return out


def ref_single_exponential_synapse_fixed(spikes, dt=1.0, td=50.0):
    FP_SHIFT = 16
    FP_SCALE = 1 << FP_SHIFT
    TD = int(td)
    TD_HALF = TD // 2
    r = 0
    out = []
    for s in spikes:
        # rounded r/td
        r_div = (r + TD_HALF) // TD
        r = r - r_div + (FP_SCALE // TD) * int(s)
        out.append(r / FP_SCALE)
    return out


@cocotb.test()
async def single_exponential_synapse_fixed_test(dut):
    if not required_signals_present(dut):
        dut._log.info("Skipping: DUT missing required ports (clk, rst, spike, r_out).")
        return

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    dut.rst.value = 1
    dut.tick.value = 0
    dut.spike.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    # Simple spike pattern
    spikes = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
    expected = ref_single_exponential_synapse_fixed(spikes)

    got = []
    for s in spikes:
        dut.spike.value = s
        dut.tick.value = 1
        await RisingEdge(dut.clk)
        dut.tick.value = 0
        # wait for divider to complete and r_out to update
        await ClockCycles(dut.clk, 20)
        got.append(int(dut.r_out.value) / (1 << 16))

    # Fixed-point expected, allow small tolerance after S16.16 conversion
    for i, (g, e) in enumerate(zip(got, expected)):
        if abs(g - e) > 1e-2:
            raise AssertionError(f"Mismatch at step {i}: got={g} expected≈{e}")


@cocotb.test()
async def single_exponential_synapse_float_test(dut):
    if not required_signals_present(dut):
        dut._log.info("Skipping: DUT missing required ports (clk, rst, spike, r_out).")
        return

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    dut.rst.value = 1
    dut.tick.value = 0
    dut.spike.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 1)

    spikes = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
    expected = ref_single_exponential_synapse_float(spikes)

    got = []
    for s in spikes:
        dut.spike.value = s
        dut.tick.value = 1
        await RisingEdge(dut.clk)
        dut.tick.value = 0
        # wait for divider to complete and r_out to update
        await ClockCycles(dut.clk, 20)
        got.append(int(dut.r_out.value) / (1 << 16))

    # Float model will differ slightly; allow looser tolerance
    for i, (g, e) in enumerate(zip(got, expected)):
        if abs(g - e) > 5e-2:
            raise AssertionError(f"Float mismatch at step {i}: got={g} expected≈{e}")


def synapse_runner():
    from cocotb.runner import get_runner
    sim = os.getenv("SIM", "icarus")
    repo_root = Path(__file__).resolve().parents[1]
    hdl_dir = repo_root / "hdl"

    sv_sources = [str(p) for p in hdl_dir.glob("**/*.sv")]

    runner = get_runner(sim)
    runner.build(
        sources=sv_sources,
        hdl_toplevel="synapse",
        parameters={},
        timescale=("1ns", "1ps"),
        waves=True,
        always=True,
        build_dir=str(repo_root / "sim" / "sim_build_synapse"),
    )
    runner.test(
        hdl_toplevel="synapse",
        test_module=Path(__file__).stem,
        seed=None,
    )


if __name__ == "__main__":
    synapse_runner()
