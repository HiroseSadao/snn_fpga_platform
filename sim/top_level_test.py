# file: sim/lif_test.py
import os
from pathlib import Path
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


CLK_PERIOD_NS = 0.5  # 2 GHz (matches current test speed)


def set_default_inputs(dut):
    dut.start.value = 0
    dut.tick.value = 0


async def reset_and_start(dut):
    set_default_inputs(dut)
    await ClockCycles(dut.clk, 2)
    dut.start.value = 1
    await ClockCycles(dut.clk, 1)
    dut.start.value = 0
    await ClockCycles(dut.clk, 2)


def ref_lif_spike_times_fixed(steps):
    # Fixed-point S16.16 reference model (matches lif.sv)
    FP_SHIFT = 16
    FP_SCALE = 1 << FP_SHIFT
    V_REST = -60
    V_RESET = -65
    V_THR = -40
    I_IN = 21
    TAU_M = 200
    REFRACT = 40

    v_mem = V_RESET * FP_SCALE
    refr_cnt = 0
    spike_steps = []

    for i in range(steps):
        if refr_cnt != 0:
            refr_cnt -= 1
            v_mem = V_RESET * FP_SCALE
            continue

        num = (V_REST * FP_SCALE) - v_mem + (I_IN * FP_SCALE)
        # trunc toward zero
        dv = int(num / TAU_M)
        v_next = v_mem + dv

        if v_next >= (V_THR * FP_SCALE):
            spike_steps.append(i)
            v_mem = V_RESET * FP_SCALE
            refr_cnt = REFRACT
        else:
            v_mem = v_next

    return spike_steps


def ref_lif_spike_times_float(steps):
    # Python-like floating-point model with DUT-matched parameters.
    # Use dt=1 step, tau_m=200 steps, tref=40 steps, constant input I=21.
    dt = 1.0
    tc_m = 200.0
    tref = 40.0
    vrest = -60.0
    vreset = -65.0
    vthr = -40.0
    I_in = 21.0

    v = vreset
    tlast = -1e9  # ensure no refractory at t=0
    spike_steps = []

    for i in range(steps):
        dv = (vrest - v + I_in) / tc_m
        if (dt * i) > (tlast + tref):
            v = v + dv * dt

        s = 1 if v >= vthr else 0
        if s == 1:
            tlast = dt * i
            spike_steps.append(i)
            v = vreset  # reset after spike (no peak in DUT)

    return spike_steps


async def lif_step(dut):
    dut.tick.value = 1
    await RisingEdge(dut.clk)
    dut.tick.value = 0
    spike_seen = False
    for _ in range(20):
        await RisingEdge(dut.clk)
        if int(dut.spike_pulse.value) == 1:
            spike_seen = True
    return spike_seen


@cocotb.test()
async def lif_count_test_fixed_point(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    await reset_and_start(dut)

    steps = 200
    spike_steps = []
    for _ in range(steps):
        if await lif_step(dut):
            spike_steps.append(_)

    expected_steps = ref_lif_spike_times_fixed(steps)
    assert spike_steps == expected_steps, f"spike timing mismatch: exp {expected_steps} got {spike_steps}"
    assert int(dut.spike_count.value) == len(expected_steps), "spike_count does not match spike timings"


@cocotb.test()
async def lif_count_test_python_model(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    await reset_and_start(dut)

    steps = 200
    spike_steps = []
    for i in range(steps):
        if await lif_step(dut):
            spike_steps.append(i)

    expected_steps = ref_lif_spike_times_float(steps)
    assert spike_steps == expected_steps, f"python-model spike timing mismatch: exp {expected_steps} got {spike_steps}"
    assert int(dut.spike_count.value) == len(expected_steps), "spike_count does not match python-model timings"


def lif_runner():
    from cocotb.runner import get_runner
    sim = os.getenv("SIM", "icarus")
    repo_root = Path(__file__).resolve().parents[1]
    hdl_dir = repo_root / "hdl"

    sv_sources = [str(p) for p in hdl_dir.glob("**/*.sv")]

    runner = get_runner(sim)
    runner.build(
        sources=sv_sources,
        hdl_toplevel="lif",
        parameters={},
        timescale=("1ns", "1ps"),
        waves=True,
        always=True,
        build_dir=str(repo_root / "sim" / "sim_build_lif"),
    )
    runner.test(
        hdl_toplevel="lif",
        test_module=Path(__file__).stem,
        seed=None,
    )


if __name__ == "__main__":
    lif_runner()
