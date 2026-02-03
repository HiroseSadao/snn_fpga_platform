# file: sim/lif_test.py
import os
from pathlib import Path
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    V_REST = -65
    V_RESET = -65
    INIT_VTHR = -52
    V_PEAK = 20
    TAU_M = 200
    REFRACT = 40
    TC_THETA = 10000
    THETA_MAX = 35
    THETA_PLUS_FP = 3277  # 0.05 in S16.16
    E_EXC = 0
    E_INH = -100
    G_EXC_FP = 21134  # 0.3230769 in S16.16
    G_INH_FP = 0

    v_mem = V_RESET * FP_SCALE
    vthr = INIT_VTHR * FP_SCALE
    theta = 0
    refr_cnt = 0
    spike_steps = []

    for i in range(steps):
        if refr_cnt != 0:
            refr_cnt -= 1
            v_mem = V_RESET * FP_SCALE
            theta = theta - int(theta / TC_THETA)
        else:
            i_syn_exc = (G_EXC_FP * ((E_EXC * FP_SCALE) - v_mem)) >> FP_SHIFT
            i_syn_inh = (G_INH_FP * ((E_INH * FP_SCALE) - v_mem)) >> FP_SHIFT
            num = (V_REST * FP_SCALE) - v_mem + i_syn_exc + i_syn_inh
            dv = int(num / TAU_M)
            v_next = v_mem + dv

            if v_next >= vthr:
                spike_steps.append(i)
                v_mem = V_RESET * FP_SCALE
                refr_cnt = REFRACT
                theta = theta - int(theta / TC_THETA) + THETA_PLUS_FP
            else:
                v_mem = v_next
                theta = theta - int(theta / TC_THETA)

        if theta < 0:
            theta = 0
        if theta > (THETA_MAX * FP_SCALE):
            theta = THETA_MAX * FP_SCALE
        vthr = (INIT_VTHR * FP_SCALE) + theta

    return spike_steps


def ref_lif_spike_times_float(steps):
    # Python-like floating-point model with DUT-matched parameters.
    dt = 1.0
    tc_m = 200.0
    tref = 40.0
    vrest = -65.0
    vreset = -65.0
    init_vthr = -52.0
    vpeak = 20.0
    theta_plus = 0.05
    theta_max = 35.0
    tc_theta = 10000.0
    e_exc = 0.0
    e_inh = -100.0
    g_exc = 0.3230769
    g_inh = 0.0

    v = vreset
    tlast = -1e9
    tcount = 0
    theta = 0.0
    vthr = init_vthr
    spike_steps = []

    for i in range(steps):
        I_synExc = g_exc * (e_exc - v)
        I_synInh = g_inh * (e_inh - v)
        dv = (vrest - v + I_synExc + I_synInh) / tc_m
        if (dt * tcount) > (tlast + tref):
            v = v + dv * dt

        s = 1 if v >= vthr else 0
        theta = (1 - dt / tc_theta) * theta + theta_plus * s
        theta = min(max(theta, 0.0), theta_max)
        vthr = theta + init_vthr
        if s == 1:
            tlast = dt * tcount
            spike_steps.append(i)
            v = vpeak
            v = vreset
        tcount += 1

    return spike_steps


async def lif_step(dut):
    dut.tick.value = 1
    await RisingEdge(dut.clk)
    dut.tick.value = 0
    spike_seen = False
    for _ in range(50):
        await RisingEdge(dut.clk)
        if int(dut.spike_pulse.value) == 1:
            spike_seen = True
    return spike_seen


@cocotb.test()
async def lif_count_test_fixed_point(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    await reset_and_start(dut)

    target_spikes = 10
    steps = 0
    spike_steps = []
    while len(spike_steps) < target_spikes:
        if await lif_step(dut):
            spike_steps.append(steps)
        steps += 1

    expected_steps = ref_lif_spike_times_fixed(steps)
    assert spike_steps == expected_steps, f"spike timing mismatch: exp {expected_steps} got {spike_steps}"
    assert int(dut.spike_count.value) == len(expected_steps), "spike_count does not match spike timings"


@cocotb.test()
async def lif_count_test_python_model(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    await reset_and_start(dut)

    target_spikes = 10
    steps = 0
    spike_steps = []
    while len(spike_steps) < target_spikes:
        if await lif_step(dut):
            spike_steps.append(steps)
        steps += 1

    expected_steps = ref_lif_spike_times_float(steps)
    assert spike_steps == expected_steps, f"python-model spike timing mismatch: exp {expected_steps} got {spike_steps}"
    assert int(dut.spike_count.value) == len(expected_steps), "spike_count does not match python-model timings"


@cocotb.test()
async def lif_theta_vthr_plot_test(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset_and_start(dut)

    target_spikes = 10
    steps = 0
    spikes = 0
    theta_vals = []
    vthr_vals = []
    time_vals = []

    while spikes < target_spikes:
        if await lif_step(dut):
            spikes += 1
        theta = int(dut.theta_out.value) / (1 << 16)
        vthr = int(dut.vthr_out.value) / (1 << 16)
        theta_vals.append(theta)
        vthr_vals.append(vthr)
        time_vals.append(steps)
        steps += 1

    repo_root = Path(__file__).resolve().parents[1]
    out_theta = repo_root / "sim" / "lif_theta.png"
    out_vthr = repo_root / "sim" / "lif_vthr.png"

    plt.figure(figsize=(6, 4))
    plt.plot(time_vals, theta_vals, label="theta")
    plt.xlabel("time (steps)")
    plt.ylabel("theta")
    plt.tight_layout()
    plt.savefig(out_theta)

    plt.figure(figsize=(6, 4))
    plt.plot(time_vals, vthr_vals, label="vthr")
    plt.xlabel("time (steps)")
    plt.ylabel("vthr")
    plt.tight_layout()
    plt.savefig(out_vthr)


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
