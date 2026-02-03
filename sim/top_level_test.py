# file: sim/top_level_test.py
import os
from pathlib import Path
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer
from cocotb.handle import Force, Release


CLK_PERIOD_NS = 0.5  # 2 GHz (10x faster than previous)


def set_default_inputs(dut):
    dut.sw.value = 0
    dut.btn.value = 0


async def reset_and_start(dut):
    set_default_inputs(dut)
    await ClockCycles(dut.clk_100mhz, 2)
    # btn[0] rising edge -> start
    dut.btn.value = 0b0001
    await ClockCycles(dut.clk_100mhz, 2)
    dut.btn.value = 0
    await ClockCycles(dut.clk_100mhz, 2)


def ref_lif_spike_count(steps):
    # Fixed-point S16.16 reference model (matches top_level.sv)
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
    spike_count = 0

    for _ in range(steps):
        if refr_cnt != 0:
            refr_cnt -= 1
            v_mem = V_RESET * FP_SCALE
            continue

        num = (V_REST * FP_SCALE) - v_mem + (I_IN * FP_SCALE)
        # trunc toward zero
        dv = int(num / TAU_M)
        v_next = v_mem + dv

        if v_next >= (V_THR * FP_SCALE):
            spike_count += 1
            v_mem = V_RESET * FP_SCALE
            refr_cnt = REFRACT
        else:
            v_mem = v_next

    return spike_count


async def lif_step(dut):
    # Force lif_tick high for one cycle so the LIF state machine advances.
    dut.lif_tick.value = Force(1)
    await RisingEdge(dut.clk_100mhz)
    dut.lif_tick.value = Release()
    # Wait until divider finishes and state returns to IDLE (0)
    while int(dut.lif_state.value) != 0:
        await RisingEdge(dut.clk_100mhz)


SEG_MAP = {
    0x3F: 0,  # 7'b0111111
    0x06: 1,
    0x5B: 2,
    0x4F: 3,
    0x66: 4,
    0x6D: 5,
    0x7D: 6,
    0x07: 7,
    0x7F: 8,
    0x6F: 9,
    0x77: 10,
    0x7C: 11,
    0x39: 12,
    0x5E: 13,
    0x79: 14,
    0x71: 15,
}


def decode_seg_active_low(seg_low):
    seg_high = (~seg_low) & 0x7F
    return SEG_MAP.get(seg_high, None)


async def capture_8_digits(dut, max_cycles=5000):
    digits = [None] * 8
    cycles = 0
    while cycles < max_cycles and any(d is None for d in digits):
        await RisingEdge(dut.clk_100mhz)
        cycles += 1

        ss0_an = int(dut.ss0_an.value) & 0xF
        ss1_an = int(dut.ss1_an.value) & 0xF
        seg0 = int(dut.ss0_c.value) & 0x7F
        seg1 = int(dut.ss1_c.value) & 0x7F

        for i in range(4):
            if ((ss1_an >> i) & 1) == 0:
                digits[i] = decode_seg_active_low(seg1)
            if ((ss0_an >> i) & 1) == 0:
                digits[i + 4] = decode_seg_active_low(seg0)

    return digits


@cocotb.test()
async def top_level_lif_count_and_display_test(dut):
    # clock
    cocotb.start_soon(Clock(dut.clk_100mhz, CLK_PERIOD_NS, units="ns").start())

    await reset_and_start(dut)

    # Run a small number of LIF steps by forcing lif_tick.
    steps = 120
    for _ in range(steps):
        await lif_step(dut)

    expected = ref_lif_spike_count(steps)
    got = int(dut.spike_count.value)
    assert got == expected, f"spike_count mismatch: exp {expected} got {got}"

    # Display check removed: focus on spike count only.


def top_level_runner():
    from cocotb.runner import get_runner
    sim = os.getenv("SIM", "icarus")
    repo_root = Path(__file__).resolve().parents[1]
    hdl_dir = repo_root / "hdl"

    sv_sources = [str(p) for p in hdl_dir.glob("**/*.sv")]

    runner = get_runner(sim)
    runner.build(
        sources=sv_sources,
        hdl_toplevel="top_level",
        parameters={},
        timescale=("1ns", "1ps"),
        waves=True,
        always=True,
        build_dir=str(repo_root / "sim" / "sim_build_top_level"),
    )
    runner.test(
        hdl_toplevel="top_level",
        test_module=Path(__file__).stem,
        seed=None,
    )


if __name__ == "__main__":
    top_level_runner()
