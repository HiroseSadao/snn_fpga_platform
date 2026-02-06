import cocotb
import os, sys
from pathlib import Path
from cocotb.clock import Clock
from cocotb.runner import get_runner
from cocotb.triggers import RisingEdge, ClockCycles

test_file = os.path.basename(__file__).replace(".py", "")

async def hw_reset(dut):
    dut.pixel_valid.value = 0
    dut.calculate.value = 0
    dut.pixel_x.value = 0
    dut.pixel_y.value = 0
    dut.rst.value = 1
    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)

async def feed_valid_pixel(dut, px, py):
    dut.pixel_x.value = int(px)
    dut.pixel_y.value = int(py)
    dut.pixel_valid.value = 1
    await RisingEdge(dut.clk)
    dut.pixel_valid.value = 0
    await RisingEdge(dut.clk)

async def trigger_calculate_and_wait_valid(dut, max_cycles=10000):
    dut.calculate.value = 1
    await RisingEdge(dut.clk)
    dut.calculate.value = 0
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.com_valid.value) == 1:
            await RisingEdge(dut.clk)
            return int(dut.com_x.value), int(dut.com_y.value)
    raise TimeoutError("com_valid did not assert within timeout window")

@cocotb.test()
async def com_single_pixel_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await hw_reset(dut)
    x, y = 123, 45
    await feed_valid_pixel(dut, x, y)
    cx, cy = await trigger_calculate_and_wait_valid(dut)
    assert cx == x and cy == y

@cocotb.test()
async def com_average_sweep_xy_same(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await hw_reset(dut)
    N = 359
    for i in range(N):
        await feed_valid_pixel(dut, i, i)
    cx, cy = await trigger_calculate_and_wait_valid(dut)
    exp_floor = (N - 1) // 2
    assert cx in {exp_floor, exp_floor - 1}
    assert cy in {exp_floor, exp_floor - 1}

@cocotb.test()
async def com_average_x_sweep_y_const(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await hw_reset(dut)
    N = 412
    for i in range(N):
        await feed_valid_pixel(dut, i, 10)
    cx, cy = await trigger_calculate_and_wait_valid(dut)
    exp_floor = (N - 1) // 2
    assert cx in {exp_floor, exp_floor - 1}
    assert cy == 10

@cocotb.test()
async def com_no_valid_pixels(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await hw_reset(dut)
    dut.calculate.value = 1
    await RisingEdge(dut.clk)
    dut.calculate.value = 0
    never_asserted = True
    for _ in range(2000):
        await RisingEdge(dut.clk)
        if int(dut.com_valid.value) == 1:
            never_asserted = False
            break
    assert never_asserted

@cocotb.test()
async def com_multi_frame_back_to_back(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await hw_reset(dut)
    for i in range(100):
        await feed_valid_pixel(dut, i, 100 + i)
    cx1, cy1 = await trigger_calculate_and_wait_valid(dut)
    assert cx1 in {49, 50}
    assert cy1 in {149, 150}

    for i in range(100):
        await feed_valid_pixel(dut, 200 + i, i)
    cx2, cy2 = await trigger_calculate_and_wait_valid(dut)
    assert cx2 in {249, 250}
    assert cy2 in {49, 50}

@cocotb.test()
async def com_stress_many_pixels(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await hw_reset(dut)
    for y in range(64):
        for x in range(64):
            await feed_valid_pixel(dut, x, 31)
    cx, cy = await trigger_calculate_and_wait_valid(dut)
    assert cx in {31, 32}
    assert cy == 31

def center_of_mass_runner():
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent
    sources = [
        proj_path / "hdl" / "divider.sv",
        proj_path / "hdl" / "center_of_mass.sv",
    ]
    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel="center_of_mass",
        parameters={},
        timescale=("1ns", "1ps"),
        waves=True,
        always=True,
    )
    runner.test(hdl_toplevel="center_of_mass", test_module=test_file, test_args=[])

if __name__ == "__main__":
    center_of_mass_runner()
