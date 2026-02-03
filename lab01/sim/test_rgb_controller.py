import cocotb
import os
import random
import sys
import logging
from pathlib import Path
from cocotb.triggers import Timer
from cocotb.utils import get_sim_time as gst
from cocotb.runner import get_runner
test_file = os.path.basename(__file__).replace(".py","")
 
async def generate_clock(clock_wire):
	while True: # repeat forever
		clock_wire.value = 0
		await Timer(5,units="ns")
		clock_wire.value = 1
		await Timer(5,units="ns")
 
@cocotb.test()
async def first_test(dut):
    """First cocotb test?"""
    await cocotb.start( generate_clock( dut.clk ) ) #launches clock
    dut.rst.value = 1;
    dut.r_in.value = 100
    dut.g_in.value = 10
    dut.b_in.value = 220
    await Timer(20, "ns")
    dut.rst.value = 0; #rst is off...let it run
    await Timer(11400, "ns")# run for many cycles

    dut.r_in.value = 20
    dut.g_in.value = 150
    dut.b_in.value = 40
    await Timer(10000, "ns")
 
"""the code below here should largely remain unchanged in structure, though the specific files and things
specified will get updated for different simulations.
"""
def rgb_runner():
    """Simulate the counter using the Python runner."""
    hdl_toplevel_lang = os.getenv("HDL_TOPLEVEL_LANG", "verilog")
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent
    sys.path.append(str(proj_path / "sim" / "model"))
    sources = [proj_path / "hdl" / "rgb_controller.sv"] #grow/modify this as needed.
    sources += [proj_path / "hdl" / "counter.sv"] #grow/modify this as needed.
    sources += [proj_path / "hdl" / "pwm.sv"] #grow/modify this as needed.
    hdl_toplevel="rgb_controller"
    build_test_args = ["-Wall"]#,"COCOTB_RESOLVE_X=ZEROS"]
    parameters = {}
    sys.path.append(str(proj_path / "sim"))
    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel=hdl_toplevel,
        always=True,
        build_args=build_test_args,
        parameters=parameters,
        timescale = ('1ns','1ps'),
        waves=True
    )
    run_test_args = []
    runner.test(
        hdl_toplevel=hdl_toplevel,
        test_module=test_file,
        test_args=run_test_args,
        waves=True
    )
 
if __name__ == "__main__":
    rgb_runner()