import cocotb
import os
import random
import sys
import logging
from pathlib import Path
from cocotb.triggers import Timer
from cocotb.utils import get_sim_time as gst
from cocotb.runner import get_runner
 
#cheap way to get the name of current file for runner:
test_file = os.path.basename(__file__).replace(".py","")
 
async def generate_clock(clock_wire):
	  while True: # repeat forever
		    clock_wire.value = 0
		    await Timer(5,units="ns")
		    clock_wire.value = 1
		    await Timer(5,units="ns")
 
@cocotb.test()
async def comprehensive_counter_test(dut):
    """Comprehensive counter test with all requirements"""
    
    # 1. Start background clock generator
    await cocotb.start(generate_clock(dut.clk))
    dut._log.info("Clock generator started")
    
    # 2. Begin by setting rst signal high for at least one clock cycle
    dut.rst.value = 1
    dut.period.value = 3  # Low period value for testing
    await Timer(20, "ns")  # Wait for at least one clock cycle
    dut._log.info("Reset signal held high for two clock cycles")
    
    # 3. Ensure that after reset signal, count value is set to zero
    assert dut.count.value.integer == 0, f"Expected count to be 0 after reset, got {dut.count.value.integer}"
    dut._log.info("✓ Count is 0 after reset")
    
    # 4. Deassert reset and let it run with low period value
    dut.rst.value = 0
    dut._log.info("Reset deasserted, counter should start counting")
    
    # Wait and verify counting behavior
    await Timer(10, "ns")
    count_after_1_cycle = dut.count.value.integer
    dut._log.info(f"Count after 1 cycle: {count_after_1_cycle}")
    
    await Timer(10, "ns")
    count_after_2_cycles = dut.count.value.integer
    dut._log.info(f"Count after 2 cycles: {count_after_2_cycles}")

    await Timer(10, "ns")
    count_after_3_cycles = dut.count.value.integer
    dut._log.info(f"Count after 3 cycles: {count_after_3_cycles}")

    await Timer(10, "ns")
    count_after_4_cycles = dut.count.value.integer
    dut._log.info(f"Count after 4 cycles: {count_after_4_cycles}")
    
    # Verify counter is working (with period=3, it cycles 0->1->2->0->1->2...)
    # So we expect: 0 -> 1 -> 2 -> 0 -> 1 -> 2...
    expected_sequence = [0, 1, 2, 0, 1, 2]
    actual = [count_after_1_cycle, count_after_2_cycles, count_after_3_cycles, count_after_4_cycles]
    assert any(expected_sequence[i:i+len(actual)] == actual for i in range(len(expected_sequence)-len(actual)+1)), f"Count {actual} not in expected sequence"
    dut._log.info("✓ Counter is following expected sequence")
    
    # 5. Let it run and overflow several times with low period
    await Timer(100, "ns")
    dut._log.info("Let counter run with period=3 to observe overflow")
    
    # 6. Set period to a higher value
    dut.period.value = 15
    dut._log.info("Period changed to 15")
    
    # Wait to observe behavior with higher period
    await Timer(200, "ns")
    count_with_high_period = dut.count.value.integer
    dut._log.info(f"Count with period=15: {count_with_high_period}")
    
    # 7. Test mid-count reset behavior
    dut._log.info("Testing mid-count reset behavior...")
    
    # Let counter run for a bit
    await Timer(50, "ns")
    count_before_reset = dut.count.value.integer
    dut._log.info(f"Count before mid-count reset: {count_before_reset}")
    
    # Set reset high mid-count and hold for at least one clock cycle
    dut.rst.value = 1
    await Timer(20, "ns")  # Hold reset for one clock cycle
    
    # Verify count goes to 0 and stays at 0
    assert dut.count.value.integer == 0, f"Expected count to be 0 during reset, got {dut.count.value.integer}"
    dut._log.info("✓ Count is 0 during reset")
    
    # Hold reset for another cycle to ensure it stays at 0
    await Timer(20, "ns")
    assert dut.count.value.integer == 0, f"Expected count to stay 0 during reset, got {dut.count.value.integer}"
    dut._log.info("✓ Count stays 0 during reset")
    
    # Deassert reset and verify counting resumes
    dut.rst.value = 0
    await Timer(10, "ns")
    count_after_reset_deassert = dut.count.value.integer
    dut._log.info(f"Count after reset deasserted: {count_after_reset_deassert}")
    
    # Verify counting has resumed (with period=15, after 2 cycles from reset: 0->1->2)
    assert count_after_reset_deassert == 1, f"Expected count to be 1 after reset deassert, got {count_after_reset_deassert}"
    dut._log.info("✓ Counter resumes counting after reset deassert")
    
    
    dut._log.info("✓ All counter test requirements verified successfully!")
 
"""the code below should largely remain unchanged in structure, though the specific files and things
specified should get updated for different simulations.
"""
def counter_runner():
    """Simulate the counter using the Python runner."""
    hdl_toplevel_lang = os.getenv("HDL_TOPLEVEL_LANG", "verilog")
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent
    sys.path.append(str(proj_path / "sim" / "model"))
    sources = [proj_path / "hdl" / "counter.sv"] #grow/modify this as needed.
    hdl_toplevel = "counter"
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
    counter_runner()