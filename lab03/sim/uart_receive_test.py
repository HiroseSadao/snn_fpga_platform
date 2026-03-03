import cocotb
import os
import random
import sys
from math import log
import logging
from pathlib import Path
from cocotb.clock import Clock
from cocotb.triggers import Timer, ClockCycles, RisingEdge, FallingEdge, ReadOnly,with_timeout
from cocotb.utils import get_sim_time as gst
from cocotb.runner import get_runner
test_file = os.path.basename(__file__).replace(".py","")

# utility function to reverse bits:
def reverse_bits(n,size):
    reversed_n = 0
    for i in range(size):
        reversed_n = (reversed_n << 1) | (n & 1)
        n >>= 1
    return reversed_n

# test uart message:
UART_RESP_MSG = 0x92
#flip them:
UART_RESP_MSG = reverse_bits(UART_RESP_MSG,8)

async def test_uart_device(dut, received_data_list):
  count = 0
  count_max = 16 #change for different sizes
  while True:
    await FallingEdge(dut.busy) #listen for falling busy
    dut.cipo.value = (SPI_RESP_MSG>>count)&0x1 #feed in lowest bit
    dut._log.info(f"SPI peripheral Device Sending: {dut.cipo.value}")
    count+=1
    count%=16
    while dut.cs.value.integer ==0:
      await RisingEdge(dut.dclk)
      bit = dut.copi.value.integer #grab value
      dut._log.info(f"SPI peripheral Device Receiving: {bit}")
      received_data_list.append(bit)
      dut._log.info(f"Received data list: {received_data_list}")
      await FallingEdge(dut.dclk)
      dut.cipo.value = (SPI_RESP_MSG>>count)&0x1 #feed in lowest bit
      dut._log.info(f"SPI peripheral Device Sending: {dut.cipo.value}")
      count+=1
      count%=16

@cocotb.test()
async def test_a(dut):
    """cocotb test for the UART module"""
    dut._log.info("Starting...")
    
    # 受信データを格納するリストを作成
    received_data_list = []
    
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    # cocotb.start_soon(test_uart_device(dut, received_data_list))
    dut._log.info("Holding reset...")
    dut.rst.value = 1
    dut.din.value = 1 #set in 16 bit input value
    await ClockCycles(dut.clk, 3) #wait three clock cycles
    await  FallingEdge(dut.clk)
    dut.rst.value = 0 #un reset device
    await ClockCycles(dut.clk, 3) #wait a few clock cycles
    await  FallingEdge(dut.clk)
    dut._log.info("Setting Trigger")
    dut.din.value = 0 # once trigger in is off, don't expect data_in to stay the same!!
    await ClockCycles(dut.clk, 10416)
    dut.din.value = 0
    await ClockCycles(dut.clk, 10416)
    dut.din.value = 1
    await ClockCycles(dut.clk, 10416)
    dut.din.value = 0
    await ClockCycles(dut.clk, 10416)
    dut.din.value = 1
    await ClockCycles(dut.clk, 10416)
    dut.din.value = 0
    await ClockCycles(dut.clk, 10416)
    dut.din.value = 1
    await ClockCycles(dut.clk, 10416)
    dut.din.value = 0
    await ClockCycles(dut.clk, 10416)
    dut.din.value = 1
    await ClockCycles(dut.clk, 10416)
    dut.din.value = 1
    await ClockCycles(dut.clk, 110000)
    dut.trigger.value = 0
    
    await ClockCycles(dut.clk, 300)

def uart_receive_runner():
    """Simulate the counter using the Python runner."""
    hdl_toplevel_lang = os.getenv("HDL_TOPLEVEL_LANG", "verilog")
    sim = os.getenv("SIM", "icarus")
    proj_path = Path(__file__).resolve().parent.parent
    sys.path.append(str(proj_path / "sim" / "model"))
    sources = [proj_path / "hdl" / "uart_receive.sv"]
    build_test_args = ["-Wall"]
    parameters = {'INPUT_CLOCK_FREQ': 100000000, 'BAUD_RATE': 9600} #!!!change these to do different versions
    sys.path.append(str(proj_path / "sim"))
    hdl_toplevel = "uart_receive"
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
    uart_receive_runner()