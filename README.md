# snn_cuda

Fixed-backend FPGA SNN demo prototype.

This repository currently demonstrates the following flow:

`Python SNN model -> compiler/IR -> fixed FPGA backend -> FPGA execution`

The FPGA side is fixed to the current `top_level.sv` batch train/infer pipeline. The Python side provides a small model API, a simple IR, a compiler that lowers the model to a fixed FPGA-oriented execution plan, and a runtime that executes that plan through UART.

## Main Files

- `hdl/top_level.sv`
  - Fixed FPGA execution pipeline for the current MNIST/STDP-style SNN demo.
- `ctrl/use_fpga_calc.py`
  - Low-level UART runtime and FPGA command implementation.
- `ctrl/snn_api.py`
  - Python API for describing the current fixed SNN model.
- `ctrl/demo_ir.py`
  - Minimal intermediate representation used by the demo compiler.
- `ctrl/demo_compiler.py`
  - Lowers a Python model into a fixed FPGA execution plan.
- `ctrl/demo_framework.py`
  - Runtime/backend layer that executes the compiled plan on FPGA.
- `ctrl/run_demo.py`
  - Main entry point for compiling and running the demo.
- `ctrl/mnist_stdp_fixed_demo.py`
  - Example Python model used by the current demo.

## Usage

Compile only:

```bash
python ctrl/run_demo.py --model ctrl/mnist_stdp_fixed_demo.py --compile-only
```

Run the demo on FPGA:

```bash
python ctrl/run_demo.py --model ctrl/mnist_stdp_fixed_demo.py
```

Run the compiler self-check:

```bash
python ctrl/demo_selfcheck.py
```

## Output

Running the demo writes:

- `ctrl/demo_result.json`

This includes the compiler plan, backend name, timing counters, and inference metrics such as `correct` and `accuracy`.

## Scope

This is a prototype, not a general SNN framework.

- The supported model structure is currently fixed to:
  - `input`
  - `exc`
  - `inh`
  - `input_to_exc`
  - `exc_to_inh`
  - `inh_to_exc`
- The backend is fixed to the current FPGA design.
- The goal is to demonstrate the core project flow from Python model description to FPGA execution.
