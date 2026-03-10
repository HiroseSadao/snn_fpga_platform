from __future__ import annotations

import argparse
from types import SimpleNamespace

import use_fpga_calc as fpga


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FPGA SNN demo through a simplified wrapper."
    )
    parser.add_argument(
        "--port",
        type=str,
        default=fpga.SERIAL_PORTNAME,
        help="Serial port connected to the FPGA board.",
    )
    parser.add_argument(
        "--start-lba",
        type=int,
        default=2048,
        help="Start LBA of the MNIST raw image region on SD.",
    )
    parser.add_argument(
        "--seed",
        type=lambda x: int(x, 0),
        default=0x12345678,
        help="Seed forwarded to the FPGA batch engine.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Batch operation timeout in seconds.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=100,
        help="Number of samples used for the training phase.",
    )
    parser.add_argument(
        "--infer-samples",
        type=int,
        default=20,
        help="Number of samples used for the inference phase.",
    )
    parser.add_argument(
        "--mode",
        choices=("train-then-infer", "train-only"),
        default="train-then-infer",
        help="Demo mode to run.",
    )
    return parser.parse_args()


def build_fpga_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        image_source="fpga",
        port=args.port,
        start_lba=int(args.start_lba),
        seed=int(args.seed),
        timeout=float(args.timeout),
        train_then_infer_train_samples=int(args.train_samples),
        train_then_infer_infer_samples=int(args.infer_samples),
        chunk_nsteps=16,
        train_e2e_mine_timing=False,
        batch_control_smoke=False,
        batch_single_infer_smoke=False,
        batch_single_train_smoke=False,
        sample_idx=0,
        batch_num_samples=1,
        batch_train_then_infer=(args.mode == "train-then-infer"),
        batch_train_only=(args.mode == "train-only"),
    )


def main() -> None:
    args = parse_args()
    demo_args = build_fpga_args(args)

    if fpga.serial is None:
        raise RuntimeError(
            "pyserial is not installed. Install it to use FPGA communication paths."
        )

    print(
        "Demo start: "
        f"mode={args.mode}, train_samples={args.train_samples}, infer_samples={args.infer_samples}"
    )
    print(f"Opening serial port {args.port}")

    with fpga.serial.Serial(
        args.port,
        fpga.BAUD,
        timeout=fpga.TIMEOUT_SEC,
        write_timeout=fpga.WRITE_TIMEOUT_SEC,
    ) as ser:
        fpga.time.sleep(0.05)
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass

        caps = fpga.fpga_train_query_caps(ser)
        print(f"Train kernel caps: 0x{caps:08X}")

        if args.mode == "train-only":
            fpga.fpga_batch_train_only(ser, demo_args)
        else:
            fpga.fpga_batch_train_then_infer(ser, demo_args)


if __name__ == "__main__":
    main()
