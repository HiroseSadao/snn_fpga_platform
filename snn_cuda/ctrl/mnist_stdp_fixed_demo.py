from __future__ import annotations

from snn_api import create_fixed_mnist_stdp_model


def build_model():
    return create_fixed_mnist_stdp_model(
        name="mnist_stdp_fixed_demo",
        port="COM7",
        start_lba=2048,
        train_samples=1000,
        infer_samples=100,
        timeout_sec=7200.0,
        seed=0x12345678,
        train_seed=0x12345678,
        infer_seed=0x12345679,
        output_path="ctrl/demo_result.json",
    )


MODEL = build_model()
