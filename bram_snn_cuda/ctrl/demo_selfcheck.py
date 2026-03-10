from __future__ import annotations

import json
from pathlib import Path

from demo_compiler import compile_demo_spec, model_from_config, model_spec_from_model
from mnist_stdp_fixed_demo import build_model


def _check_plan(label: str, plan) -> None:
    assert plan.spec.network.n_in == 784
    assert plan.spec.network.n_neurons == 50
    assert plan.spec.execution.mode in ("train-then-infer", "train-only")
    assert plan.expected_total_samples == (
        plan.spec.execution.train_samples
        + (plan.spec.execution.infer_samples if plan.spec.execution.mode == "train-then-infer" else 0)
    )
    assert plan.expected_image_bytes == plan.expected_total_samples * 784
    assert any("--batch-train-then-infer" == arg or "--batch-train-only" == arg for arg in plan.legacy_cli)
    assert plan.spec.ir.backend_name == "fixed_mnist_fpga"
    print(f"demo_selfcheck[{label}]: ok")


def main() -> None:
    config_path = Path(__file__).resolve().parent / "demo_config.json"
    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    spec_from_config = model_spec_from_model(model_from_config(config), source_kind="config")
    plan_from_config = compile_demo_spec(spec_from_config)
    _check_plan("config", plan_from_config)

    spec_from_model = model_spec_from_model(build_model(), source_kind="python_model")
    plan_from_model = compile_demo_spec(spec_from_model)
    _check_plan("python_model", plan_from_model)

    print(json.dumps(plan_from_model.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
