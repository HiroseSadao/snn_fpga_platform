from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
from pathlib import Path

from demo_compiler import DemoModelSpec, compile_demo_spec, model_from_config, model_spec_from_config, model_spec_from_model
from demo_framework import DemoRunner, FpgaDemoBackend

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def _resolve_input_path(path: str) -> Path:
    raw = Path(path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(SCRIPT_DIR / raw)
        candidates.append(PROJECT_DIR / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def _resolve_output_path(path: str) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return (Path.cwd() / raw).resolve()


def _load_config(path: str | None) -> dict[str, object]:
    if not path:
        return {}
    config_path = _resolve_input_path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a JSON object: {config_path}")
    return data


def _load_model_from_path(path: str):
    model_path = _resolve_input_path(path)
    spec = importlib.util.spec_from_file_location("demo_model_module", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import model module: {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "build_model"):
        return module.build_model()
    if hasattr(module, "MODEL"):
        return module.MODEL
    raise ValueError(f"model module must export build_model() or MODEL: {model_path}")


def _flat_cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "mode": args.mode,
        "port": args.port,
        "start_lba": args.start_lba,
        "seed": args.seed,
        "timeout": args.timeout,
        "train_samples": args.train_samples,
        "infer_samples": args.infer_samples,
        "input_source": args.input_source,
        "chunk_nsteps": args.chunk_nsteps,
        "train_e2e_mine_timing": args.train_e2e_mine_timing,
        "output": args.output,
        "n_in": args.n_in,
        "n_neurons": args.n_neurons,
        "n_classes": args.n_classes,
    }


def _merge_config(config: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    merged = dict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, remaining = pre_parser.parse_known_args()
    config = _load_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Compile and run the FPGA SNN demo.")
    parser.add_argument("--config", type=str, default=pre_args.config)
    parser.add_argument("--model", type=str, default=None, help="Python model file exporting build_model() or MODEL.")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--start-lba", type=int, default=None)
    parser.add_argument("--seed", type=lambda x: int(x, 0), default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--infer-samples", type=int, default=None)
    parser.add_argument("--mode", choices=("train-then-infer", "train-only"), default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--input-source", choices=("fpga",), default=None)
    parser.add_argument("--chunk-nsteps", type=int, default=None)
    parser.add_argument("--train-e2e-mine-timing", action="store_true", default=None)
    parser.add_argument("--n-in", type=int, default=None)
    parser.add_argument("--n-neurons", type=int, default=None)
    parser.add_argument("--n-classes", type=int, default=None)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--emit-legacy-cli", action="store_true")
    args = parser.parse_args(remaining)
    args._config = config
    return args


def build_model_spec(args: argparse.Namespace) -> DemoModelSpec:
    if args.model:
        model = _load_model_from_path(args.model)
        return model_spec_from_model(model, source_kind="python_model")
    merged = _merge_config(args._config, _flat_cli_overrides(args))
    return model_spec_from_config(merged)


def write_result(path: str, payload: dict[str, object]) -> None:
    out_path = _resolve_output_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def print_plan(plan: dict[str, object]) -> None:
    print(json.dumps(plan, indent=2, sort_keys=True))


def print_legacy_cli(legacy_cli: list[str]) -> None:
    print("Equivalent legacy command:")
    print(" ".join(shlex.quote(part) for part in legacy_cli))


def main() -> None:
    args = parse_args()
    spec = build_model_spec(args)
    plan = compile_demo_spec(spec)

    print(
        "Demo compile: "
        f"mode={spec.execution.mode}, total_samples={plan.expected_total_samples}, "
        f"input={spec.dataset.source}, source={spec.source_kind}"
    )

    if args.print_plan or args.compile_only:
        print_plan(plan.to_dict())

    if args.emit_legacy_cli or args.compile_only:
        print_legacy_cli(plan.legacy_cli)

    if args.compile_only:
        return

    print(
        "Demo start: "
        f"train_samples={spec.execution.train_samples}, infer_samples={spec.execution.infer_samples}, "
        f"port={spec.execution.port}"
    )

    runner = DemoRunner(FpgaDemoBackend(), plan)
    try:
        session, result = runner.run()
        print(f"Train kernel caps: 0x{session.caps:08X}")
    except Exception as exc:
        result = {
            "status": "error",
            "compiler_plan": plan.to_dict(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        write_result(spec.artifacts.output_path, result)
        print(f"Demo result saved to {spec.artifacts.output_path}")
        raise

    write_result(spec.artifacts.output_path, result)
    print(f"Demo result saved to {spec.artifacts.output_path}")


if __name__ == "__main__":
    main()
