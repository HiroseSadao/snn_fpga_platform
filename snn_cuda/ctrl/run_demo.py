from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
from pathlib import Path

from demo_compiler import DemoModelSpec, compile_demo_spec, model_spec_from_model
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
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile and run the FPGA SNN demo.")
    parser.add_argument("--model", required=True, help="Python model file exporting build_model() or MODEL.")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--emit-legacy-cli", action="store_true")
    return parser.parse_args()


def build_model_spec(args: argparse.Namespace) -> DemoModelSpec:
    model = _load_model_from_path(args.model)
    return model_spec_from_model(model, source_kind="python_model")


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
