from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace

import use_fpga_calc as fpga

from demo_ir import ModelIR, lower_model_to_ir
from snn_api import ArtifactConfig, DatasetConfig, RunConfig, SNNModel

RAW1_BYTES_PER_IMAGE = 784


@dataclass(frozen=True)
class NetworkSpec:
    name: str
    n_in: int
    n_neurons: int
    n_classes: int


@dataclass(frozen=True)
class DemoModelSpec:
    network: NetworkSpec
    dataset: DatasetConfig
    execution: RunConfig
    artifacts: ArtifactConfig
    ir: ModelIR
    source_kind: str


@dataclass(frozen=True)
class RuntimeStep:
    kind: str
    description: str


@dataclass(frozen=True)
class DemoCompilePlan:
    spec: DemoModelSpec
    batch_train_then_infer: bool
    batch_train_only: bool
    expected_total_samples: int
    expected_image_bytes: int
    expected_image_sectors: int
    estimated_timeout_sec: float
    legacy_cli: list[str]
    runtime_steps: tuple[RuntimeStep, ...]

    def to_runtime_args(self) -> SimpleNamespace:
        exe = self.spec.execution
        ds = self.spec.dataset
        return SimpleNamespace(
            image_source=ds.source,
            port=exe.port,
            start_lba=int(ds.start_lba),
            seed=int(exe.seed),
            timeout=float(exe.timeout_sec),
            train_then_infer_train_samples=int(exe.train_samples),
            train_then_infer_infer_samples=int(exe.infer_samples),
            chunk_nsteps=int(exe.chunk_nsteps),
            train_e2e_mine_timing=bool(exe.train_e2e_mine_timing),
            batch_control_smoke=False,
            batch_single_infer_smoke=False,
            batch_single_train_smoke=False,
            sample_idx=0,
            batch_num_samples=1,
            batch_train_then_infer=bool(self.batch_train_then_infer),
            batch_train_only=bool(self.batch_train_only),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": {
                "network": asdict(self.spec.network),
                "dataset": asdict(self.spec.dataset),
                "execution": asdict(self.spec.execution),
                "artifacts": asdict(self.spec.artifacts),
                "source_kind": self.spec.source_kind,
                "ir": self.spec.ir.to_dict(),
            },
            "batch_train_then_infer": bool(self.batch_train_then_infer),
            "batch_train_only": bool(self.batch_train_only),
            "expected_total_samples": int(self.expected_total_samples),
            "expected_image_bytes": int(self.expected_image_bytes),
            "expected_image_sectors": int(self.expected_image_sectors),
            "estimated_timeout_sec": float(self.estimated_timeout_sec),
            "legacy_cli": list(self.legacy_cli),
            "runtime_steps": [asdict(step) for step in self.runtime_steps],
        }


def model_spec_from_model(model: SNNModel, *, source_kind: str) -> DemoModelSpec:
    n_neurons = 0
    for group in model.neuron_groups:
        if group.role == "excitatory":
            n_neurons = int(group.size)
            break
    spec = DemoModelSpec(
        network=NetworkSpec(
            name=model.name,
            n_in=int(model.input_size),
            n_neurons=int(n_neurons),
            n_classes=int(model.n_classes),
        ),
        dataset=model.dataset,
        execution=model.run,
        artifacts=model.artifacts,
        ir=lower_model_to_ir(model),
        source_kind=source_kind,
    )
    return spec

def _build_runtime_steps(mode: str) -> tuple[RuntimeStep, ...]:
    steps = [
        RuntimeStep("query_caps", "Query FPGA training kernel capabilities."),
        RuntimeStep("lower_to_fixed_ir", "Lower the Python SNN model into the fixed FPGA-oriented IR."),
        RuntimeStep("preload_labels", "Upload label metadata used by the batch engine."),
        RuntimeStep("train_batch", "Run the FPGA batch training phase."),
    ]
    if mode == "train-then-infer":
        steps.extend(
            [
                RuntimeStep("derive_assignments", "Convert aggregated spike statistics into neuron-label assignments."),
                RuntimeStep("preload_assignments", "Upload the compiled neuron-label assignments."),
                RuntimeStep("infer_batch", "Run the FPGA batch inference phase."),
                RuntimeStep("collect_metrics", "Read back accuracy and timing counters."),
            ]
        )
    else:
        steps.append(RuntimeStep("collect_metrics", "Read back training timing counters and label accumulation stats."))
    return tuple(steps)


def _build_legacy_cli(spec: DemoModelSpec) -> list[str]:
    exe = spec.execution
    ds = spec.dataset
    cmd = [
        "python",
        "ctrl/use_fpga_calc.py",
        "--port",
        exe.port,
        "--start-lba",
        str(ds.start_lba),
        "--timeout",
        str(exe.timeout_sec),
        "--seed",
        hex(exe.seed),
    ]
    if exe.mode == "train-only":
        cmd.append("--batch-train-only")
    else:
        cmd.append("--batch-train-then-infer")
    cmd.extend(
        [
            "--train-then-infer-train-samples",
            str(exe.train_samples),
            "--train-then-infer-infer-samples",
            str(exe.infer_samples),
        ]
    )
    if exe.train_e2e_mine_timing:
        cmd.append("--train-e2e-mine-timing")
    return cmd


def _validate_fixed_backend(spec: DemoModelSpec) -> None:
    net = spec.network
    ds = spec.dataset
    exe = spec.execution
    ir = spec.ir

    if exe.mode not in ("train-then-infer", "train-only"):
        raise ValueError(f"unsupported demo mode: {exe.mode}")
    if ds.source != "fpga":
        raise ValueError(f"unsupported input source: {ds.source}")
    if ds.name != "mnist":
        raise ValueError(f"unsupported dataset: {ds.name}")
    if net.n_in != fpga.N_IN:
        raise ValueError(f"network.n_in={net.n_in} does not match FPGA build N_IN={fpga.N_IN}")
    if net.n_neurons != fpga.N_NEURONS:
        raise ValueError(
            f"network.n_neurons={net.n_neurons} does not match FPGA build N_NEURONS={fpga.N_NEURONS}"
        )
    if net.n_classes != 10:
        raise ValueError("network.n_classes must be 10 for the current MNIST demo")
    if len(ir.groups) != 3:
        raise ValueError("fixed FPGA backend expects exactly 3 neuron groups")
    if len(ir.connections) != 3:
        raise ValueError("fixed FPGA backend expects exactly 3 synapse groups")
    group_names = {group.name for group in ir.groups}
    if group_names != {"input", "exc", "inh"}:
        raise ValueError(f"fixed FPGA backend expects input/exc/inh groups, got {sorted(group_names)}")
    conn_names = {conn.name for conn in ir.connections}
    expected_conn = {"input_to_exc", "exc_to_inh", "inh_to_exc"}
    if conn_names != expected_conn:
        raise ValueError(f"fixed FPGA backend expects {sorted(expected_conn)}, got {sorted(conn_names)}")
    if exe.train_samples <= 0:
        raise ValueError("execution.train_samples must be positive")
    if exe.mode == "train-then-infer" and exe.infer_samples <= 0:
        raise ValueError("execution.infer_samples must be positive for train-then-infer mode")
    if exe.chunk_nsteps <= 0:
        raise ValueError("execution.chunk_nsteps must be positive")
    if ds.start_lba < 0:
        raise ValueError("dataset.start_lba must be non-negative")
    if exe.timeout_sec <= 0.0:
        raise ValueError("execution.timeout_sec must be positive")


def compile_demo_spec(spec: DemoModelSpec) -> DemoCompilePlan:
    _validate_fixed_backend(spec)

    exe = spec.execution
    batch_train_then_infer = exe.mode == "train-then-infer"
    batch_train_only = exe.mode == "train-only"
    expected_total_samples = int(exe.train_samples) + (int(exe.infer_samples) if batch_train_then_infer else 0)
    expected_image_bytes = expected_total_samples * RAW1_BYTES_PER_IMAGE
    start_byte = int(fpga.RAW1_HEADER_BYTES)
    end_byte = start_byte + expected_image_bytes
    sectors_per_image_span = (end_byte + 511) // 512
    estimated_timeout_sec = max(float(exe.timeout_sec), 60.0 if batch_train_then_infer else 30.0)

    return DemoCompilePlan(
        spec=spec,
        batch_train_then_infer=batch_train_then_infer,
        batch_train_only=batch_train_only,
        expected_total_samples=expected_total_samples,
        expected_image_bytes=expected_image_bytes,
        expected_image_sectors=sectors_per_image_span,
        estimated_timeout_sec=estimated_timeout_sec,
        legacy_cli=_build_legacy_cli(spec),
        runtime_steps=_build_runtime_steps(exe.mode),
    )
