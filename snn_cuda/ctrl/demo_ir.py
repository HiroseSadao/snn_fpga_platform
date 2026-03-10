from __future__ import annotations

from dataclasses import asdict, dataclass

from snn_api import SNNModel


@dataclass(frozen=True)
class GroupIR:
    name: str
    role: str
    size: int
    neuron_model: str


@dataclass(frozen=True)
class ConnectionIR:
    name: str
    src: str
    dst: str
    pattern: str
    learning_rule: str
    delay_steps: int


@dataclass(frozen=True)
class LearningIR:
    mode: str
    train_enabled: bool
    assignment_source: str


@dataclass(frozen=True)
class ExecutionIR:
    mode: str
    phases: tuple[str, ...]
    train_samples: int
    infer_samples: int
    chunk_nsteps: int


@dataclass(frozen=True)
class MemoryRegionIR:
    name: str
    kind: str
    logical_size: int
    note: str


@dataclass(frozen=True)
class ModelIR:
    model_name: str
    groups: tuple[GroupIR, ...]
    connections: tuple[ConnectionIR, ...]
    learning: LearningIR
    execution: ExecutionIR
    memory_regions: tuple[MemoryRegionIR, ...]
    backend_name: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def lower_model_to_ir(model: SNNModel) -> ModelIR:
    phases = ["query_caps", "preload_labels", "train_batch"]
    if model.run.mode == "train-then-infer":
        phases.extend(["derive_assignments", "preload_assignments", "infer_batch", "collect_metrics"])
    else:
        phases.append("collect_metrics")

    groups = tuple(
        GroupIR(
            name=group.name,
            role=group.role,
            size=int(group.size),
            neuron_model=group.neuron_model,
        )
        for group in model.neuron_groups
    )
    connections = tuple(
        ConnectionIR(
            name=syn.name,
            src=syn.src,
            dst=syn.dst,
            pattern=syn.pattern,
            learning_rule=syn.learning_rule,
            delay_steps=int(syn.delay_steps),
        )
        for syn in model.synapse_groups
    )
    memory_regions = (
        MemoryRegionIR("weights", "synapse_state", model.input_size * 50, "input->exc dense weights"),
        MemoryRegionIR("theta", "neuron_state", 50, "exc adaptive thresholds"),
        MemoryRegionIR("v_state", "neuron_state", 50, "exc membrane state"),
        MemoryRegionIR("label_stats", "summary_state", 10 * 50, "aggregated label spike sums"),
    )
    learning = LearningIR(
        mode="local_stdp",
        train_enabled=True,
        assignment_source="aggregated_label_stats",
    )
    execution = ExecutionIR(
        mode=model.run.mode,
        phases=tuple(phases),
        train_samples=int(model.run.train_samples),
        infer_samples=int(model.run.infer_samples),
        chunk_nsteps=int(model.run.chunk_nsteps),
    )
    return ModelIR(
        model_name=model.name,
        groups=groups,
        connections=connections,
        learning=learning,
        execution=execution,
        memory_regions=memory_regions,
        backend_name="fixed_mnist_fpga",
    )
