from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RunConfig:
    mode: str = "train-then-infer"
    train_samples: int = 100
    infer_samples: int = 20
    port: str = "COM7"
    seed: int = 0x12345678
    timeout_sec: float = 180.0
    chunk_nsteps: int = 16
    train_e2e_mine_timing: bool = False


@dataclass(frozen=True)
class ArtifactConfig:
    output_path: str = "ctrl/demo_result.json"


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "mnist"
    source: str = "fpga"
    start_lba: int = 2048


@dataclass(frozen=True)
class NeuronGroup:
    name: str
    size: int
    role: str
    neuron_model: str
    threshold: str
    reset: str


@dataclass(frozen=True)
class SynapseGroup:
    name: str
    src: str
    dst: str
    pattern: str
    learning_rule: str
    delay_steps: int = 0


@dataclass
class SNNModel:
    name: str
    dataset: DatasetConfig
    run: RunConfig
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    n_classes: int = 10
    input_size: int = 784
    neuron_groups: list[NeuronGroup] = field(default_factory=list)
    synapse_groups: list[SynapseGroup] = field(default_factory=list)

    def add_neuron_group(
        self,
        *,
        name: str,
        size: int,
        role: str,
        neuron_model: str,
        threshold: str,
        reset: str,
    ) -> NeuronGroup:
        group = NeuronGroup(
            name=name,
            size=int(size),
            role=role,
            neuron_model=neuron_model,
            threshold=threshold,
            reset=reset,
        )
        self.neuron_groups.append(group)
        return group

    def add_synapse_group(
        self,
        *,
        name: str,
        src: str,
        dst: str,
        pattern: str,
        learning_rule: str,
        delay_steps: int = 0,
    ) -> SynapseGroup:
        syn = SynapseGroup(
            name=name,
            src=src,
            dst=dst,
            pattern=pattern,
            learning_rule=learning_rule,
            delay_steps=int(delay_steps),
        )
        self.synapse_groups.append(syn)
        return syn


def create_fixed_mnist_stdp_model(
    *,
    name: str = "diehl_cook_fpga_demo",
    port: str = "COM7",
    start_lba: int = 2048,
    train_samples: int = 500,
    infer_samples: int = 100,
    timeout_sec: float = 600.0,
    seed: int = 0x12345678,
    output_path: str = "ctrl/demo_result.json",
) -> SNNModel:
    model = SNNModel(
        name=name,
        dataset=DatasetConfig(name="mnist", source="fpga", start_lba=int(start_lba)),
        run=RunConfig(
            mode="train-then-infer",
            train_samples=int(train_samples),
            infer_samples=int(infer_samples),
            port=port,
            seed=int(seed),
            timeout_sec=float(timeout_sec),
        ),
        artifacts=ArtifactConfig(output_path=output_path),
        n_classes=10,
        input_size=784,
    )
    model.add_neuron_group(
        name="input",
        size=784,
        role="input",
        neuron_model="poisson_input",
        threshold="external_spike",
        reset="none",
    )
    model.add_neuron_group(
        name="exc",
        size=50,
        role="excitatory",
        neuron_model="diehl_cook_exc",
        threshold="v > v_thresh_base + theta",
        reset="v = v_reset; theta += theta_plus",
    )
    model.add_neuron_group(
        name="inh",
        size=50,
        role="inhibitory",
        neuron_model="conductance_inh",
        threshold="v > v_thresh",
        reset="v = v_reset",
    )
    model.add_synapse_group(
        name="input_to_exc",
        src="input",
        dst="exc",
        pattern="dense",
        learning_rule="stdp_input_exc",
        delay_steps=5,
    )
    model.add_synapse_group(
        name="exc_to_inh",
        src="exc",
        dst="inh",
        pattern="one_to_one",
        learning_rule="fixed_exc_inh",
        delay_steps=2,
    )
    model.add_synapse_group(
        name="inh_to_exc",
        src="inh",
        dst="exc",
        pattern="all_but_self",
        learning_rule="fixed_inh_exc",
        delay_steps=0,
    )
    return model
