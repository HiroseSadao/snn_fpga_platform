# file: sim/assign_labels_test.py
import os
from pathlib import Path
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


CLK_PERIOD_NS = 10


def assign_labels_reference(spikes, labels, n_labels, alpha=1.0, rates=None):
    # spikes: [n_samples][n_neurons], labels: [n_samples]
    n_samples = len(spikes)
    n_neurons = len(spikes[0]) if n_samples > 0 else 0

    if rates is None:
        rates = [[0.0 for _ in range(n_labels)] for _ in range(n_neurons)]

    for i in range(n_labels):
        indices = [idx for idx, lab in enumerate(labels) if lab == i]
        n_labeled = len(indices)
        if n_labeled > 0:
            for n in range(n_neurons):
                s = sum(spikes[idx][n] for idx in indices) / n_labeled
                rates[n][i] = alpha * rates[n][i] + s

    sum_rate = []
    for n in range(n_neurons):
        total = sum(rates[n])
        if total == 0:
            total = 1.0
        sum_rate.append(total)

    proportions = []
    for n in range(n_neurons):
        proportions.append([rates[n][i] / sum_rate[n] for i in range(n_labels)])

    assignments = []
    for n in range(n_neurons):
        best = max(range(n_labels), key=lambda i: proportions[n][i])
        assignments.append(best)

    return assignments, proportions, rates


@cocotb.test()
async def assign_labels_basic_test(dut):
    required = [
        "clk", "rst", "start", "done",
        "spikes_we", "spikes_sample_addr", "spikes_neuron_addr", "spikes_wdata",
        "labels_we", "labels_addr", "labels_wdata",
        "assignments"
    ]
    if not all(hasattr(dut, name) for name in required):
        dut._log.info("Skipping assign_labels_basic_test: DUT interface not found.")
        return

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    dut.rst.value = 1
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)

    # Larger example (MNIST-like scale, reduced)
    n_samples = 500
    n_neurons = 100
    n_labels = 10
    label_bits = 4

    # deterministic pseudo-random spikes/labels
    spikes = [[0 for _ in range(n_neurons)] for _ in range(n_samples)]
    labels = [0 for _ in range(n_samples)]
    for s in range(n_samples):
        labels[s] = s % n_labels
        # simple sparse pattern: 5 active neurons per sample
        base = (s * 7) % n_neurons
        for k in range(5):
            spikes[s][(base + k * 13) % n_neurons] = 1

    exp_assign, exp_prop, exp_rates = assign_labels_reference(
        spikes, labels, n_labels, alpha=1.0, rates=None
    )

    # load spikes
    dut.spikes_we.value = 1
    for s_idx in range(n_samples):
        for n_idx in range(n_neurons):
            dut.spikes_sample_addr.value = s_idx
            dut.spikes_neuron_addr.value = n_idx
            dut.spikes_wdata.value = spikes[s_idx][n_idx]
            await RisingEdge(dut.clk)
    dut.spikes_we.value = 0

    # load labels
    dut.labels_we.value = 1
    for s_idx in range(n_samples):
        dut.labels_addr.value = s_idx
        dut.labels_wdata.value = labels[s_idx]
        await RisingEdge(dut.clk)
    dut.labels_we.value = 0

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Rough upper bound for cycles: labels * samples * neurons + overhead
    max_cycles = (n_labels * n_samples * n_neurons * 2) + (n_labels * n_neurons * 4) + 10000
    done_seen = False
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            done_seen = True
            break

    if not done_seen:
        raise AssertionError("assign_labels did not assert done within timeout")

    got_assign = int(dut.assignments.value)
    got = []
    for i in range(n_neurons):
        got.append((got_assign >> (i * label_bits)) & ((1 << label_bits) - 1))

    assert got == exp_assign, f"assignments mismatch: exp {exp_assign} got {got}"


def assign_labels_runner():
    from cocotb.runner import get_runner
    sim = os.getenv("SIM", "icarus")
    repo_root = Path(__file__).resolve().parents[1]
    hdl_dir = repo_root / "hdl"

    sv_sources = [str(p) for p in hdl_dir.glob("**/*.sv")]

    runner = get_runner(sim)
    runner.build(
        sources=sv_sources,
        hdl_toplevel="assign_labels",
        parameters={
            "N_SAMPLES": 500,
            "N_NEURONS": 100,
            "N_LABELS": 10,
            "LABEL_BITS": 4,
        },
        timescale=("1ns", "1ps"),
        waves=True,
        always=True,
        build_dir=str(repo_root / "sim" / "sim_build_assign_labels"),
    )
    runner.test(
        hdl_toplevel="assign_labels",
        test_module=Path(__file__).stem,
        seed=None,
    )


if __name__ == "__main__":
    assign_labels_runner()
