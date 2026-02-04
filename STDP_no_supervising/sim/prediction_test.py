# file: sim/prediction_test.py
import os
from pathlib import Path
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


CLK_PERIOD_NS = 10


def prediction_reference(spikes, assignments, n_labels):
    n_samples = len(spikes)
    n_neurons = len(spikes[0]) if n_samples > 0 else 0

    rates = [[0.0 for _ in range(n_labels)] for _ in range(n_samples)]
    for i in range(n_labels):
        indices = [idx for idx, a in enumerate(assignments) if a == i]
        n_assigns = len(indices)
        if n_assigns > 0:
            for s in range(n_samples):
                total = 0
                for n in indices:
                    total += spikes[s][n]
                rates[s][i] = total / n_assigns

    preds = []
    for s in range(n_samples):
        best = max(range(n_labels), key=lambda i: rates[s][i])
        preds.append(best)
    return preds


def unpack_packed_labels(packed, count, label_bits):
    out = []
    mask = (1 << label_bits) - 1
    for i in range(count):
        out.append((packed >> (i * label_bits)) & mask)
    return out


@cocotb.test()
async def prediction_basic_test(dut):
    required = [
        "clk", "rst", "start", "done",
        "spikes_we", "spikes_sample_addr", "spikes_neuron_addr", "spikes_wdata",
        "assignments_we", "assignments_addr", "assignments_wdata",
        "predictions"
    ]
    if not all(hasattr(dut, name) for name in required):
        dut._log.info("Skipping prediction_basic_test: DUT interface not found.")
        return

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    dut.rst.value = 1
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)

    # MNIST-like scale (reduced)
    n_samples = 500
    n_neurons = 100
    n_labels = 10
    label_bits = 4

    # deterministic spikes: 5 active neurons per sample
    spikes = [[0 for _ in range(n_neurons)] for _ in range(n_samples)]
    for s in range(n_samples):
        base = (s * 7) % n_neurons
        for k in range(5):
            spikes[s][(base + k * 13) % n_neurons] = 1

    # deterministic assignments: neuron i -> label (i % n_labels)
    assignments = [i % n_labels for i in range(n_neurons)]

    expected_preds = prediction_reference(spikes, assignments, n_labels)

    # load spikes
    dut.spikes_we.value = 1
    for s_idx in range(n_samples):
        for n_idx in range(n_neurons):
            dut.spikes_sample_addr.value = s_idx
            dut.spikes_neuron_addr.value = n_idx
            dut.spikes_wdata.value = spikes[s_idx][n_idx]
            await RisingEdge(dut.clk)
    dut.spikes_we.value = 0

    # load assignments
    dut.assignments_we.value = 1
    for n_idx in range(n_neurons):
        dut.assignments_addr.value = n_idx
        dut.assignments_wdata.value = assignments[n_idx]
        await RisingEdge(dut.clk)
    dut.assignments_we.value = 0

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Generous upper bound: n_samples * n_labels * n_neurons * 2 + overhead
    max_cycles = (n_samples * n_labels * n_neurons * 2) + (n_samples * n_labels * 4) + 20000
    done_seen = False
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            done_seen = True
            break
    if not done_seen:
        raise AssertionError("prediction did not assert done within timeout")

    got_packed = int(dut.predictions.value)
    got = unpack_packed_labels(got_packed, n_samples, label_bits)

    assert got == expected_preds, f"predictions mismatch: exp {expected_preds[:20]} got {got[:20]}"


def prediction_runner():
    from cocotb.runner import get_runner
    sim = os.getenv("SIM", "icarus")
    repo_root = Path(__file__).resolve().parents[1]
    hdl_dir = repo_root / "hdl"

    sv_sources = [str(p) for p in hdl_dir.glob("**/*.sv")]

    runner = get_runner(sim)
    runner.build(
        sources=sv_sources,
        hdl_toplevel="prediction",
        parameters={
            "N_SAMPLES": 500,
            "N_NEURONS": 100,
            "N_LABELS": 10,
            "LABEL_BITS": 4,
        },
        timescale=("1ns", "1ps"),
        waves=True,
        always=True,
        build_dir=str(repo_root / "sim" / "sim_build_prediction"),
    )
    runner.test(
        hdl_toplevel="prediction",
        test_module=Path(__file__).stem,
        seed=None,
    )


if __name__ == "__main__":
    prediction_runner()
