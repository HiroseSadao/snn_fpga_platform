# -*- coding: utf-8 -*-

import gzip
import urllib.request
from pathlib import Path

import numpy as np

try:
    from brian2 import (
        Equations,
        Network,
        NeuronGroup,
        SpikeGeneratorGroup,
        SpikeMonitor,
        Synapses,
        Hz,
        ms,
        mV,
        defaultclock,
        device,
        prefs,
        seed,
        set_device,
    )
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "This script requires brian2. Install it with `pip install brian2`."
    ) from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


SEED = 0
BACKEND = "cpp_standalone"
N_INP = 784
N_CLASSES = 10
N_NEURONS = 50
N_TRAIN = 2_000
N_TEST = 100
TRAIN_TIME = 350 * ms
REST_TIME = 150 * ms
BASE_INTENSITY = 2.0
W_EXC_INH = 10.4
W_INH_EXC = 17.0
ACTIVE_STEPS = 350
REST_STEPS = 150
SAMPLE_STEPS = ACTIVE_STEPS + REST_STEPS

BASE_DIR = Path(__file__).resolve().parent
MNIST_DIR = BASE_DIR / "mnist_data"
RESULT_DIR = BASE_DIR / "brian2_minimal_results"
MNIST_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)


def _iter_range(n):
    if tqdm is not None:
        return tqdm(range(n))
    return range(n)


def _download_mnist_file(filename, cache_dir=MNIST_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / filename
    if out_path.exists():
        return out_path

    last_error = None
    for base_url in MNIST_MIRRORS:
        try:
            url = base_url + filename
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, out_path)
            return out_path
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to download {filename} from all mirrors. Last error: {last_error}"
    )


def _load_idx_images(path):
    with gzip.open(path, "rb") as f:
        if int.from_bytes(f.read(4), "big") != 2051:
            raise ValueError(f"Invalid image file: {path}")
        n_images = int.from_bytes(f.read(4), "big")
        n_rows = int.from_bytes(f.read(4), "big")
        n_cols = int.from_bytes(f.read(4), "big")
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8).reshape(n_images, n_rows * n_cols)


def _load_idx_labels(path):
    with gzip.open(path, "rb") as f:
        if int.from_bytes(f.read(4), "big") != 2049:
            raise ValueError(f"Invalid label file: {path}")
        n_labels = int.from_bytes(f.read(4), "big")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if labels.size != n_labels:
        raise ValueError(f"Label count mismatch in {path}")
    return labels


def load_mnist(train=True):
    prefix = "train" if train else "t10k"
    images_path = _download_mnist_file(f"{prefix}-images-idx3-ubyte.gz")
    labels_path = _download_mnist_file(f"{prefix}-labels-idx1-ubyte.gz")
    images = _load_idx_images(images_path).astype(np.float32) / 8.0
    labels = _load_idx_labels(labels_path).astype(np.int64)
    return images, labels


def generate_input_spikes(images, intensity=BASE_INTENSITY, seed_offset=0):
    rng = np.random.default_rng(SEED + seed_offset)
    all_indices = []
    all_steps = []

    for sample_idx in _iter_range(len(images)):
        rates_hz = images[sample_idx] * intensity
        spike_prob = np.clip(rates_hz * 1e-3, 0.0, 1.0)
        spike_mask = rng.random((ACTIVE_STEPS, N_INP)) < spike_prob
        step_offsets, neuron_ids = np.nonzero(spike_mask)
        if neuron_ids.size == 0:
            continue

        step_times = sample_idx * SAMPLE_STEPS + step_offsets

        all_indices.append(neuron_ids.astype(np.int32))
        all_steps.append(step_times.astype(np.int32))

    if not all_indices:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    indices = np.concatenate(all_indices)
    steps = np.concatenate(all_steps)
    order = np.argsort(steps, kind="stable")
    return indices[order], steps[order]


def build_network(training, input_indices, input_steps, n_neurons=N_NEURONS,
                  initial_weights=None, initial_theta=None):
    inp = SpikeGeneratorGroup(
        N_INP,
        indices=input_indices,
        times=input_steps * ms,
        sorted=True,
        name="inp",
    )

    exc_eqs = Equations(
        """
        dv/dt = ((v_rest - v) + ge * (e_exc - v) + gi * (e_inh - v)) / tau_m : volt (unless refractory)
        dge/dt = -ge / tau_ge : 1
        dgi/dt = -gi / tau_gi : 1
        dtheta/dt = -theta / tau_theta : volt
        """
    )
    exc = NeuronGroup(
        n_neurons,
        exc_eqs,
        threshold="v > (v_thresh_base + theta)",
        reset="v = v_reset; theta += theta_plus",
        refractory=5 * ms,
        method="euler",
        name="exc",
    )
    exc.v = -65 * mV
    exc.theta = 0 * mV if initial_theta is None else initial_theta * mV
    exc.namespace.update(
        {
            "tau_m": 100 * ms,
            "tau_ge": 1 * ms,
            "tau_gi": 2 * ms,
            "tau_theta": 1e7 * ms,
            "v_rest": -65 * mV,
            "v_reset": -65 * mV,
            "v_thresh_base": -52 * mV,
            "theta_plus": 0.05 * mV if training else 0.0 * mV,
            "e_exc": 0 * mV,
            "e_inh": -100 * mV,
        }
    )

    inh_eqs = Equations(
        """
        dv/dt = ((v_rest - v) + ge * (e_exc - v)) / tau_m : volt (unless refractory)
        dge/dt = -ge / tau_ge : 1
        """
    )
    inh = NeuronGroup(
        n_neurons,
        inh_eqs,
        threshold="v > v_thresh",
        reset="v = v_reset",
        refractory=2 * ms,
        method="euler",
        name="inh",
    )
    inh.v = -60 * mV
    inh.namespace.update(
        {
            "tau_m": 10 * ms,
            "tau_ge": 1 * ms,
            "v_rest": -60 * mV,
            "v_reset": -45 * mV,
            "v_thresh": -40 * mV,
            "e_exc": 0 * mV,
        }
    )

    model = "w : 1"
    on_pre = "ge_post += w"
    on_post = ""
    if training:
        model += """
        dpre/dt = -pre / (20 * ms) : 1 (event-driven)
        dpost1/dt = -post1 / (20 * ms) : 1 (event-driven)
        dpost2/dt = -post2 / (40 * ms) : 1 (event-driven)
        post2_before : 1
        """
        on_pre += "; pre = 1.0; w = clip(w - 0.0001 * post1, 0.0, 1.0)"
        on_post = (
            "post2_before = post2; "
            "w = clip(w + 0.01 * pre * post2_before, 0.0, 1.0); "
            "post1 = 1.0; "
            "post2 = 1.0"
        )

    inp_exc = Synapses(
        inp,
        exc,
        model=model,
        on_pre=on_pre,
        on_post=on_post,
        method="euler",
        name="inp_exc",
    )
    inp_exc.connect()
    if initial_weights is None:
        weights = 0.3 * np.random.rand(N_INP * n_neurons)
        weights = weights.reshape(N_INP, n_neurons)
        col_sums = weights.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        weights *= 78.0 / col_sums
        inp_exc.w = weights.reshape(-1)
    else:
        inp_exc.w = initial_weights.reshape(-1)
    inp_exc.delay = "rand() * 10 * ms"

    exc_inh = Synapses(exc, inh, on_pre=f"ge_post += {W_EXC_INH}", name="exc_inh")
    exc_inh.connect(j="i")

    inh_exc = Synapses(inh, exc, on_pre=f"gi_post += {W_INH_EXC}", name="inh_exc")
    inh_exc.connect(condition="i != j")

    spikes = SpikeMonitor(exc, name="spikes")
    net = Network(inp, exc, inh, inp_exc, exc_inh, inh_exc, spikes)
    return net, inp_exc, exc, spikes


def collect_sample_responses(spike_monitor, n_samples):
    spike_steps = np.asarray(np.rint(spike_monitor.t[:] / ms), dtype=np.int64)
    spike_neurons = np.asarray(spike_monitor.i[:], dtype=np.int64)
    active_mask = (spike_steps % SAMPLE_STEPS) < ACTIVE_STEPS
    sample_ids = spike_steps // SAMPLE_STEPS
    valid_mask = active_mask & (sample_ids >= 0) & (sample_ids < n_samples)

    responses = np.zeros((n_samples, N_NEURONS), dtype=np.int16)
    np.add.at(
        responses,
        (sample_ids[valid_mask], spike_neurons[valid_mask]),
        1,
    )
    return responses


def assign_neuron_labels(responses, labels, n_classes=N_CLASSES):
    rates = np.zeros((responses.shape[1], n_classes), dtype=np.float32)
    for cls in range(n_classes):
        idx = np.where(labels == cls)[0]
        if idx.size:
            rates[:, cls] = responses[idx].mean(axis=0)
    return np.argmax(rates, axis=1)


def predict_labels(responses, assignments, n_classes=N_CLASSES):
    scores = np.zeros((responses.shape[0], n_classes), dtype=np.float32)
    for cls in range(n_classes):
        idx = np.where(assignments == cls)[0]
        if idx.size:
            scores[:, cls] = responses[:, idx].mean(axis=1)
    return np.argmax(scores, axis=1)


def _configure_backend():
    defaultclock.dt = 1 * ms
    if BACKEND == "cpp_standalone":
        set_device("cpp_standalone", build_on_run=False)
        prefs.codegen.target = "cpp_standalone"
    else:
        prefs.codegen.target = "numpy"


def _run_network(net, duration, build_dir=None):
    net.run(duration)
    if BACKEND == "cpp_standalone":
        device.build(directory=str(build_dir), compile=True, run=True, debug=False)


def _reset_cpp_device():
    if BACKEND == "cpp_standalone":
        device.reinit()
        device.activate(build_on_run=False)


def train_and_evaluate():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_x, train_y = load_mnist(train=True)
    test_x, test_y = load_mnist(train=False)
    train_x = train_x[:N_TRAIN]
    train_y = train_y[:N_TRAIN]
    test_x = test_x[:N_TEST]
    test_y = test_y[:N_TEST]

    print("Generating training input spikes...")
    train_indices, train_steps = generate_input_spikes(train_x, seed_offset=0)
    train_duration = len(train_x) * SAMPLE_STEPS * ms

    train_net, train_syn, train_exc, train_spikes = build_network(
        training=True,
        input_indices=train_indices,
        input_steps=train_steps,
    )
    _run_network(train_net, train_duration, RESULT_DIR / "cpp_train")

    train_responses = collect_sample_responses(train_spikes, len(train_x))
    assignments = assign_neuron_labels(train_responses, train_y)
    learned_weights = np.array(train_syn.w[:], dtype=np.float32)
    learned_theta = np.array(train_exc.theta[:] / mV, dtype=np.float32)

    np.save(RESULT_DIR / "weights.npy", learned_weights)
    np.save(RESULT_DIR / "theta.npy", learned_theta)
    np.save(RESULT_DIR / "assignments.npy", assignments)
    np.save(RESULT_DIR / "train_responses.npy", train_responses)

    _reset_cpp_device()
    _configure_backend()

    print("Generating test input spikes...")
    test_indices, test_steps = generate_input_spikes(test_x, seed_offset=1)
    test_duration = len(test_x) * SAMPLE_STEPS * ms

    test_net, _, _, test_spikes = build_network(
        training=False,
        input_indices=test_indices,
        input_steps=test_steps,
        initial_weights=learned_weights,
        initial_theta=learned_theta,
    )
    _run_network(test_net, test_duration, RESULT_DIR / "cpp_test")

    test_responses = collect_sample_responses(test_spikes, len(test_x))
    predicted = predict_labels(test_responses, assignments)
    accuracy = float(np.mean(predicted == test_y))

    np.save(RESULT_DIR / "test_predictions.npy", predicted)
    np.save(RESULT_DIR / "test_labels.npy", test_y)
    np.save(RESULT_DIR / "test_responses.npy", test_responses)
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    np.random.seed(SEED)
    seed(SEED)
    _configure_backend()
    train_and_evaluate()
