from brian2 import *
from collections import defaultdict
from pathlib import Path
from random import randrange, seed as rseed
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


# Switch between "train", "observe", and "test".
# Use "plot" to plot the confusion matrix.
MODE = "test"

# Number of training, observation, and testing samples
N_TRAIN = 25_000
N_OBSERVE = 2_000
N_TEST = 1_000

# Random seed value
SEED = 42

# Storage path
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
DATA_PATH = BASE_DIR / "data"

# Number of weight save points
N_SAVE_POINTS = 100

# Don't change these values unless you know what you're doing.
N_INP = 784
N_NEURONS = 400
V_EXC_REST = -65 * mV
V_INH_REST = -60 * mV
INTENSITY = 2

# Weights of exc->inh and inh->exc synapses
W_EXC_INH = 10.4
W_INH_EXC = 17.0


def load_mnist(training):
    try:
        from torchvision import datasets
        from torchvision import transforms

        ds = datasets.MNIST(
            root="./data",
            train=training,
            download=True,
            transform=transforms.ToTensor(),
        )
        images = np.stack([np.array(ds[i][0]).squeeze() for i in range(len(ds))])
        labels = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int64)
        return images, labels
    except Exception:
        pass

    try:
        from tensorflow.keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return (x_train, y_train) if training else (x_test, y_test)
    except Exception as exc:
        raise RuntimeError(
            "MNIST loading failed. Install torchvision or tensorflow, "
            "or provide your own MNIST loader."
        ) from exc


def save_npy(arr, path):
    arr = np.array(arr)
    print("%-9s %-15s => %-30s" % ("Saving", arr.shape, path))
    np.save(path, arr)


def load_npy(path):
    arr = np.load(path)
    print("%-9s %-30s => %-15s" % ("Loading", path, arr.shape))
    return arr


def build_network(training):
    eqs = """
    dv/dt = (v_rest - v + i_exc + i_inh) / tau_mem  : volt (unless refractory)
    i_exc = ge * -v                         : volt
    i_inh = gi * (v_inh_base - v)           : volt
    dge/dt = -ge/(1 * ms)                   : 1
    dgi/dt = -gi/(2 * ms)                   : 1
    dtimer/dt = 1                           : second
    """
    reset = "v = %r; timer = 0 * ms" % V_EXC_REST
    if training:
        exc_eqs = eqs + """
        dtheta/dt = -theta / (1e7 * ms)         : volt
        """
        arr_theta = np.ones(N_NEURONS) * 20 * mV
        reset += "; theta += 0.05 * mV"
    else:
        exc_eqs = eqs + """
        theta                                   : volt
        """
        arr_theta = load_npy(DATA_PATH / "theta.npy") * volt
    exc_eqs = Equations(
        exc_eqs,
        tau_mem=100 * ms,
        v_rest=V_EXC_REST,
        v_inh_base=-100 * mV,
    )

    ng_exc = NeuronGroup(
        N_NEURONS,
        exc_eqs,
        threshold="v > (theta - 72 * mV) and (timer > 50 * ms)",
        refractory=5 * ms,
        reset=reset,
        method="euler",
        name="exc",
    )
    ng_exc.v = V_EXC_REST
    ng_exc.theta = arr_theta

    inh_eqs = Equations(
        eqs,
        tau_mem=10 * ms,
        v_rest=V_INH_REST,
        v_inh_base=-85 * mV,
    )
    ng_inh = NeuronGroup(
        N_NEURONS,
        inh_eqs,
        threshold="v > -40 * mV",
        refractory=2 * ms,
        reset="v = -45 * mV",
        method="euler",
        name="inh",
    )
    ng_inh.v = V_INH_REST

    syns_exc_inh = Synapses(ng_exc, ng_inh, on_pre="ge_post += %f" % W_EXC_INH)
    syns_exc_inh.connect(j="i")

    syns_inh_exc = Synapses(ng_inh, ng_exc, on_pre="gi_post += %f" % W_INH_EXC)
    syns_inh_exc.connect("i != j")

    pg_inp = PoissonGroup(N_INP, 0 * Hz, name="inp")

    model = "w : 1"
    on_post = ""
    on_pre = "ge_post += w"
    if training:
        on_pre += "; pre = 1.; w = clip(w - 0.0001 * post1, 0, 1.0)"
        on_post += (
            "post2bef = post2; w = clip(w + 0.01 * pre * post2bef, 0, 1.0); "
            "post1 = 1.; post2 = 1."
        )
        model += """
        post2bef                        : 1
        dpre/dt   = -pre/(20 * ms)      : 1 (event-driven)
        dpost1/dt = -post1/(20 * ms)    : 1 (event-driven)
        dpost2/dt = -post2/(40 * ms)    : 1 (event-driven)
        """
        weights = (np.random.random(N_INP * N_NEURONS) + 0.01) * 0.3
    else:
        weights = load_npy(DATA_PATH / "weights.npy")

    syns_inp_exc = Synapses(
        pg_inp,
        ng_exc,
        model=model,
        on_pre=on_pre,
        on_post=on_post,
        name="inp_exc",
    )
    syns_inp_exc.connect(True)
    syns_inp_exc.delay = "rand() * 10 * ms"
    syns_inp_exc.w = weights

    exc_mon = SpikeMonitor(ng_exc, name="sp_exc")
    net = Network(
        [pg_inp, ng_exc, ng_inh, syns_inp_exc, syns_exc_inh, syns_inh_exc, exc_mon]
    )
    net.run(0 * ms)
    return net


def show_sample(net, sample, intensity):
    exc_mon = net["sp_exc"]
    prev = exc_mon.count[:]
    net["inp"].rates = sample * intensity * Hz
    net.run(350 * ms)
    next = exc_mon.count[:]
    net["inp"].rates = 0 * Hz
    net.run(150 * ms)
    pat = next - prev
    cnt = np.sum(pat)
    if cnt < 5:
        return show_sample(net, sample, intensity + 1)
    return pat


def predict(groups, rates):
    return np.argmax([rates[grp].mean() for grp in groups])


def normalize_plastic_weights(syns):
    conns = np.reshape(syns.w, (N_INP, N_NEURONS))
    col_sums = np.sum(conns, axis=0)
    factors = 78.0 / col_sums
    conns *= factors
    syns.w = conns.reshape(-1)


def stats(net):
    tick = defaultclock.timestep[:]
    cnt = np.sum(net["sp_exc"].count[:])

    inp_exc = net["inp_exc"]
    w_mu = np.mean(inp_exc.w)
    w_std = np.std(inp_exc.w)

    exc = net["exc"]
    theta = exc.theta / mV
    theta_mu = np.mean(theta)
    theta_sig = np.std(theta)
    return [tick, cnt, w_mu, w_std, theta_mu, theta_sig]


def iter_range(n):
    if tqdm is not None:
        return tqdm(range(n))
    return range(n)


def train():
    X, Y = load_mnist(True)
    X = X.reshape(X.shape[0], -1) / 8.0
    n_samples = X.shape[0]
    net = build_network(True)
    rows = [stats(net) + [-1]]
    w_hist = [np.array(net["inp_exc"].w)]

    ratio = max(N_TRAIN // N_SAVE_POINTS, 1)
    for i in iter_range(N_TRAIN):
        ix = i % n_samples
        normalize_plastic_weights(net["inp_exc"])
        show_sample(net, X[ix], INTENSITY)
        rows.append(stats(net) + [Y[ix]])
        if i % ratio == 0:
            w_hist.append(np.array(net["inp_exc"].w))

    save_npy(rows, DATA_PATH / "train_stats.npy")
    save_npy(w_hist, DATA_PATH / "train_w_hist.npy")
    save_npy(net["inp_exc"].w, DATA_PATH / "weights.npy")
    save_npy(net["exc"].theta, DATA_PATH / "theta.npy")


def observe():
    X, Y = load_mnist(True)
    X = X.reshape(X.shape[0], -1) / 8.0
    n_samples = X.shape[0]
    net = build_network(False)
    rows = [stats(net) + [-1]]
    responses = defaultdict(list)

    for i in iter_range(N_OBSERVE):
        ix = i % n_samples
        sample = X[ix]
        cls = Y[ix]
        exc = show_sample(net, sample, INTENSITY)
        rows.append(stats(net) + [Y[ix]])
        responses[cls].append(exc)

    res = np.zeros((10, N_NEURONS))
    for cls, vals in responses.items():
        res[cls] = np.array(vals).mean(axis=0)

    assign = np.argmax(res, axis=0)
    save_npy(assign, DATA_PATH / "assign.npy")
    save_npy(rows, DATA_PATH / "observe_stats.npy")


def test():
    conf = np.zeros((10, 10))
    assign = np.load(DATA_PATH / "assign.npy")
    groups = [np.where(assign == i)[0] for i in range(10)]

    X, Y = load_mnist(False)
    X = X.reshape(X.shape[0], -1) / 8.0
    net = build_network(False)
    for _ in iter_range(N_TEST):
        ix = randrange(len(X))
        exc = show_sample(net, X[ix], INTENSITY)
        guess = predict(groups, exc)
        real = Y[ix]
        conf[real, guess] += 1

    print("Accuracy: %6.3f" % (np.trace(conf) / np.sum(conf)))
    conf = conf / conf.sum(axis=1)[:, None]
    print(np.around(conf, 2))
    save_npy(conf, DATA_PATH / "confusion.npy")


def plot():
    conf = np.load(DATA_PATH / "confusion.npy")
    import matplotlib.pyplot as plt

    plt.imshow(100 * conf, interpolation="nearest", cmap=plt.cm.Blues)
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        if conf[i, j] == 0:
            continue
        plt.text(
            j,
            i,
            f"{round(100*conf[i, j])}%",
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if conf[i, j] > 0.5 else "black",
        )
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()


if __name__ == "__main__":
    seed(SEED)
    rseed(SEED)
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    cmds = dict(train=train, observe=observe, test=test, plot=plot)

    def ensure_files_for_mode(mode):
        weights_path = DATA_PATH / "weights.npy"
        theta_path = DATA_PATH / "theta.npy"
        assign_path = DATA_PATH / "assign.npy"
        conf_path = DATA_PATH / "confusion.npy"

        if mode == "train":
            return
        if mode == "observe":
            if not weights_path.exists() or not theta_path.exists():
                print("weights/theta not found. Running train() first...")
                train()
            return
        if mode == "test":
            if not assign_path.exists():
                if not weights_path.exists() or not theta_path.exists():
                    print("weights/theta not found. Running train() first...")
                    train()
                print("assign not found. Running observe() first...")
                observe()
            return
        if mode == "plot":
            if not conf_path.exists():
                if not assign_path.exists():
                    if not weights_path.exists() or not theta_path.exists():
                        print("weights/theta not found. Running train() first...")
                        train()
                    print("assign not found. Running observe() first...")
                    observe()
                print("confusion not found. Running test() first...")
                test()

    ensure_files_for_mode(MODE)
    cmds[MODE]()
