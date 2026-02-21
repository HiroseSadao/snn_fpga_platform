import os
from pathlib import Path
import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("This script requires PyTorch (torch).") from exc
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def load_mnist():
    try:
        from torchvision import datasets
        from torchvision import transforms

        ds = datasets.MNIST(
            root="./data",
            train=True,
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

        (x_train, y_train), _ = mnist.load_data()
        return x_train, y_train
    except Exception as exc:
        raise RuntimeError(
            "MNIST loading failed. Install torchvision or tensorflow, "
            "or provide your own MNIST loader."
        ) from exc


def online_load_and_encoding_dataset(dataset, i, dt, n_time, max_fr=32, norm=140):
    img = dataset[i].reshape(-1)
    fr_tmp = max_fr * norm / np.sum(img)
    fr = fr_tmp * np.repeat(np.expand_dims(img, axis=0), n_time, axis=0)
    input_spikes = np.where(np.random.rand(n_time, 784) < fr * dt, 1, 0)
    input_spikes = input_spikes.astype(np.uint8)
    return input_spikes


def online_load_and_encoding_dataset_torch(
    images_t, i, dt, n_time, max_fr=32, norm=140
):
    img = images_t[i].reshape(-1)
    fr_tmp = max_fr * norm / torch.sum(img)
    fr = fr_tmp * img
    rand = torch.rand((n_time, 784), device=img.device)
    input_spikes = (rand < (fr * dt)).to(torch.float32)
    return input_spikes


def _load_mem_file(path, w_bits=18):
    mask = (1 << w_bits) - 1
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            raw = int(s, 16) & mask
            if raw & (1 << (w_bits - 1)):
                raw -= (1 << w_bits)
            vals.append(raw)
    return np.array(vals, dtype=np.int32)


def load_initial_weights(n_neurons, n_in, data_dir):
    # Try to load 4-bank init files (w_init0..3) and concatenate
    files = [Path(data_dir) / f"w_init{i}.mem" for i in range(4)]
    if all(p.exists() for p in files):
        banks = [_load_mem_file(p) for p in files]
        total = sum(b.size for b in banks)
        if total == n_neurons * n_in:
            vals = np.concatenate(banks, axis=0)
            w = vals.astype(np.float32) / (1 << 16)
            return w.reshape(n_neurons, n_in)

    # Fallback: single file
    one = Path(data_dir) / "w_init0.mem"
    if one.exists():
        vals = _load_mem_file(one)
        if vals.size == n_neurons * n_in:
            w = vals.astype(np.float32) / (1 << 16)
            return w.reshape(n_neurons, n_in)

    # Final fallback: INIT_VAL = 66 (S2.16)
    return (np.full((n_neurons, n_in), 66, dtype=np.float32) / (1 << 16))


class LIFWTA_STDP:
    def __init__(self, n_in=784, n_neurons=100, device=None):
        self.n_in = n_in
        self.n_neurons = n_neurons
        self.device = device or torch.device("cpu")

        # Constants (match pipeline_small.sv)
        self.td_in_steps = 1
        self.td_exc_steps = 1
        self.td_inh_steps = 2
        self.td_x_steps = 20
        self.td_x2_steps = 40
        self.delay_in_steps = 11
        self.delay_e2i_steps = 2

        self.wexc = 2.25
        self.winh = 0.875
        self.wmin = 0.0
        self.wmax = 0.05
        self.a_p = 0.01
        self.a_m = 0.0001

        # Exc LIF params
        self.exc_vrest = -65.0
        self.exc_vreset = -65.0
        self.exc_init_vthr = -52.0
        self.exc_vpeak = 20.0
        self.exc_tau_m = 100.0
        self.exc_refract = 5
        self.exc_tc_theta = 10000000.0
        self.exc_theta_max = 35.0
        self.exc_theta_plus = 0.05
        self.exc_e_exc = 0.0
        self.exc_e_inh = -100.0

        # Inh LIF params
        self.inh_vrest = -60.0
        self.inh_vreset = -45.0
        self.inh_vthr = -40.0
        self.inh_vpeak = 20.0
        self.inh_tau_m = 10.0
        self.inh_refract = 2
        self.inh_e_exc = 0.0

        self.winh_div = self.winh / (self.n_neurons - 1) if self.n_neurons > 1 else 0.0

        # State tensors
        self.g_in_state = torch.zeros(self.n_neurons, device=self.device)
        self.r_exc = torch.zeros(self.n_neurons, device=self.device)
        self.x_exc = torch.zeros(self.n_neurons, device=self.device)
        self.x_exc2 = torch.zeros(self.n_neurons, device=self.device)
        self.g_inh_state = torch.zeros(self.n_neurons, device=self.device)
        self.r_inh = torch.zeros(self.n_neurons, device=self.device)
        self.v_exc = torch.full((self.n_neurons,), self.exc_vreset, device=self.device)
        self.v_inh = torch.full((self.n_neurons,), self.inh_vreset, device=self.device)
        self.theta = torch.zeros(self.n_neurons, device=self.device)
        self.vthr = torch.full((self.n_neurons,), self.exc_init_vthr, device=self.device)
        self.refr_exc = torch.zeros(self.n_neurons, dtype=torch.int32, device=self.device)
        self.refr_inh = torch.zeros(self.n_neurons, dtype=torch.int32, device=self.device)

        self.x_in = torch.zeros(self.n_in, device=self.device)
        self.delay_in_mem = torch.zeros(self.delay_in_steps, self.n_neurons, device=self.device)
        self.delay_e2i_mem = torch.zeros(self.delay_e2i_steps, self.n_neurons, device=self.device)
        self.delay_in_wr_idx = 0
        self.delay_e2i_wr_idx = 0

        # Precomputed factors
        self.inv_td_in = 1.0 / self.td_in_steps
        self.inv_td_exc = 1.0 / self.td_exc_steps
        self.inv_td_inh = 1.0 / self.td_inh_steps
        self.inv_td_x = 1.0 / self.td_x_steps
        self.inv_td_x2 = 1.0 / self.td_x2_steps

        self.exc_spike_amp = 1.0 / self.td_exc_steps
        self.inh_spike_amp = 1.0 / self.td_inh_steps

        self.input_spike_gain = 1.0 / self.td_in_steps

        self._neuron_indices = torch.arange(self.n_neurons, device=self.device)

        self.W = None

    def load_weights(self, w_numpy):
        self.W = torch.tensor(w_numpy, dtype=torch.float32, device=self.device)

    def step(self, s_in, stdp_en):
        # s_in: shape [n_in], 0/1 float or int
        s_in = s_in.to(self.device)
        s_in_f = s_in.float()

        # Update input trace x_in
        decayed_x_in = torch.clamp(self.x_in - self.x_in * self.inv_td_x, min=0.0)
        self.x_in = torch.where(s_in_f > 0, torch.ones_like(self.x_in), decayed_x_in)

        # Accumulate input conductance
        pre_idx = torch.nonzero(s_in_f, as_tuple=False).flatten()
        if pre_idx.numel() == 0:
            g_in_accum = torch.zeros(self.n_neurons, device=self.device)
        else:
            g_in_accum = self.W.index_select(1, pre_idx).sum(dim=1) * self.input_spike_gain

        g_in_state_next = self.g_in_state - self.g_in_state * self.inv_td_in + g_in_accum

        # Random delay for input conductance read
        if self.delay_in_steps == 1:
            delay_in_rd = self.delay_in_mem[0]
        else:
            delay_in_rand = torch.randint(
                0, self.delay_in_steps, (self.n_neurons,), device=self.device
            )
            rd_idx = torch.where(
                delay_in_rand == 0,
                (self.delay_in_wr_idx - 1) % self.delay_in_steps,
                (self.delay_in_wr_idx - delay_in_rand) % self.delay_in_steps,
            )
            delay_in_rd = self.delay_in_mem[rd_idx, self._neuron_indices]

        # Exc neuron dynamics
        i_syn_exc = delay_in_rd * (self.exc_e_exc - self.v_exc)
        i_syn_inh = self.g_inh_state * (self.exc_e_inh - self.v_exc)
        num_exc = (self.exc_vrest - self.v_exc) + i_syn_exc + i_syn_inh
        v_next_exc = self.v_exc + num_exc / self.exc_tau_m

        theta_decay = self.theta - self.theta / self.exc_tc_theta
        refr_mask = self.refr_exc > 0
        spike_mask = (~refr_mask) & (v_next_exc >= self.vthr)

        v_exc_next = torch.where(refr_mask | spike_mask, self.exc_vreset, v_next_exc)
        refr_exc_next = torch.where(
            refr_mask,
            self.refr_exc - 1,
            torch.where(spike_mask, torch.full_like(self.refr_exc, self.exc_refract), 0),
        )
        theta_tmp = torch.where(
            refr_mask,
            theta_decay,
            torch.where(spike_mask, theta_decay + self.exc_theta_plus, theta_decay),
        )
        theta_next = torch.clamp(theta_tmp, min=0.0, max=self.exc_theta_max)
        vthr_next = self.exc_init_vthr + theta_next

        r_exc_next = self.r_exc - self.r_exc * self.inv_td_exc + spike_mask.float() * self.exc_spike_amp

        x_exc_next = torch.where(
            spike_mask,
            torch.ones_like(self.x_exc),
            torch.clamp(self.x_exc - self.x_exc * self.inv_td_x, min=0.0),
        )
        x_exc2_next = torch.where(
            spike_mask,
            torch.ones_like(self.x_exc2),
            torch.clamp(self.x_exc2 - self.x_exc2 * self.inv_td_x2, min=0.0),
        )

        g_exc_next = self.wexc * r_exc_next

        # Inh neuron dynamics (delayed excit conductance)
        if self.delay_e2i_steps == 1:
            delay_e2i_rd = self.delay_e2i_mem[0]
        else:
            rd_idx_e2i = (self.delay_e2i_wr_idx - 1) % self.delay_e2i_steps
            delay_e2i_rd = self.delay_e2i_mem[rd_idx_e2i]

        i_syn_i = delay_e2i_rd * (self.inh_e_exc - self.v_inh)
        num_i = (self.inh_vrest - self.v_inh) + i_syn_i
        v_next_i = self.v_inh + num_i / self.inh_tau_m

        refr_i_mask = self.refr_inh > 0
        spike_i_mask = (~refr_i_mask) & (v_next_i >= self.inh_vthr)

        v_inh_next = torch.where(refr_i_mask | spike_i_mask, self.inh_vreset, v_next_i)
        refr_inh_next = torch.where(
            refr_i_mask,
            self.refr_inh - 1,
            torch.where(spike_i_mask, torch.full_like(self.refr_inh, self.inh_refract), 0),
        )

        r_inh_next = self.r_inh - self.r_inh * self.inv_td_inh + spike_i_mask.float() * self.inh_spike_amp

        sum_r_inh = r_inh_next.sum()
        g_inh_next = self.winh_div * (sum_r_inh - r_inh_next)

        # Write back states
        self.g_in_state = g_in_state_next
        self.r_exc = r_exc_next
        self.x_exc = x_exc_next
        self.x_exc2 = x_exc2_next
        self.g_inh_state = g_inh_next
        self.r_inh = r_inh_next
        self.v_exc = v_exc_next
        self.v_inh = v_inh_next
        self.theta = theta_next
        self.vthr = vthr_next
        self.refr_exc = refr_exc_next
        self.refr_inh = refr_inh_next

        # Update delay lines
        self.delay_in_mem[self.delay_in_wr_idx] = g_in_state_next
        self.delay_e2i_mem[self.delay_e2i_wr_idx] = g_exc_next
        self.delay_in_wr_idx = (self.delay_in_wr_idx + 1) % self.delay_in_steps
        self.delay_e2i_wr_idx = (self.delay_e2i_wr_idx + 1) % self.delay_e2i_steps

        # STDP updates (Brian2-style, pre/post events)
        if stdp_en:
            post_idx = torch.nonzero(spike_mask, as_tuple=False).flatten()
            if post_idx.numel() != 0:
                post_scale = (self.a_p * x_exc2_next.index_select(0, post_idx)).unsqueeze(1)
                self.W.index_add_(0, post_idx, post_scale * self.x_in.unsqueeze(0))

            if pre_idx.numel() != 0:
                self.W[:, pre_idx] = self.W[:, pre_idx] - self.a_m * x_exc_next.unsqueeze(1)

            self.W.clamp_(self.wmin, self.wmax)

        return spike_mask


def assign_labels_from_spikes(spike_counts, labels, n_labels):
    n_samples, n_neurons = spike_counts.shape
    rates = np.zeros((n_neurons, n_labels), dtype=np.float32)
    for lab in range(n_labels):
        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            continue
        rates[:, lab] = spike_counts[idx].sum(axis=0) / float(idx.size)
    assignments = np.argmax(rates, axis=1).astype(np.uint8)
    return assignments


def predict_from_spikes(spike_counts, assignments, n_labels):
    n_samples, n_neurons = spike_counts.shape
    preds = np.zeros(n_samples, dtype=np.uint8)
    for i in range(n_samples):
        best_label = 0
        best_rate = -1.0
        for lab in range(n_labels):
            idx = np.where(assignments == lab)[0]
            if idx.size == 0:
                rate = 0.0
            else:
                rate = spike_counts[i, idx].sum() / float(idx.size)
            if rate > best_rate:
                best_rate = rate
                best_label = lab
        preds[i] = best_label
    return preds


def main():
    # Parameters (match top_level.sv / import_MNIST.py)
    dt = 0.001
    t_inj = 0.350
    t_blank = 0.150
    n_time = round(t_inj / dt)
    n_blank = round(t_blank / dt)

    n_in = 784
    n_neurons = 100
    n_labels = 10
    n_total = 10000
    n_train = 9000
    n_eval = 1000
    seed = 0
    enable_torch_compile = False  # CUDAGraphの上書き問題回避。必要ならTrueに

    images, labels = load_mnist()
    images = images.astype(np.float32)
    labels = labels.astype(np.int64)
    images = images[:n_total]
    labels = labels[:n_total]

    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    data_dir = base_dir / "data"
    w_init = load_initial_weights(n_neurons, n_in, data_dir)

    cpu_device = torch.device("cpu")
    net = LIFWTA_STDP(n_in=n_in, n_neurons=n_neurons, device=cpu_device)
    net.load_weights(w_init)

    device = net.device
    images_t = torch.from_numpy(images).to(device=device, dtype=torch.float32)
    zero_input = torch.zeros(n_in, device=device)

    train_counts = torch.zeros((n_train, n_neurons), dtype=torch.int32, device=device)
    eval_counts = torch.zeros((n_eval, n_neurons), dtype=torch.int32, device=device)

    rng = np.random.default_rng(seed)

    def run_sample(spikes_t, stdp_en):
        spike_sum = torch.zeros(n_neurons, dtype=torch.int32, device=device)
        for t in range(n_time):
            s_exc = net.step(spikes_t[t], stdp_en=stdp_en)
            spike_sum += s_exc.int()
        for _ in range(n_blank):
            _ = net.step(zero_input, stdp_en=False)
        return spike_sum

    if enable_torch_compile and hasattr(torch, "compile"):
        try:
            try:
                torch._inductor.config.triton.cudagraphs = False
            except Exception:
                pass
            run_sample = torch.compile(run_sample, mode="reduce-overhead")
            print("torch.compile enabled")
        except Exception:
            print("torch.compile unavailable; running without compile")

    if tqdm is not None:
        train_iter = tqdm(range(n_train), desc="Training", leave=True)
        eval_iter = tqdm(range(n_train, n_total), desc="Inference", leave=True)
    else:
        train_iter = range(n_train)
        eval_iter = range(n_train, n_total)

    print("Start training...")
    with torch.inference_mode():
        for i in train_iter:
            if enable_torch_compile and hasattr(torch, "compiler"):
                try:
                    torch.compiler.cudagraph_mark_step_begin()
                except Exception:
                    pass
            # deterministic random stream per sample (same as import_MNIST.py)
            seed_i = int(rng.integers(0, 2**31 - 1, dtype=np.int64))
            torch.manual_seed(seed_i)
            spikes_t = online_load_and_encoding_dataset_torch(
                images_t=images_t, i=i, dt=dt, n_time=n_time
            )
            train_counts[i] = run_sample(spikes_t, stdp_en=True)

    print("Start inference...")
    with torch.inference_mode():
        for i in eval_iter:
            if enable_torch_compile and hasattr(torch, "compiler"):
                try:
                    torch.compiler.cudagraph_mark_step_begin()
                except Exception:
                    pass
            # deterministic random stream per sample (same as import_MNIST.py)
            seed_i = int(rng.integers(0, 2**31 - 1, dtype=np.int64))
            torch.manual_seed(seed_i)
            spikes_t = online_load_and_encoding_dataset_torch(
                images_t=images_t, i=i, dt=dt, n_time=n_time
            )
            eval_counts[i - n_train] = run_sample(spikes_t, stdp_en=False)

    # Move to CPU for label assignment and metrics
    train_counts_np = train_counts.cpu().numpy()
    eval_counts_np = eval_counts.cpu().numpy()
    train_labels = labels[:n_train]
    eval_labels = labels[n_train:n_train + n_eval]

    assignments = assign_labels_from_spikes(train_counts_np, train_labels, n_labels)
    preds = predict_from_spikes(eval_counts_np, assignments, n_labels)

    # Per-label precision/recall
    precision = np.zeros(n_labels, dtype=np.float32)
    recall = np.zeros(n_labels, dtype=np.float32)
    total_correct = 0
    for lab in range(n_labels):
        pred_mask = preds == lab
        true_mask = eval_labels == lab
        tp = np.sum(pred_mask & true_mask)
        pred_cnt = np.sum(pred_mask)
        true_cnt = np.sum(true_mask)
        precision[lab] = (tp / pred_cnt) if pred_cnt != 0 else 0.0
        recall[lab] = (tp / true_cnt) if true_cnt != 0 else 0.0
        total_correct += tp

    # Output
    print("Per-label precision/recall:")
    for lab in range(n_labels):
        print(f"label {lab}: precision={precision[lab]*100:.2f}%  recall={recall[lab]*100:.2f}%")
    print(f"Total correct: {int(total_correct)} / {n_eval}")


if __name__ == "__main__":
    main()
