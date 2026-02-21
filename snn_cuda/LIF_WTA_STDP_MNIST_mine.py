# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import gzip
import os 
import time
import urllib.request
try:
    from numba import njit
except Exception:
    njit = None

from Models.Neurons import ConductanceBasedLIF, DiehlAndCook2015LIF
from Models.Synapses import SingleExponentialSynapse
from Models.Connections import FullConnection, DelayConnection

np.random.seed(seed=0)

MNIST_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)


def _download_mnist_file(filename, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, filename)
    if os.path.exists(out_path):
        return out_path

    last_error = None
    for base_url in MNIST_MIRRORS:
        url = base_url + filename
        try:
            print("Downloading from {} ...".format(url))
            urllib.request.urlretrieve(url, out_path)
            return out_path
        except Exception as e:
            last_error = e

    raise RuntimeError(
        "Failed to download {} from all mirrors. Last error: {}".format(
            filename, last_error
        )
    )


def _load_mnist_images(gz_path):
    with gzip.open(gz_path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError("Invalid image file magic number: {}".format(magic))
        n_images = int.from_bytes(f.read(4), "big")
        n_rows = int.from_bytes(f.read(4), "big")
        n_cols = int.from_bytes(f.read(4), "big")
        raw = f.read()

    images = np.frombuffer(raw, dtype=np.uint8).reshape(n_images, n_rows * n_cols)
    return images.astype(np.float32) / 255.0


def _load_mnist_labels(gz_path):
    with gzip.open(gz_path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError("Invalid label file magic number: {}".format(magic))
        n_labels = int.from_bytes(f.read(4), "big")
        raw = f.read()

    labels = np.frombuffer(raw, dtype=np.uint8)
    if labels.shape[0] != n_labels:
        raise ValueError(
            "Label count mismatch. header={}, actual={}".format(
                n_labels, labels.shape[0]
            )
        )
    return labels


def load_mnist_train(cache_dir="./mnist_data"):
    images_path = _download_mnist_file("train-images-idx3-ubyte.gz", cache_dir)
    labels_path = _download_mnist_file("train-labels-idx1-ubyte.gz", cache_dir)
    images = _load_mnist_images(images_path)
    labels = _load_mnist_labels(labels_path)
    return images, labels


def validate_accumulator_equivalence():
    """A/B accumulatorがdot実装と数式同値かを小規模に検証する。"""
    rng = np.random.RandomState(0)  # グローバル乱数状態は変更しない
    update_nt, n_in, n_neurons = 10, 8, 4

    s_in_hist = (rng.rand(update_nt, n_in) < 0.25).astype(np.uint8)
    s_exc_hist = (rng.rand(update_nt, n_neurons) < 0.20).astype(np.uint8)
    x_in_hist = rng.rand(update_nt, n_in)
    x_exc_hist = rng.rand(update_nt, n_neurons)

    # 旧実装（保存+dot）
    a_old = np.dot(s_exc_hist.T, x_in_hist)
    b_old = np.dot(x_exc_hist.T, s_in_hist)

    # 新実装（イベント駆動accumulator）
    a_new = np.zeros((n_neurons, n_in))
    b_t = np.zeros((n_in, n_neurons))
    for t in range(update_nt):
        post_active = np.flatnonzero(s_exc_hist[t])
        if post_active.size > 0:
            a_new[post_active, :] += x_in_hist[t]

        pre_active = np.flatnonzero(s_in_hist[t])
        if pre_active.size > 0:
            b_t[pre_active, :] += x_exc_hist[t]

    b_new = b_t.T
    max_abs_a = float(np.max(np.abs(a_old - a_new)))
    max_abs_b = float(np.max(np.abs(b_old - b_new)))
    print(
        "Accumulator equivalence check: max|A_old-A_new|={:.3e}, max|B_old-B_new|={:.3e}".format(
            max_abs_a, max_abs_b
        )
    )


def validate_gin_state_equivalence():
    """g_in_state更新が W@c_in と数式同値かを小規模に検証する。"""
    rng = np.random.RandomState(1)  # グローバル乱数状態は変更しない
    n_in, n_neurons = 8, 4
    steps = 12
    dt = 1e-3
    td = 1e-3
    a = 1.0 - dt / td
    b = 1.0 / td

    w = rng.rand(n_neurons, n_in)
    c = np.zeros(n_in)
    g_state = np.zeros(n_neurons)
    max_abs = 0.0

    for t in range(steps):
        s = (rng.rand(n_in) < 0.25).astype(np.uint8)

        # 旧（dense）
        c = a * c + b * s
        g_dense = np.dot(w, c)

        # 新（state）
        g_state *= a
        pre_active = np.flatnonzero(s)
        if pre_active.size > 0:
            for j in pre_active:
                g_state += b * w[:, j]

        max_abs = max(max_abs, float(np.max(np.abs(g_dense - g_state))))

        # 重み更新を1回挟み、リベースの正しさも確認
        if t == 6:
            delta_w = 1e-3 * (rng.rand(n_neurons, n_in) - 0.5)
            w = np.clip(w + delta_w, 0.0, 1.0)
            g_state = np.dot(w, c)
            max_abs = max(max_abs, float(np.max(np.abs(np.dot(w, c) - g_state))))

    print("g_in state equivalence check: max|W@c - g_state|={:.3e}".format(max_abs))


if njit is not None:
    @njit(cache=True)
    def add_columns_scaled_inplace(dst, w, indices, scale):
        for i in range(dst.shape[0]):
            acc = 0.0
            for k in range(indices.shape[0]):
                acc += scale * w[i, indices[k]]
            dst[i] += acc
else:
    def add_columns_scaled_inplace(dst, w, indices, scale):
        for j in indices:
            dst += scale * w[:, j]

#################
####  Utils  ####
#################
# 画像をポアソンスパイク列に変換
def online_load_and_encoding_dataset(images, i, dt, n_time, max_fr=32,
                                     norm=140):
    fr_tmp = max_fr*norm/np.sum(images[i])
    fr = fr_tmp*np.repeat(np.expand_dims(images[i],
                                         axis=0), n_time, axis=0)
    input_spikes = np.where(np.random.rand(n_time, 784) < fr*dt, 1, 0)
    input_spikes = input_spikes.astype(np.uint8)

    return input_spikes

# ラベルの割り当て
def assign_labels(spikes, labels, n_labels, rates=None, alpha=1.0):
    """
    Assign labels to the neurons based on highest average spiking activity.
    
    Args:
        spikes (n_samples, n_neurons) : A single layer's spiking activity.
        labels (n_samples,) : Data labels corresponding to input samples.
        n_labels (int)      : The number of target labels in the data.
        rates (n_neurons, n_labels) : If passed, these represent spike rates
                                      from a previous ``assign_labels()`` call.
        alpha (float): Rate of decay of label assignments.
    return: Class assignments, per-class spike proportions, and per-class firing rates.
    """
    n_neurons = spikes.shape[1] 
    
    if rates is None:        
        rates = np.zeros((n_neurons, n_labels)).astype(np.float32)
    
    # 時間の軸でスパイク数の和を取る
    for i in range(n_labels):
        # サンプル内の同じラベルの数を求める
        n_labeled = np.sum(labels == i).astype(np.int16)
    
        if n_labeled > 0:
            # label == iのサンプルのインデックスを取得
            indices = np.where(labels == i)[0]
            
            # label == iに対する各ニューロンごとの平均発火率を計算(前回の発火率との移動平均)
            rates[:, i] = alpha*rates[:, i] + (np.sum(spikes[indices], axis=0)/n_labeled)
    
    sum_rate = np.sum(rates, axis=1)
    sum_rate[sum_rate==0] = 1
    # クラスごとの発火頻度の割合を計算する
    proportions = rates / np.expand_dims(sum_rate, 1) # (n_neurons, n_labels)
    proportions[proportions != proportions] = 0  # Set NaNs to 0
    
    # 最も発火率が高いラベルを各ニューロンに割り当てる
    assignments = np.argmax(proportions, axis=1).astype(np.uint8) # (n_neurons,)

    return assignments, proportions, rates

# assign_labelsで割り当てたラベルからサンプルのラベルの予測をする
def prediction(spikes, assignments, n_labels):
    """
    Classify data with the label with highest average spiking activity over all neurons.

    Args:
        spikes  (n_samples, n_neurons) : A layer's spiking activity.
        assignments (n_neurons,) : Neuron label assignments.
        n_labels (int): The number of target labels in the data.
    return: Predictions (n_samples,)
    """
        
    n_samples = spikes.shape[0]
    
    # 各サンプルについて各ラベルの発火率を見る
    rates = np.zeros((n_samples, n_labels)).astype(np.float32)
    
    for i in range(n_labels):
        # 各ラベルが振り分けられたニューロンの数
        n_assigns = np.sum(assignments == i).astype(np.uint8)
    
        if n_assigns > 0:
            # 各ラベルのニューロンのインデックスを取得
            indices = np.where(assignments == i)[0]
    
            # 各ラベルのニューロンのレイヤー全体における平均発火数を求める
            rates[:, i] = np.sum(spikes[:, indices], axis=1) / n_assigns
    
    # レイヤーの平均発火率が最も高いラベルを出力
    return np.argmax(rates, axis=1).astype(np.uint8) # (n_samples, )


#################
####  Model  ####
#################
class DiehlAndCook2015Network:
    def __init__(self, n_in=784, n_neurons=100, wexc=2.25, winh=0.875,
                 dt=1e-3, wmin=0.0, wmax=5e-2, lr=(1e-2, 1e-4),
                 update_nt=100, profile_every=100):
        """
        Network of Diehl and Cooks (2015) 
        https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full
        
        Args:
            n_in: Number of input neurons. Matches the 1D size of the input data.
            n_neurons: Number of excitatory, inhibitory neurons.
            wexc: Strength of synapse weights from excitatory to inhibitory layer.
            winh: Strength of synapse weights from inhibitory to excitatory layer.
            dt: Simulation time step.
            lr: Single or pair of learning rates for pre- and post-synaptic events, respectively.
            wmin: Minimum allowed weight on input to excitatory synapses.
            wmax: Maximum allowed weight on input to excitatory synapses.
            update_nt: Number of time steps of weight updates.
        """
        
        self.dt = dt
        self.lr_p, self.lr_m = lr
        self.wmax = wmax
        self.wmin = wmin

        # Neurons
        self.exc_neurons = DiehlAndCook2015LIF(n_neurons, dt=dt, tref=5e-3,
                                               tc_m=1e-1,
                                               vrest=-65, vreset=-65, 
                                               init_vthr=-52,
                                               vpeak=20, theta_plus=0.05,
                                               theta_max=35,
                                               tc_theta=1e4,
                                               e_exc=0, e_inh=-100)

        self.inh_neurons = ConductanceBasedLIF(n_neurons, dt=dt, tref=2e-3,
                                               tc_m=1e-2,
                                               vrest=-60, vreset=-45,
                                               vthr=-40, vpeak=20,
                                               e_exc=0, e_inh=-85)
        # Synapses
        self.input_synapse = SingleExponentialSynapse(n_in, dt=dt, td=1e-3)
        self.exc_synapse = SingleExponentialSynapse(n_neurons, dt=dt, td=1e-3)
        self.inh_synapse = SingleExponentialSynapse(n_neurons, dt=dt, td=2e-3)
        
        self.input_synaptictrace = SingleExponentialSynapse(n_in, dt=dt,
                                                            td=2e-2)
        self.exc_synaptictrace = SingleExponentialSynapse(n_neurons, dt=dt,
                                                          td=2e-2)
        
        # Connections
        initW = 1e-3*np.random.rand(n_neurons, n_in)
        self.input_conn = FullConnection(n_in, n_neurons,
                                         initW=initW)
        self.W_in = self.input_conn.W
        self.wexc = wexc
        self.inh_coeff = winh / (n_neurons - 1)
        self.exc2inh_W = wexc*np.eye(n_neurons)
        self.inh2exc_W = (winh/(n_neurons-1))*(np.ones((n_neurons, n_neurons)) - np.eye(n_neurons))
        
        self.delay_input = DelayConnection(N=n_neurons, delay=5e-3, dt=dt)
        self.delay_exc2inh = DelayConnection(N=n_neurons, delay=2e-3, dt=dt)
        
        self.norm = 0.1
        self.g_inh = np.zeros(n_neurons)
        self.tcount = 0
        self.update_nt = update_nt
        self.n_neurons = n_neurons
        self.n_in = n_in
        self.A = np.zeros((n_neurons, n_in))
        # Bは転置で保持: B_T[q, p] == B[p, q]
        self.B_T = np.zeros((n_in, n_neurons))
        # input_synapseと同じ差分方程式で c_in/g_in を状態化
        self.input_decay = 1.0 - self.input_synapse.dt / self.input_synapse.td
        self.input_scale = 1.0 / self.input_synapse.td
        self.c_in_state = np.zeros(n_in)
        self.g_in_state = np.zeros(n_neurons)
        self.profile_every = profile_every
        self.profile_cycle_count = 0
        self.profile_synapse = 0.0
        self.profile_conn_in_update = 0.0
        self.profile_conn_ei = 0.0
        self.profile_conn_ie = 0.0
        self.profile_neuron = 0.0
        self.profile_accumulator = 0.0
        self.profile_update = 0.0
        self.profile_pre_active_count = 0.0
        
    # スパイクトレースのリセット
    def reset_trace(self):
        self.A.fill(0.0)
        self.B_T.fill(0.0)
        self.tcount = 0
    
    # 状態の初期化
    def initialize_states(self):
        self.exc_neurons.initialize_states()
        self.inh_neurons.initialize_states()
        self.delay_input.initialize_states()
        self.delay_exc2inh.initialize_states()
        self.c_in_state.fill(0.0)
        self.g_in_state.fill(0.0)
        self.reset_trace()
        self.exc_synapse.initialize_states()
        self.inh_synapse.initialize_states()
        
    def __call__(self, s_in, stdp=True):
        if s_in.any():
            pre_active = np.nonzero(s_in)[0]
        else:
            pre_active = np.empty(0, dtype=np.int64)
        self.profile_pre_active_count += float(pre_active.size)

        # 入力層
        t0 = time.perf_counter()
        self.c_in_state = self.c_in_state * self.input_decay + self.input_scale * s_in
        x_in = self.input_synaptictrace(s_in)
        self.profile_synapse += time.perf_counter() - t0

        t0 = time.perf_counter()
        self.g_in_state *= self.input_decay
        if pre_active.size > 0:
            add_columns_scaled_inplace(self.g_in_state, self.W_in, pre_active, self.input_scale)
        delayed_g_in = self.delay_input(self.g_in_state)
        self.profile_conn_in_update += time.perf_counter() - t0

        # 興奮性ニューロン層
        t0 = time.perf_counter()
        s_exc = self.exc_neurons(delayed_g_in, self.g_inh)
        self.profile_neuron += time.perf_counter() - t0

        t0 = time.perf_counter()
        c_exc = self.exc_synapse(s_exc)
        x_exc = self.exc_synaptictrace(s_exc)
        self.profile_synapse += time.perf_counter() - t0

        t0 = time.perf_counter()
        g_exc = self.wexc * c_exc
        delayed_g_exc = self.delay_exc2inh(g_exc)
        self.profile_conn_ei += time.perf_counter() - t0

        # 抑制性ニューロン層        
        t0 = time.perf_counter()
        s_inh = self.inh_neurons(delayed_g_exc, 0)
        self.profile_neuron += time.perf_counter() - t0

        t0 = time.perf_counter()
        c_inh = self.inh_synapse(s_inh)
        self.profile_synapse += time.perf_counter() - t0

        t0 = time.perf_counter()
        sum_c_inh = np.sum(c_inh)
        self.g_inh = self.inh_coeff * (sum_c_inh - c_inh)
        self.profile_conn_ie += time.perf_counter() - t0

        if stdp:
            # dot(s_exc_, x_in_) と dot(x_exc_, s_in_) の数式同値accumulate
            t0 = time.perf_counter()
            p = int(np.argmax(s_exc))
            if s_exc[p] != 0:
                self.A[p, :] += x_in

            if pre_active.size > 0:
                np.add.at(self.B_T, pre_active, x_exc)
            self.profile_accumulator += time.perf_counter() - t0

            self.tcount += 1

            # Online STDP
            if self.tcount == self.update_nt:
                t0 = time.perf_counter()
                W = np.copy(self.input_conn.W)
                
                # postに投射される重みが均一になるようにする
                W_abs_sum = np.expand_dims(np.sum(np.abs(W), axis=1), 1)
                W_abs_sum[W_abs_sum == 0] = 1.0
                W *= self.norm / W_abs_sum
                
                # STDP則
                # B_old = dot(x_exc_, s_in_) と同値にするため B_T.T を使う
                dW = self.lr_p*(self.wmax - W)*self.A
                dW -= self.lr_m*W*self.B_T.T
                clipped_dW = np.clip(dW / self.update_nt, -1e-3, 1e-3)
                self.input_conn.W = np.clip(W + clipped_dW,
                                            self.wmin, self.wmax)
                self.W_in = self.input_conn.W
                # W更新後は g_in_state = W @ c_in_state へリベース
                self.g_in_state = np.dot(self.W_in, self.c_in_state)
                self.reset_trace() # スパイク列とスパイクトレースをリセット
                self.profile_update += time.perf_counter() - t0

        self.profile_cycle_count += 1
        if self.profile_every > 0 and self.profile_cycle_count >= self.profile_every:
            scale = 1000.0 / self.profile_cycle_count
            pre_active_mean = self.profile_pre_active_count / self.profile_cycle_count
            print(
                "[perf avg/{} cycles] synapse={:.4f}ms conn_in_update={:.4f}ms conn_ei={:.4f}ms conn_ie={:.4f}ms neuron={:.4f}ms accumulator={:.4f}ms update={:.4f}ms pre_active/timestep={:.4f}".format(
                    self.profile_cycle_count,
                    self.profile_synapse * scale,
                    self.profile_conn_in_update * scale,
                    self.profile_conn_ei * scale,
                    self.profile_conn_ie * scale,
                    self.profile_neuron * scale,
                    self.profile_accumulator * scale,
                    self.profile_update * scale,
                    pre_active_mean,
                )
            )
            self.profile_cycle_count = 0
            self.profile_synapse = 0.0
            self.profile_conn_in_update = 0.0
            self.profile_conn_ei = 0.0
            self.profile_conn_ie = 0.0
            self.profile_neuron = 0.0
            self.profile_accumulator = 0.0
            self.profile_update = 0.0
            self.profile_pre_active_count = 0.0
        
        return s_exc
        
if __name__ == '__main__':
    validate_accumulator_equivalence()
    validate_gin_state_equivalence()
    # 350ms画像入力、150ms入力なしでリセットさせる(膜電位の閾値以外)
    dt = 1e-3 # タイムステップ(sec)
    t_inj = 0.350 # 刺激入力時間(sec)
    t_blank = 0.150 # ブランク時間(sec)
    nt_inj = round(t_inj/dt)
    nt_blank = round(t_blank/dt)
    
    n_neurons = 100 #興奮性/抑制性ニューロンの数
    n_labels = 10 #ラベル数
    n_epoch = 30 #エポック数
    
    n_train = 10000 # 訓練データの数
    update_nt = nt_inj # STDP則による重みの更新間隔
    
    train_images, train_labels = load_mnist_train() # MNISTデータの読み込み
    labels = train_labels[:n_train] # ラベルの配列
    
    # ネットワークの定義
    network = DiehlAndCook2015Network(n_in=784, n_neurons=n_neurons,
                                      wexc=2.25, winh=0.85,
                                      dt=dt, wmin=0.0, wmax=5e-2,
                                      lr=(1e-2, 1e-4),
                                      update_nt=update_nt)
    
    network.initialize_states() # ネットワークの初期化
    spikes = np.zeros((n_train, n_neurons)).astype(np.uint8) #スパイクを記録する変数
    accuracy_all = np.zeros(n_epoch) # 訓練精度を記録する変数
    blank_input = np.zeros(784) # ブランク入力
    init_max_fr = 32 # 初期のポアソンスパイクの最大発火率
    
    results_save_dir = "./LIF_WTA_STDP_MNIST_results/" # 結果を保存するディレクトリ
    os.makedirs(results_save_dir, exist_ok=True) # ディレクトリが無ければ作成
    
    #################
    ##　Simulation  ##
    #################
    for epoch in range(n_epoch):
        for i in tqdm(range(n_train)):
            max_fr = init_max_fr
            while(True):
                # 入力スパイクをオンラインで生成
                input_spikes = online_load_and_encoding_dataset(train_images, i, dt,
                                                                nt_inj, max_fr)
                spike_list = [] # サンプルごとにスパイクを記録するリスト
                # 画像刺激の入力
                for t in range(nt_inj):
                    s_exc = network(input_spikes[t], stdp=True)
                    spike_list.append(s_exc)
                
                spikes[i] = np.sum(np.array(spike_list), axis=0) # スパイク数を記録
                
                # ブランク刺激の入力
                for _ in range(nt_blank):
                    _ = network(blank_input, stdp=False)
    
                num_spikes_exc = np.sum(np.array(spike_list)) # スパイク数を計算
                if num_spikes_exc >= 5: # スパイク数が5より大きければ次のサンプルへ
                    break
                else: # スパイク数が5より小さければ入力発火率を上げて再度刺激
                    max_fr += 16
        
        # ニューロンを各ラベルに割り当てる
        if epoch == 0:
            assignments, proportions, rates = assign_labels(spikes, labels,
                                                            n_labels)
        else:
            assignments, proportions, rates = assign_labels(spikes, labels,
                                                            n_labels, rates)
        print("Assignments:\n", assignments)
        
        # スパイク数の確認(正常に発火しているか確認)
        sum_nspikes = np.sum(spikes, axis=1)
        mean_nspikes = np.mean(sum_nspikes).astype(np.float16)
        print("Ave. spikes:", mean_nspikes)
        print("Min. spikes:", sum_nspikes.min())
        print("Max. spikes:", sum_nspikes.max())
    
        # 入力サンプルのラベルを予測する
        predicted_labels = prediction(spikes, assignments, n_labels)
        
        # 訓練精度を計算
        accuracy = np.mean(np.where(labels==predicted_labels, 1, 0)).astype(np.float16)
        print("epoch :", epoch, " accuracy :", accuracy)
        accuracy_all[epoch] = accuracy
        
        # 学習率の減衰
        network.lr_p *= 0.75
        network.lr_m *= 0.75
        
        # 重みの保存(エポック毎)
        np.save(results_save_dir+"weight_epoch"+str(epoch)+".npy",
                network.input_conn.W)
        
    #################
    ###　 Results  ###
    #################
    plt.figure(figsize=(5,4))
    plt.plot(np.arange(1, n_epoch+1), accuracy_all*100,
             color="k")
    plt.xlabel("Epoch")
    plt.ylabel("Train accuracy (%)")
    plt.savefig(results_save_dir+"accuracy.svg")
    #plt.show()
    
    # パラメータの保存
    np.save(results_save_dir+"assignments.npy", assignments)
    np.save(results_save_dir+"weight.npy", network.input_conn.W)
    np.save(results_save_dir+"exc_neurons_theta.npy",
            network.exc_neurons.theta)
