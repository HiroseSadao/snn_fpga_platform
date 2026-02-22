from __future__ import annotations

import argparse
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
try:
    import serial
except Exception:
    serial = None

# Communication parameters
SERIAL_PORTNAME = "COM7"
BAUD = 115200
TIMEOUT_SEC = 2.0
WRITE_TIMEOUT_SEC = 2.0
TRANSIENT_RETRY_MAX = 5
TRANSIENT_RETRY_SLEEP_SEC = 0.003

# Protocol constants
REQ_SYNC = 0xA5
RESP_SYNC = 0x5A
PROTO_VER = 0x01

OP_ADD_I32 = 0x01
OP_SD_TO_DDR_COPY = 0x11
OP_RUN_SAMPLE_INFER = 0x20
OP_READ_SPIKE_COUNT = 0x21
OP_READ_RAW_U8 = 0x22
OP_READ_POISSON_THRESH = 0x23
OP_READ_INFER_DEBUG = 0x24
OP_WRITE_INFER_WEIGHT = 0x25
OP_TRAIN_QUERY_CAPS = 0x30
OP_TRACE_UPDATE = 0x31
OP_STDP_UPDATE_TILE = 0x32

STATUS_OK = 0x00
STATUS_BAD_PACKET = 0xE1
STATUS_UNSUPPORTED_OP = 0xE2

# Fixed-point S16.16 model constants (must match top_level.sv)
FXP_SHIFT = 16
FXP_ALPHA = 62259
FXP_ALPHA_INH = 58982
FXP_INPUT_W = 8192
FXP_THRESH = 65536
FXP_BIAS_LSB = 512
FXP_ONE = 65536
FXP_HALF = 32768
FXP_WEXC = 147456
FXP_INH_COEFF = 563
FXP_INH_THRESH = -2621440
N_IN = 784
N_NEURONS = 100
N_WEIGHTS = N_NEURONS * N_IN

# Fixed-point training constants (mine.py defaults)
TRAIN_WMAX_Q16 = int(round(0.05 * (1 << FXP_SHIFT)))
TRAIN_WMIN_Q16 = 0
TRAIN_NORM_Q16 = int(round(0.1 * (1 << FXP_SHIFT)))
TRAIN_LR_P_Q16 = int(round(1e-2 * (1 << FXP_SHIFT)))
TRAIN_LR_M_Q16 = int(round(1e-4 * (1 << FXP_SHIFT)))
TRAIN_CLIP_DW_Q16 = int(round(1e-3 * (1 << FXP_SHIFT)))

# Inference-only reference parameters from LIF_WTA_STDP_MNIST_mine.py
MINE_DT = 1e-3
MINE_WEXC = 2.25
MINE_WINH = 0.85

# Poisson/RNG constants (must match top_level.sv)
POISSON_NUM_CONST = 9175  # floor(32*140*2048*1e-3)
RNG_MAX = 2047
LCG_A = 1664525
LCG_C = 1013904223

LAST_IO: dict[str, object] = {}


@dataclass(frozen=True)
class TrainDDRLayout:
    # Word-addressed logical map (4-byte word units).
    base_w_q16: int
    base_a_q16: int
    base_bt_q16: int
    base_exc_theta: int
    base_v_state: int
    base_delay_lines: int
    base_g_in_state: int
    total_words: int


def build_train_ddr_layout() -> TrainDDRLayout:
    """Step1: 学習用メモリマップ（DDR前提）の論理配置を固定する。

    ここでは word address (32-bit word) 単位で定義する。
    実FPGA側がまだDDR未接続でも、host/HDL間の契約として先に固定しておく。
    """
    word = 0
    base_w_q16 = word                    # 1 weight / word (lower 16b used)
    word += N_WEIGHTS
    base_a_q16 = word                    # 1 trace / word (Q16.16 or unsigned Q16 host-side)
    word += N_WEIGHTS
    base_bt_q16 = word                   # transposed [N_IN, N_NEURONS]
    word += N_WEIGHTS
    base_exc_theta = word                # [N_NEURONS] s32
    word += N_NEURONS
    base_v_state = word                  # [N_NEURONS] s32
    word += N_NEURONS
    base_delay_lines = word              # placeholder aggregate region
    word += (N_NEURONS * 8)
    base_g_in_state = word               # [N_NEURONS] s32
    word += N_NEURONS
    return TrainDDRLayout(
        base_w_q16=base_w_q16,
        base_a_q16=base_a_q16,
        base_bt_q16=base_bt_q16,
        base_exc_theta=base_exc_theta,
        base_v_state=base_v_state,
        base_delay_lines=base_delay_lines,
        base_g_in_state=base_g_in_state,
        total_words=word,
    )


def calc_checksum(payload: bytes) -> int:
    checksum = 0
    for b in payload:
        checksum ^= b
    return checksum & 0xFF


def pack_i32_twos_complement(value: int) -> bytes:
    v = int(value) & 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    return struct.pack("<i", v)


def build_request(opcode: int, args: list[int]) -> bytes:
    payload = bytearray()
    payload.append(PROTO_VER)
    payload.append(opcode & 0xFF)
    payload.append(len(args) & 0xFF)
    for value in args:
        payload.extend(pack_i32_twos_complement(value))

    packet = bytearray()
    packet.append(REQ_SYNC)
    packet.extend(payload)
    packet.append(calc_checksum(payload))
    return bytes(packet)


def read_exact(ser: serial.Serial, size: int) -> bytes:
    data = ser.read(size)
    if len(data) != size:
        raise TimeoutError(f"Timeout while reading {size} bytes (got {len(data)})")
    return data


def parse_response(raw: bytes) -> tuple[int, int]:
    if len(raw) != 7:
        raise ValueError(f"Invalid response length: {len(raw)}")
    if raw[0] != RESP_SYNC:
        raise ValueError(f"Invalid response sync byte: 0x{raw[0]:02X}")

    status = raw[1]
    result = struct.unpack("<i", raw[2:6])[0]
    checksum = raw[6]
    payload = raw[1:6]
    expected_checksum = calc_checksum(payload)
    if checksum != expected_checksum:
        raise ValueError(
            f"Checksum mismatch: got 0x{checksum:02X}, expected 0x{expected_checksum:02X}"
        )
    return status, result


def hex_bytes(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


def decode_bad_packet_result(result: int) -> str:
    u = result & 0xFFFFFFFF
    reason = (u >> 24) & 0xFF
    opcode = (u >> 16) & 0xFF
    arg0_lo16 = u & 0xFFFF
    if reason == 0x11:
        return (
            f"reason=READ_SPIKE_ARG_RANGE(0x11), "
            f"opcode=0x{opcode:02X}, arg0_lo16=0x{arg0_lo16:04X}({arg0_lo16})"
        )
    if reason == 0x12:
        return (
            f"reason=READ_RAW_U8_ARG_OR_NOT_READY(0x12), "
            f"opcode=0x{opcode:02X}, arg0_lo16=0x{arg0_lo16:04X}({arg0_lo16})"
        )
    if reason == 0x13:
        return (
            f"reason=READ_POISSON_THRESH_ARG(0x13), "
            f"opcode=0x{opcode:02X}, arg0_lo16=0x{arg0_lo16:04X}({arg0_lo16})"
        )
    if reason == 0x20:
        return (
            f"reason=SD_REQ_ARG(0x20), "
            f"opcode=0x{opcode:02X}, arg0_lo16=0x{arg0_lo16:04X}({arg0_lo16})"
        )
    if reason == 0x21:
        sd_cd_n = (u >> 15) & 0x1
        sd_status = u & 0x1F
        return (
            f"reason=SD_CD_N(0x21), opcode=0x{opcode:02X}, "
            f"SD_CD_N={sd_cd_n}, sd_status=0x{sd_status:02X}"
        )
    if reason == 0x22:
        return (
            f"reason=SD_WAIT_TIMEOUT(0x22), "
            f"opcode=0x{opcode:02X}, wait_lo16=0x{arg0_lo16:04X}({arg0_lo16})"
        )
    if reason == 0x23:
        return (
            f"reason=SD_BAD_HEADER(0x23), "
            f"opcode=0x{opcode:02X}, file_bytes_seen_lo16=0x{arg0_lo16:04X}({arg0_lo16})"
        )
    if reason == 0x24:
        return (
            f"reason=SD_SECTOR_LIMIT_END(0x24), "
            f"opcode=0x{opcode:02X}, sectors_left_lo16=0x{arg0_lo16:04X}({arg0_lo16})"
        )
    if reason == 0x14:
        return (
            f"reason=READ_INFER_DEBUG_ARG(0x14), "
            f"opcode=0x{opcode:02X}, arg0_lo16=0x{arg0_lo16:04X}({arg0_lo16})"
        )
    if reason == 0x15:
        return (
            f"reason=WRITE_INFER_WEIGHT_ARG(0x15), "
            f"opcode=0x{opcode:02X}, arg0_lo16=0x{arg0_lo16:04X}({arg0_lo16})"
        )
    if u == 0:
        return "no debug payload (result=0)"
    return f"reason=0x{reason:02X}, opcode=0x{opcode:02X}, arg0_lo16=0x{arg0_lo16:04X}"


def send_request(
    ser: serial.Serial,
    opcode: int,
    args: list[int],
    response_timeout: float = TIMEOUT_SEC,
    transient_retry_max: int = TRANSIENT_RETRY_MAX,
) -> tuple[int, int]:
    req = build_request(opcode, args)
    old_timeout = ser.timeout
    ser.timeout = response_timeout
    try:
        for attempt in range(transient_retry_max + 1):
            LAST_IO["opcode"] = opcode
            LAST_IO["args"] = list(args)
            LAST_IO["req"] = req
            LAST_IO["resp_raw"] = None
            LAST_IO["status"] = None
            LAST_IO["result"] = None
            LAST_IO["retry_attempt"] = attempt

            ser.reset_input_buffer()
            ser.write(req)
            ser.flush()
            try:
                resp_raw = read_exact(ser, 7)
            except TimeoutError:
                LAST_IO["resp_raw"] = None
                if attempt < transient_retry_max:
                    # FPGA can drop a back-to-back request while response_ready/sd_copy_active
                    # is still deasserting; retry after a short gap.
                    time.sleep(TRANSIENT_RETRY_SLEEP_SEC)
                    continue
                raise
            LAST_IO["resp_raw"] = resp_raw
            status, result = parse_response(resp_raw)
            LAST_IO["status"] = status
            LAST_IO["result"] = result

            transient_bad_packet = (status == STATUS_BAD_PACKET) and ((result & 0xFFFFFFFF) == 0)
            if transient_bad_packet and attempt < transient_retry_max:
                time.sleep(TRANSIENT_RETRY_SLEEP_SEC)
                continue
            return status, result
    finally:
        ser.timeout = old_timeout


def require_ok(status: int, context: str) -> None:
    if status == STATUS_OK:
        return
    req_hex = hex_bytes(LAST_IO.get("req", b"") if isinstance(LAST_IO.get("req"), bytes) else b"")
    resp_raw = LAST_IO.get("resp_raw")
    resp_hex = hex_bytes(resp_raw) if isinstance(resp_raw, bytes) else "<none>"
    opcode = LAST_IO.get("opcode")
    args = LAST_IO.get("args")
    result = int(LAST_IO.get("result") or 0)
    retry_attempt = int(LAST_IO.get("retry_attempt") or 0)
    debug_msg = (
        f"{context}: opcode=0x{int(opcode) & 0xFF:02X} args={args} "
        f"req=[{req_hex}] resp=[{resp_hex}] result=0x{result & 0xFFFFFFFF:08X} "
        f"retry_attempt={retry_attempt}"
    ) if opcode is not None else (
        f"{context}: req=[{req_hex}] resp=[{resp_hex}] result=0x{result & 0xFFFFFFFF:08X} "
        f"retry_attempt={retry_attempt}"
    )
    if status == STATUS_BAD_PACKET:
        raise RuntimeError(
            f"{context}: FPGA rejected packet (BAD_PACKET). "
            f"{debug_msg}. decoded={decode_bad_packet_result(result)}"
        )
    if status == STATUS_UNSUPPORTED_OP:
        raise RuntimeError(f"{context}: FPGA rejected packet (UNSUPPORTED_OP). {debug_msg}")
    raise RuntimeError(f"{context}: FPGA returned unknown status 0x{status:02X}. {debug_msg}")


def fpga_sd_to_ddr_copy(
    ser: serial.Serial,
    start_lba: int,
    num_sectors: int,
    timeout_sec: float = 120.0
) -> int:
    sectors_msg = "auto(RAW1 header)" if num_sectors == 0 else str(num_sectors)
    print(f"Requesting SD->DDR copy: start_lba={start_lba}, sectors={sectors_msg}")
    t0 = time.time()
    status, result = send_request(
        ser=ser,
        opcode=OP_SD_TO_DDR_COPY,
        args=[start_lba, num_sectors],
        response_timeout=timeout_sec,
    )
    require_ok(status, "SD->DDR copy")
    elapsed = time.time() - t0
    print(f"SD->DDR copy completed in {elapsed:.2f}s, words_written={result}")
    return result


def fpga_run_sample_infer(
    ser: serial.Serial,
    seed: int,
    n_steps: int,
    timeout_sec: float = 120.0
) -> int:
    seed_i32 = int(seed) & 0xFFFFFFFF
    if seed_i32 >= 0x80000000:
        seed_i32 -= 0x100000000
    print(f"Requesting FPGA inference: seed=0x{seed & 0xFFFFFFFF:08X}, steps={n_steps}")
    status, total_spikes = send_request(
        ser=ser,
        opcode=OP_RUN_SAMPLE_INFER,
        args=[seed_i32, n_steps],
        response_timeout=timeout_sec,
    )
    require_ok(status, "RUN_SAMPLE_INFER")
    print(f"FPGA inference finished, total_spikes={total_spikes}")
    return total_spikes


def build_fixed_weight_matrix_q16() -> np.ndarray:
    # Must match the fallback fixed connectivity used by the Python mine-style reference.
    w_q16 = np.zeros((N_NEURONS, N_IN), dtype=np.uint16)
    for n in range(N_NEURONS):
        for i in range(N_IN):
            if ((i + n) & 0x3) == 0:
                w_q16[n, i] = np.uint16(FXP_INPUT_W & 0xFFFF)
    return w_q16


def load_weight_matrix_q16_from_file(path: str) -> np.ndarray:
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if "w_in" in arr:
            data = arr["w_in"]
        else:
            first_key = next(iter(arr.files), None)
            if first_key is None:
                raise ValueError(f"No arrays found in npz: {path}")
            data = arr[first_key]
    else:
        data = arr

    data_np = np.asarray(data)
    if data_np.shape != (N_NEURONS, N_IN):
        raise ValueError(
            f"weights shape mismatch: got {data_np.shape}, expected {(N_NEURONS, N_IN)}"
        )

    if np.issubdtype(data_np.dtype, np.integer):
        q16 = np.asarray(data_np, dtype=np.int64)
        if np.any((q16 < 0) | (q16 > 0xFFFF)):
            raise ValueError("integer weight file must contain Q0.16 values in [0, 65535]")
        return q16.astype(np.uint16)

    data_f = np.asarray(data_np, dtype=np.float64)
    q16 = np.clip(np.rint(data_f * (1 << FXP_SHIFT)), 0, 0xFFFF).astype(np.uint16)
    return q16


def fpga_write_infer_weights(ser: serial.Serial, w_q16: np.ndarray) -> None:
    if w_q16.shape != (N_NEURONS, N_IN):
        raise ValueError(f"w_q16 shape mismatch: got {w_q16.shape}, expected {(N_NEURONS, N_IN)}")
    flat = np.asarray(w_q16, dtype=np.uint16).reshape(-1)
    nnz = int(np.count_nonzero(flat))
    print(
        "Uploading infer weights via UART: "
        f"count={flat.size}, nonzero={nnz}, q16_sum={int(np.sum(flat, dtype=np.uint64))}"
    )
    t0 = time.time()
    for idx, val in enumerate(flat):
        status, result = send_request(ser, OP_WRITE_INFER_WEIGHT, [idx, int(val)])
        require_ok(status, f"WRITE_INFER_WEIGHT[{idx}]")
        if (idx & 0x0FFF) == 0x0FFF or idx == (flat.size - 1):
            elapsed = time.time() - t0
            print(f"  uploaded {idx + 1}/{flat.size} ({elapsed:.1f}s)")
    print(f"Weight upload completed in {time.time() - t0:.2f}s")


def fpga_read_spike_counts(ser: serial.Serial) -> list[int]:
    counts = []
    for neuron_idx in range(N_NEURONS):
        status, value = send_request(ser, OP_READ_SPIKE_COUNT, [neuron_idx, 0])
        require_ok(status, f"READ_SPIKE_COUNT[{neuron_idx}]")
        counts.append(int(value) & 0xFFFF)
    return counts


def fpga_read_raw_image_u8(ser: serial.Serial) -> list[int]:
    pixels = []
    for idx in range(N_IN):
        status, value = send_request(ser, OP_READ_RAW_U8, [idx, 0])
        require_ok(status, f"READ_RAW_U8[{idx}]")
        pixels.append(int(value) & 0xFF)
    return pixels


def fpga_read_poisson_thresh(ser: serial.Serial) -> list[int]:
    vals = []
    for idx in range(N_IN):
        status, value = send_request(ser, OP_READ_POISSON_THRESH, [idx, 0])
        require_ok(status, f"READ_POISSON_THRESH[{idx}]")
        vals.append(int(value) & 0x7FF)
    return vals


def fpga_read_infer_debug(ser: serial.Serial) -> dict[str, int]:
    names = {
        0: "total_input_spikes_generated",
        1: "total_syn_hits_applied",
        2: "last_step_input_spikes",
        3: "first_step_input_spikes",
        4: "first_step_hits_n0",
        5: "first_step_hits_n3",
        6: "first_step_hits_n7",
        7: "infer_total_spikes",
        8: "infer_steps_target",
        9: "infer_step_idx",
        10: "infer_state",
        11: "raw_image0_sum_u8",
    }
    out: dict[str, int] = {}
    for idx, name in names.items():
        status, value = send_request(ser, OP_READ_INFER_DEBUG, [idx, 0])
        require_ok(status, f"READ_INFER_DEBUG[{idx}]")
        out[name] = int(value) & 0xFFFFFFFF
    return out


def fpga_add(ser: serial.Serial, a: int, b: int) -> int:
    print(f"Sending ADD request: {a} + {b}")
    status, result = send_request(ser, OP_ADD_I32, [a, b], response_timeout=TIMEOUT_SEC)
    require_ok(status, "ADD request")
    return result


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
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


def read_mnist_image_u8(sample_idx: int) -> tuple[list[int], int]:
    images, labels = load_mnist()
    if sample_idx < 0 or sample_idx >= len(images):
        raise ValueError(f"sample_idx out of range: {sample_idx} (max={len(images)-1})")

    img = np.array(images[sample_idx], dtype=np.float32)
    if img.max() <= 1.0:
        img = img * 255.0
    img_u8 = np.clip(np.rint(img), 0, 255).astype(np.uint8).reshape(N_IN)
    return img_u8.tolist(), int(labels[sample_idx])


def resolve_raw_bin_path(raw_bin_arg: str | None) -> Path:
    candidates: list[Path] = []
    if raw_bin_arg:
        p = Path(raw_bin_arg)
        candidates.extend(
            [
                p,
                Path.cwd() / p,
                Path(__file__).resolve().parent / p,
                Path(__file__).resolve().parent.parent / p,
            ]
        )
    else:
        candidates.extend(
            [
                Path.cwd() / "raw_samples_u8.bin",
                Path.cwd() / "raw_samples.bin",
                Path(__file__).resolve().parent.parent / "raw_samples_u8.bin",
                Path(__file__).resolve().parent.parent / "raw_samples.bin",
            ]
        )

    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()

    searched = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        "RAW1 file not found. Pass the exact file written by import_MNIST_raw.py "
        f"with --raw-bin. searched=[{searched}]"
    )


def read_raw1_first_image_u8(raw_bin_path: str) -> tuple[list[int], int]:
    p = Path(raw_bin_path)
    with p.open("rb") as f:
        header = f.read(20)
        if len(header) != 20:
            raise ValueError("RAW1 header is too short")
        magic, version, num_images, n_features, bytes_per_image = struct.unpack("<4sIIII", header)
        if magic != b"RAW1":
            raise ValueError("Invalid RAW1 magic")
        if version != 1:
            raise ValueError(f"Unsupported RAW1 version: {version}")
        if n_features != N_IN:
            raise ValueError(f"Unexpected n_features: {n_features}")
        if bytes_per_image != N_IN:
            raise ValueError(
                f"RAW1 bytes_per_image must be 784 for FPGA Poisson mode, got {bytes_per_image}"
            )
        if num_images <= 0:
            raise ValueError("RAW1 has no images")

        labels = f.read(num_images)
        if len(labels) != num_images:
            raise ValueError("RAW1 labels are truncated")

        first = f.read(bytes_per_image)
        if len(first) != bytes_per_image:
            raise ValueError("RAW1 first image is truncated")
        label0 = labels[0]
        return list(first), int(label0)


def neuron_bias(neuron_idx: int) -> int:
    return ((neuron_idx & 0x7) + 1) * FXP_BIAS_LSB


def to_s32(v: int) -> int:
    v &= 0xFFFFFFFF
    if v & 0x80000000:
        v -= 0x100000000
    return v


def q16_to_float(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64) / float(1 << FXP_SHIFT)


def float_to_q16_clip(arr: np.ndarray, lo: int = 0, hi: int = 0xFFFF) -> np.ndarray:
    q = np.rint(np.asarray(arr, dtype=np.float64) * float(1 << FXP_SHIFT))
    return np.clip(q, lo, hi).astype(np.int64)


def kernel_trace_update_python(
    A: np.ndarray,
    B_T: np.ndarray,
    x_in: np.ndarray,
    x_exc: np.ndarray,
    winner_idx: int | None,
    pre_active: np.ndarray,
) -> None:
    if winner_idx is not None and winner_idx >= 0:
        A[winner_idx, :] += x_in
    if pre_active.size > 0:
        np.add.at(B_T, pre_active.astype(np.int64), x_exc)


def kernel_stdp_update_tile_python(
    W: np.ndarray,
    A: np.ndarray,
    B_T: np.ndarray,
    row0: int,
    nrows: int,
    *,
    lr_p: float = 1e-2,
    lr_m: float = 1e-4,
    wmin: float = 0.0,
    wmax: float = 5e-2,
    norm: float = 0.1,
    update_nt: int = 100,
    clip_abs: float = 1e-3,
) -> None:
    row1 = min(row0 + nrows, W.shape[0])
    if row1 <= row0:
        return
    w_tile = np.array(W[row0:row1, :], copy=True)
    w_abs_sum = np.sum(np.abs(w_tile), axis=1, keepdims=True)
    w_abs_sum[w_abs_sum == 0.0] = 1.0
    w_tile *= norm / w_abs_sum
    dW = lr_p * (wmax - w_tile) * A[row0:row1, :]
    dW -= lr_m * w_tile * B_T[:, row0:row1].T
    dW = np.clip(dW / float(update_nt), -clip_abs, clip_abs)
    W[row0:row1, :] = np.clip(w_tile + dW, wmin, wmax)


def selfcheck_training_kernels(seed: int = 0) -> None:
    rng = np.random.RandomState(seed)
    w = 1e-3 * rng.rand(N_NEURONS, N_IN)
    A_ref = np.zeros((N_NEURONS, N_IN), dtype=np.float64)
    B_T_ref = np.zeros((N_IN, N_NEURONS), dtype=np.float64)
    A_k = np.zeros_like(A_ref)
    B_T_k = np.zeros_like(B_T_ref)

    update_nt = 23
    for _ in range(update_nt):
        s_in = (rng.rand(N_IN) < 0.04).astype(np.uint8)
        s_exc = (rng.rand(N_NEURONS) < 0.02).astype(np.uint8)
        x_in = rng.rand(N_IN)
        x_exc = rng.rand(N_NEURONS)
        pre_active = np.flatnonzero(s_in)
        p = int(np.argmax(s_exc))
        winner = p if s_exc[p] else -1

        if winner >= 0:
            A_ref[winner, :] += x_in
        if pre_active.size > 0:
            np.add.at(B_T_ref, pre_active, x_exc)
        kernel_trace_update_python(A_k, B_T_k, x_in, x_exc, winner, pre_active)

    print(
        "train kernel trace selfcheck: "
        f"max|A_ref-A_k|={float(np.max(np.abs(A_ref-A_k))):.3e}, "
        f"max|B_T_ref-B_T_k|={float(np.max(np.abs(B_T_ref-B_T_k))):.3e}"
    )

    w_ref = np.array(w, copy=True)
    w_k = np.array(w, copy=True)
    # reference (full matrix)
    w_abs_sum = np.sum(np.abs(w_ref), axis=1, keepdims=True)
    w_abs_sum[w_abs_sum == 0.0] = 1.0
    w_ref *= 0.1 / w_abs_sum
    dW = 1e-2 * (0.05 - w_ref) * A_ref
    dW -= 1e-4 * w_ref * B_T_ref.T
    w_ref = np.clip(w_ref + np.clip(dW / update_nt, -1e-3, 1e-3), 0.0, 0.05)

    # kernelized by tiles
    tile_rows = 13
    for row0 in range(0, N_NEURONS, tile_rows):
        kernel_stdp_update_tile_python(
            w_k, A_k, B_T_k, row0, tile_rows, update_nt=update_nt
        )

    print(
        "train kernel STDP selfcheck: "
        f"max|W_ref-W_k|={float(np.max(np.abs(w_ref-w_k))):.3e}"
    )


def fpga_train_query_caps(ser: serial.Serial) -> int:
    status, result = send_request(ser, OP_TRAIN_QUERY_CAPS, [0, 0])
    require_ok(status, "TRAIN_QUERY_CAPS")
    return int(result) & 0xFFFFFFFF


def fpga_trace_update_kernel(
    ser: serial.Serial,
    *,
    winner_idx: int,
    pre_count: int,
) -> int:
    """Kernel trigger only. Data搬入(x_in/x_exc/prelist)は別opcode実装前提。"""
    status, result = send_request(ser, OP_TRACE_UPDATE, [winner_idx, pre_count])
    require_ok(status, "TRACE_UPDATE")
    return int(result)


def fpga_stdp_update_tile(
    ser: serial.Serial,
    *,
    row0: int,
    nrows: int,
) -> int:
    status, result = send_request(ser, OP_STDP_UPDATE_TILE, [row0, nrows])
    require_ok(status, "STDP_UPDATE_TILE")
    return int(result)


def lcg_next_u32(state: int) -> int:
    return (state * LCG_A + LCG_C) & 0xFFFFFFFF


def build_poisson_thresholds_u11(image_u8: list[int]) -> list[int]:
    sum_u8 = int(sum(image_u8))
    if sum_u8 <= 0:
        return [0] * N_IN
    out = []
    for px in image_u8:
        q = (POISSON_NUM_CONST * int(px)) // sum_u8
        if q > RNG_MAX:
            q = RNG_MAX
        out.append(int(q))
    return out


def run_fixed_point_python_poisson(image_u8: list[int], n_steps: int, seed: int) -> list[int]:
    thresholds = build_poisson_thresholds_u11(image_u8)
    return run_fixed_point_python_poisson_with_thresholds(thresholds, n_steps, seed)


def run_fixed_point_python_poisson_with_thresholds(thresholds: list[int], n_steps: int, seed: int) -> list[int]:
    v = [0] * N_NEURONS
    v_inh = [0] * N_NEURONS
    c_inh = [0] * N_NEURONS
    g_inh = [0] * N_NEURONS
    g_exc_delay0 = [0] * N_NEURONS
    g_exc_delay1 = [0] * N_NEURONS
    spike_count = [0] * N_NEURONS
    rng_state = seed & 0xFFFFFFFF

    for _ in range(n_steps):
        s_in = [0] * N_IN
        for i in range(N_IN):
            rng_state = lcg_next_u32(rng_state)
            rand11 = (rng_state >> 21) & 0x7FF
            s_in[i] = 1 if rand11 < thresholds[i] else 0

        s_exc = [0] * N_NEURONS

        for n in range(N_NEURONS):
            accum = 0
            for i in range(N_IN):
                if s_in[i] and (((i + n) & 0x3) == 0):
                    accum = to_s32(accum + FXP_INPUT_W)
            v_n_next = to_s32(((to_s32(v[n]) * FXP_ALPHA) >> FXP_SHIFT) + accum - g_inh[n])
            if v_n_next >= FXP_THRESH:
                v[n] = to_s32(v_n_next - FXP_THRESH)
                spike_count[n] += 1
                s_exc[n] = 1
            else:
                v[n] = v_n_next

        sum_c_inh = 0
        for n in range(N_NEURONS):
            g_exc_new = FXP_WEXC if s_exc[n] else 0
            delayed_g_exc = g_exc_delay1[n]
            g_exc_delay1[n] = g_exc_delay0[n]
            g_exc_delay0[n] = g_exc_new

            v_inh_next = to_s32(((to_s32(v_inh[n]) * FXP_ALPHA_INH) >> FXP_SHIFT) + delayed_g_exc)
            s_inh = 1 if v_inh_next >= FXP_INH_THRESH else 0
            if s_inh:
                v_inh[n] = to_s32(v_inh_next - FXP_INH_THRESH)
            else:
                v_inh[n] = v_inh_next

            c_inh_next = to_s32(c_inh[n] >> 1)
            if s_inh:
                c_inh_next = to_s32(c_inh_next + FXP_HALF)
            c_inh[n] = c_inh_next
            sum_c_inh = to_s32(sum_c_inh + c_inh_next)

        for n in range(N_NEURONS):
            diff = sum_c_inh - c_inh[n]
            if diff < 0:
                diff = 0
            g_inh[n] = to_s32((diff * FXP_INH_COEFF) >> FXP_SHIFT)
    return spike_count


def _build_fixed_w_in_for_mine_like() -> np.ndarray:
    # Reuse the same source used for FPGA UART initialization to avoid drift.
    return build_fixed_weight_matrix_q16().astype(np.float64) / float(1 << FXP_SHIFT)


def _single_exp_step(r: np.ndarray, spike: np.ndarray, dt: float, td: float) -> np.ndarray:
    r = r * (1.0 - dt / td) + spike / td
    return r


def _delay_step(buf: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # buf shape: (N, nt_delay), nt_delay >= 1
    out = buf[:, -1].copy()
    if buf.shape[1] > 1:
        buf[:, 1:] = buf[:, :-1]
    buf[:, 0] = x
    return out, buf


def _conductance_lif_step(
    v: np.ndarray,
    tlast: np.ndarray,
    tcount: int,
    g_exc: np.ndarray,
    g_inh: np.ndarray,
    *,
    dt: float,
    tref: float,
    tc_m: float,
    vrest: float,
    vreset: float,
    vthr: np.ndarray,
    vpeak: float,
    e_exc: float,
    e_inh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i_exc = g_exc * (e_exc - v)
    i_inh = g_inh * (e_inh - v)
    dv = (vrest - v + i_exc + i_inh) / tc_m
    refractory_mask = ((dt * tcount) > (tlast + tref)).astype(np.float64)
    v_tmp = v + refractory_mask * dv * dt
    s = (v_tmp >= vthr).astype(np.uint8)
    tlast_next = tlast * (1.0 - s) + (dt * tcount) * s
    v_peaked = v_tmp * (1.0 - s) + vpeak * s
    v_next = v_peaked * (1.0 - s) + vreset * s
    return v_next, tlast_next, s


def run_mine_style_python_poisson_with_thresholds(
    thresholds: list[int],
    n_steps: int,
    seed: int,
    w_in: np.ndarray | None = None,
) -> list[int]:
    # Inference-only port of the update ordering in LIF_WTA_STDP_MNIST_mine.py::__call__
    # using deterministic fixed weights (current FPGA connectivity) instead of learned W_in.
    dt = MINE_DT
    n = N_NEURONS
    n_in = N_IN

    rng_state = seed & 0xFFFFFFFF

    # Connectivity / gains
    if w_in is None:
        w_in = _build_fixed_w_in_for_mine_like()
    inh_coeff = MINE_WINH / (n - 1)

    # Synapse/delay states
    input_td = 1e-3
    exc_td = 1e-3
    inh_td = 2e-3
    input_decay = 1.0 - dt / input_td
    input_scale = 1.0 / input_td
    c_in_state = np.zeros(n_in, dtype=np.float64)
    g_in_state = np.zeros(n, dtype=np.float64)
    exc_syn_r = np.zeros(n, dtype=np.float64)
    inh_syn_r = np.zeros(n, dtype=np.float64)
    delay_input = np.zeros((n, max(1, round(5e-3 / dt))), dtype=np.float64)
    delay_exc2inh = np.zeros((n, max(1, round(2e-3 / dt))), dtype=np.float64)
    g_inh = np.zeros(n, dtype=np.float64)

    # Excitatory neuron (DiehlAndCook2015LIF) state
    v_exc = np.full(n, -65.0, dtype=np.float64)
    tlast_exc = np.zeros(n, dtype=np.float64)
    theta = np.zeros(n, dtype=np.float64)
    vthr_exc = np.full(n, -52.0, dtype=np.float64)
    exc_tcount = 0

    # Inhibitory neuron (ConductanceBasedLIF) state
    v_inh = np.full(n, -45.0, dtype=np.float64)  # vreset at init
    tlast_inh = np.zeros(n, dtype=np.float64)
    vthr_inh = np.full(n, -40.0, dtype=np.float64)
    inh_tcount = 0

    spike_count = np.zeros(n, dtype=np.int64)
    thresholds_arr = np.asarray(thresholds, dtype=np.uint16)

    for _ in range(n_steps):
        # Poisson input generation (same RNG sequence as FPGA/simple reference)
        s_in = np.zeros(n_in, dtype=np.uint8)
        for i in range(n_in):
            rng_state = lcg_next_u32(rng_state)
            rand11 = (rng_state >> 21) & 0x7FF
            s_in[i] = 1 if rand11 < int(thresholds_arr[i]) else 0

        pre_active = np.flatnonzero(s_in)

        # Input layer / synapses (mine.py ordering)
        c_in_state = c_in_state * input_decay + input_scale * s_in.astype(np.float64)
        # input_synaptictrace x_in exists in mine.py but not needed for inference (stdp=False)

        g_in_state *= input_decay
        if pre_active.size > 0:
            # Equivalent to add_columns_scaled_inplace(g_in_state, W_in, pre_active, input_scale)
            g_in_state += input_scale * np.sum(w_in[:, pre_active], axis=1)
        delayed_g_in, delay_input = _delay_step(delay_input, g_in_state)

        # Excitatory layer (DiehlAndCook2015LIF)
        v_exc, tlast_exc, s_exc = _conductance_lif_step(
            v_exc, tlast_exc, exc_tcount, delayed_g_in, g_inh,
            dt=dt, tref=5e-3, tc_m=1e-1,
            vrest=-65.0, vreset=-65.0, vthr=vthr_exc, vpeak=20.0,
            e_exc=0.0, e_inh=-100.0,
        )
        theta = (1.0 - dt / 1e4) * theta + 0.05 * s_exc.astype(np.float64)
        theta = np.clip(theta, 0.0, 35.0)
        vthr_exc = theta + (-52.0)
        exc_tcount += 1
        spike_count += s_exc.astype(np.int64)

        # Excitatory synapse -> inhibitory delay
        exc_syn_r = _single_exp_step(exc_syn_r, s_exc.astype(np.float64), dt, exc_td)
        g_exc = MINE_WEXC * exc_syn_r
        delayed_g_exc, delay_exc2inh = _delay_step(delay_exc2inh, g_exc)

        # Inhibitory layer
        v_inh, tlast_inh, s_inh = _conductance_lif_step(
            v_inh, tlast_inh, inh_tcount, delayed_g_exc, np.zeros(n, dtype=np.float64),
            dt=dt, tref=2e-3, tc_m=1e-2,
            vrest=-60.0, vreset=-45.0, vthr=vthr_inh, vpeak=20.0,
            e_exc=0.0, e_inh=-85.0,
        )
        inh_tcount += 1

        # Inhibitory synapse and WTA inhibition
        inh_syn_r = _single_exp_step(inh_syn_r, s_inh.astype(np.float64), dt, inh_td)
        sum_c_inh = float(np.sum(inh_syn_r))
        g_inh = inh_coeff * (sum_c_inh - inh_syn_r)

    return spike_count.astype(np.int64).tolist()


def simulate_poisson_debug_counts(thresholds: list[int], n_steps: int, seed: int) -> dict[str, int]:
    rng_state = seed & 0xFFFFFFFF
    total_input = 0
    first_step = 0
    last_step = 0

    for t in range(n_steps):
        step_count = 0
        for i in range(N_IN):
            rng_state = lcg_next_u32(rng_state)
            rand11 = (rng_state >> 21) & 0x7FF
            s = 1 if rand11 < thresholds[i] else 0
            step_count += s
        total_input += step_count
        if t == 0:
            first_step = step_count
        if t == n_steps - 1:
            last_step = step_count

    return {
        "total_input_spikes_generated": int(total_input),
        "first_step_input_spikes": int(first_step),
        "last_step_input_spikes": int(last_step),
    }


def compare_counts(fpga_counts: list[int], py_counts: list[int]) -> None:
    diffs = [abs(a - b) for a, b in zip(fpga_counts, py_counts)]
    mismatch = sum(1 for d in diffs if d != 0)
    max_diff = max(diffs) if diffs else 0

    fpga_total = sum(fpga_counts)
    py_total = sum(py_counts)
    fpga_mean = fpga_total / max(1, len(fpga_counts))
    py_mean = py_total / max(1, len(py_counts))

    print(f"Compare spike_count[100]: mismatched={mismatch}, max_diff={max_diff}")
    print(
        "Stats: "
        f"fpga_total={fpga_total}, py_total={py_total}, "
        f"fpga_mean={fpga_mean:.3f}, py_mean={py_mean:.3f}"
    )
    if mismatch == 0:
        print("Exact match: FPGA and Python spike counts are identical.")
    else:
        print("All mismatches:")
        for i, (f, p, d) in enumerate(zip(fpga_counts, py_counts, diffs)):
            if d != 0:
                print(f"  neuron {i}: fpga={f}, python={p}, diff={d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy RAW1 from SD to DDR, run FPGA Poisson+LIF(WTA+inhibition) inference, "
            "compare with Python reproduction, then run FPGA add."
        )
    )
    parser.add_argument("a", type=int, nargs="?", default=0, help="integer add operand A")
    parser.add_argument("b", type=int, nargs="?", default=0, help="integer add operand B")
    parser.add_argument(
        "--image-source",
        type=str,
        choices=["fpga", "mnist", "raw"],
        default="fpga",
        help="image source for Python-side comparison (default: fpga)",
    )
    parser.add_argument("--sample-idx", type=int, default=0, help="MNIST sample index for --image-source mnist")
    parser.add_argument(
        "--raw-bin",
        type=str,
        default=None,
        help="RAW1 file written by import_MNIST_raw.py (same file used for SD write)",
    )
    parser.add_argument("--port", type=str, default=SERIAL_PORTNAME)
    parser.add_argument("--start-lba", type=int, default=2048)
    parser.add_argument("--num-sectors", type=int, default=0, help="0 means auto from RAW1 header")
    parser.add_argument("--n-steps", type=int, default=350)
    parser.add_argument("--seed", type=lambda x: int(x, 0), default=0x12345678)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--poisson-only", action="store_true", help="validate only Poisson spike generation (skip LIF count comparison)")
    parser.add_argument(
        "--weights-npy",
        type=str,
        default=None,
        help="Optional .npy/.npz weight matrix (shape [100,784]); used for Python compare and optionally UART upload",
    )
    parser.add_argument(
        "--upload-weights",
        action="store_true",
        help="Upload weights to FPGA over UART before inference (default: off; FPGA initializes built-in weights)",
    )
    parser.add_argument(
        "--python-model",
        choices=["mine", "simple"],
        default="mine",
        help="Python reference model for spike-count comparison (default: mine)",
    )
    parser.add_argument(
        "--print-train-ddr-map",
        action="store_true",
        help="print proposed training DDR logical memory map (Step1) and continue",
    )
    parser.add_argument(
        "--train-kernel-selfcheck",
        action="store_true",
        help="run Python self-check for trace/STDP tile kernels (Step2/Step3) and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.train_kernel_selfcheck:
        layout = build_train_ddr_layout()
        print("Training DDR logical map (word addr):")
        print(
            f"  W={layout.base_w_q16}, A={layout.base_a_q16}, B_T={layout.base_bt_q16}, "
            f"theta={layout.base_exc_theta}, v={layout.base_v_state}, "
            f"delay={layout.base_delay_lines}, g_in={layout.base_g_in_state}, total={layout.total_words}"
        )
        selfcheck_training_kernels()
        raise SystemExit(0)

    if args.print_train_ddr_map:
        layout = build_train_ddr_layout()
        print("Training DDR logical map (word addr):")
        print(f"  base_w_q16      = {layout.base_w_q16}")
        print(f"  base_a_q16      = {layout.base_a_q16}")
        print(f"  base_bt_q16     = {layout.base_bt_q16}")
        print(f"  base_exc_theta  = {layout.base_exc_theta}")
        print(f"  base_v_state    = {layout.base_v_state}")
        print(f"  base_delay      = {layout.base_delay_lines}")
        print(f"  base_g_in_state = {layout.base_g_in_state}")
        print(f"  total_words     = {layout.total_words}")
    if serial is None:
        raise RuntimeError(
            "pyserial is not installed. Install it to use FPGA communication paths "
            "(self-check mode works without pyserial)."
        )
    print(f"Opening serial port {args.port}")

    with serial.Serial(
        args.port,
        BAUD,
        timeout=TIMEOUT_SEC,
        write_timeout=WRITE_TIMEOUT_SEC
    ) as ser:
        try:
            caps = fpga_train_query_caps(ser)
            print(f"Train kernel caps: 0x{caps:08X}")
        except Exception as exc:
            print(f"Train kernel caps query skipped/failed: {exc}")

        if args.weights_npy:
            fpga_weights_q16 = load_weight_matrix_q16_from_file(args.weights_npy)
            print(f"Loaded weights from file: {args.weights_npy}")
        else:
            fpga_weights_q16 = build_fixed_weight_matrix_q16()
            print("Using built-in fixed wiring weights (matches FPGA default initialization)")
        if args.upload_weights:
            fpga_write_infer_weights(ser, fpga_weights_q16)
        else:
            print("Skipping UART weight upload; using FPGA-side weight initialization")
        py_w_in = fpga_weights_q16.astype(np.float64) / float(1 << FXP_SHIFT)

        fpga_sd_to_ddr_copy(
            ser=ser,
            start_lba=args.start_lba,
            num_sectors=args.num_sectors,
            timeout_sec=args.timeout
        )

        if args.image_source == "fpga":
            image0_u8 = fpga_read_raw_image_u8(ser)
            label0 = -1
            print("Loaded comparison image from FPGA raw_image0_u8 (label=unknown)")
        elif args.image_source == "mnist":
            image0_u8, label0 = read_mnist_image_u8(args.sample_idx)
            print(f"Loaded comparison image from MNIST: sample_idx={args.sample_idx} (label={label0})")
        else:
            raw_bin_path = resolve_raw_bin_path(args.raw_bin)
            image0_u8, label0 = read_raw1_first_image_u8(str(raw_bin_path))
            print(f"Loaded comparison image from RAW1: {raw_bin_path} (label={label0})")

        sum_u8 = int(sum(image0_u8))
        nz = sum(1 for v in image0_u8 if v != 0)
        mod4_sum = [0, 0, 0, 0]
        for i, v in enumerate(image0_u8):
            mod4_sum[i & 3] += int(v)
        print(f"Image stats: sum_u8={sum_u8}, nonzero_pixels={nz}, sum_mod4={mod4_sum}")

        py_thresh = build_poisson_thresholds_u11(image0_u8)
        py_thresh_sum = int(sum(py_thresh))
        py_thresh_max = max(py_thresh) if py_thresh else 0
        print(f"Python threshold stats: sum={py_thresh_sum}, max={py_thresh_max}")
        print(
            "WTA/inhibition params (S16.16): "
            f"FXP_WEXC={FXP_WEXC}, FXP_INH_COEFF={FXP_INH_COEFF}, FXP_INH_THRESH={FXP_INH_THRESH}"
        )

        fpga_run_sample_infer(
            ser=ser,
            seed=args.seed,
            n_steps=args.n_steps,
            timeout_sec=args.timeout
        )
        fpga_counts = fpga_read_spike_counts(ser)

        fpga_thresh = fpga_read_poisson_thresh(ser)
        fpga_thresh_sum = int(sum(fpga_thresh))
        fpga_thresh_max = max(fpga_thresh) if fpga_thresh else 0
        thresh_mismatch = sum(1 for a, b in zip(fpga_thresh, py_thresh) if a != b)
        print(
            "FPGA threshold stats: "
            f"sum={fpga_thresh_sum}, max={fpga_thresh_max}, "
            f"mismatched_vs_python={thresh_mismatch}"
        )

        infer_dbg = fpga_read_infer_debug(ser)
        print("Infer debug counters:")
        print(
            "  "
            f"total_input_spikes_generated={infer_dbg['total_input_spikes_generated']}, "
            f"total_syn_hits_applied={infer_dbg['total_syn_hits_applied']}, "
            f"last_step_input_spikes={infer_dbg['last_step_input_spikes']}, "
            f"first_step_input_spikes={infer_dbg['first_step_input_spikes']}"
        )
        print(
            "  "
            f"first_step_hits_n0={infer_dbg['first_step_hits_n0']}, "
            f"first_step_hits_n3={infer_dbg['first_step_hits_n3']}, "
            f"first_step_hits_n7={infer_dbg['first_step_hits_n7']}"
        )
        print(
            "  "
            f"infer_total_spikes={infer_dbg['infer_total_spikes']}, "
            f"infer_steps_target={infer_dbg['infer_steps_target']}, "
            f"infer_step_idx={infer_dbg['infer_step_idx']}, "
            f"infer_state={infer_dbg['infer_state']}, "
            f"raw_image0_sum_u8={infer_dbg['raw_image0_sum_u8']}"
        )

        py_poisson_dbg = simulate_poisson_debug_counts(
            thresholds=py_thresh,
            n_steps=args.n_steps,
            seed=args.seed
        )
        print("Poisson-only check (FPGA vs Python):")
        for k in ["total_input_spikes_generated", "first_step_input_spikes", "last_step_input_spikes"]:
            fv = infer_dbg[k]
            pv = py_poisson_dbg[k]
            print(f"  {k}: fpga={fv}, python={pv}, diff={int(fv)-int(pv)}")

        if args.poisson_only:
            result = fpga_add(ser, args.a, args.b)
            print(f"FPGA result: {args.a} + {args.b} = {result}")
            raise SystemExit(0)

        if args.python_model == "mine":
            print("Python compare model: mine-style inference (no STDP, fixed FPGA wiring weights)")
            py_counts = run_mine_style_python_poisson_with_thresholds(
                thresholds=py_thresh,
                n_steps=args.n_steps,
                seed=args.seed,
                w_in=py_w_in,
            )
            py_counts_from_fpga_thresh = run_mine_style_python_poisson_with_thresholds(
                thresholds=fpga_thresh,
                n_steps=args.n_steps,
                seed=args.seed,
                w_in=py_w_in,
            )
        else:
            print("Python compare model: simple fixed-point debug model")
            py_counts = run_fixed_point_python_poisson(
                image_u8=image0_u8,
                n_steps=args.n_steps,
                seed=args.seed
            )
            py_counts_from_fpga_thresh = run_fixed_point_python_poisson_with_thresholds(
                thresholds=fpga_thresh,
                n_steps=args.n_steps,
                seed=args.seed
            )
        compare_counts(fpga_counts, py_counts)
        print("Compare using FPGA-read thresholds:")
        compare_counts(fpga_counts, py_counts_from_fpga_thresh)

        result = fpga_add(ser, args.a, args.b)
        print(f"FPGA result: {args.a} + {args.b} = {result}")
