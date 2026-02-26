from __future__ import annotations

import argparse
import importlib
import importlib.util
import struct
import sys
import time
import types
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
TRAIN_KERNEL_TIMEOUT_SEC = 30.0

# Protocol constants
REQ_SYNC = 0xA5
RESP_SYNC = 0x5A
PROTO_VER = 0x01

OP_ADD_I32 = 0x01
OP_DDR_WRITE32 = 0x10
OP_SD_TO_DDR_COPY = 0x11
OP_DDR_READ32 = 0x12
OP_SD_SECTORS_TO_DDR = 0x13
OP_LOAD_IMAGE_FROM_DDR = 0x14
OP_DDR_ZERO32 = 0x15
OP_RUN_SAMPLE_INFER = 0x20
OP_READ_SPIKE_COUNT = 0x21
OP_READ_RAW_U8 = 0x22
OP_READ_POISSON_THRESH = 0x23
OP_READ_INFER_DEBUG = 0x24
OP_WRITE_INFER_WEIGHT = 0x25
OP_SET_POISSON_MAX_FR = 0x26
OP_TRAIN_QUERY_CAPS = 0x30
OP_TRACE_UPDATE = 0x31
OP_STDP_UPDATE_TILE = 0x32
OP_TRAIN_GEN_WORK = 0x33
OP_READ_TRAIN_DEBUG = 0x34
OP_STDP_UPDATE_ALL = 0x35
OP_TRAIN_RUN_CHUNK = 0x36
OP_TRAIN_RUN_SAMPLE_PHASE3 = 0x37
OP_TRAIN_RUN_SAMPLE_PHASE4 = 0x38

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
RAW1_HEADER_BYTES = 20
RAW1_NUM_IMAGES_DEFAULT = 10_000
RAW1_TOTAL_BYTES_DEFAULT = RAW1_HEADER_BYTES + RAW1_NUM_IMAGES_DEFAULT + (RAW1_NUM_IMAGES_DEFAULT * N_IN)
RAW1_TOTAL_SECTORS_DEFAULT = (RAW1_TOTAL_BYTES_DEFAULT + 511) // 512

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
    base_x_in_work: int
    base_x_exc_work: int
    base_prelist_work: int
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
    base_x_in_work = word                # [N_IN] s32 (Q16.16 traces)
    word += N_IN
    base_x_exc_work = word               # [N_NEURONS] s32 (Q16.16 traces)
    word += N_NEURONS
    base_prelist_work = word             # [N_IN] u32 indices (max pre_active count)
    word += N_IN
    return TrainDDRLayout(
        base_w_q16=base_w_q16,
        base_a_q16=base_a_q16,
        base_bt_q16=base_bt_q16,
        base_exc_theta=base_exc_theta,
        base_v_state=base_v_state,
        base_delay_lines=base_delay_lines,
        base_g_in_state=base_g_in_state,
        base_x_in_work=base_x_in_work,
        base_x_exc_work=base_x_exc_work,
        base_prelist_work=base_prelist_work,
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


def fpga_sd_sectors_to_ddr(
    ser: serial.Serial,
    start_lba: int,
    num_sectors: int,
    timeout_sec: float = 120.0,
) -> int:
    print(f"Requesting SD sectors->DDR DMA: start_lba={start_lba}, sectors={num_sectors}")
    t0 = time.time()
    status, result = send_request(
        ser=ser,
        opcode=OP_SD_SECTORS_TO_DDR,
        args=[start_lba, num_sectors],
        response_timeout=timeout_sec,
    )
    require_ok(status, "SD sectors->DDR")
    elapsed = time.time() - t0
    print(f"SD sectors->DDR completed in {elapsed:.2f}s, words_written={result}")
    return result


def fpga_load_image_from_ddr(
    ser: serial.Serial,
    base_addr_byte: int,
    n_bytes: int = N_IN,
    timeout_sec: float = 120.0,
) -> None:
    if n_bytes <= 0 or n_bytes > N_IN:
        raise ValueError(f"n_bytes must be in [1, {N_IN}], got {n_bytes}")
    print(f"Requesting DDR->raw_image0 load: base_byte=0x{base_addr_byte:08X}, n_bytes={n_bytes}")
    status, result = send_request(
        ser=ser,
        opcode=OP_LOAD_IMAGE_FROM_DDR,
        args=[base_addr_byte, n_bytes],
        response_timeout=min(float(timeout_sec), 5.0),
    )
    require_ok(status, "LOAD_IMAGE_FROM_DDR")
    if int(result) != n_bytes:
        raise RuntimeError(f"LOAD_IMAGE_FROM_DDR returned unexpected byte count: {result} (expected {n_bytes})")


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
        if idx and (idx % 32 == 0):
            time.sleep(0.001)
        status, value = send_request(
            ser,
            OP_READ_RAW_U8,
            [idx, 0],
            response_timeout=1.0,
            transient_retry_max=max(TRANSIENT_RETRY_MAX, 20),
        )
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
        12: "poisson_num_const_cfg",
    }
    out: dict[str, int] = {}
    for idx, name in names.items():
        status, value = send_request(ser, OP_READ_INFER_DEBUG, [idx, 0])
        if status != STATUS_OK:
            # Backward compatibility: older bitstreams expose indices 0..11 only.
            if idx >= 12:
                continue
            require_ok(status, f"READ_INFER_DEBUG[{idx}]")
        out[name] = int(value) & 0xFFFFFFFF
    return out


def fpga_add(ser: serial.Serial, a: int, b: int) -> int:
    print(f"Sending ADD request: {a} + {b}")
    status, result = send_request(ser, OP_ADD_I32, [a, b], response_timeout=TIMEOUT_SEC)
    require_ok(status, "ADD request")
    return result


def fpga_ddr_write32(
    ser: serial.Serial,
    addr_word: int,
    value: int,
    *,
    response_timeout: float = TIMEOUT_SEC,
    transient_retry_max: int = TRANSIENT_RETRY_MAX,
) -> int:
    status, result = send_request(
        ser,
        OP_DDR_WRITE32,
        [int(addr_word), int(value)],
        response_timeout=response_timeout,
        transient_retry_max=transient_retry_max,
    )
    require_ok(status, f"DDR_WRITE32[{addr_word}]")
    return int(result)


def fpga_ddr_read32(
    ser: serial.Serial,
    addr_word: int,
    *,
    response_timeout: float = TIMEOUT_SEC,
    transient_retry_max: int = TRANSIENT_RETRY_MAX,
) -> int:
    status, result = send_request(
        ser,
        OP_DDR_READ32,
        [int(addr_word), 0],
        response_timeout=response_timeout,
        transient_retry_max=transient_retry_max,
    )
    require_ok(status, f"DDR_READ32[{addr_word}]")
    return int(result)


def fpga_ddr_smoke_test(ser: serial.Serial, base_addr_word: int = 0x100) -> None:
    print(f"Running DDR smoke test @ word_addr=0x{base_addr_word:08X}")
    patterns = [0x11223344, 0x89ABCDEF, 0x00000000, 0x55AA33CC]
    for i, pat in enumerate(patterns):
        addr = base_addr_word + i
        fpga_ddr_write32(ser, addr, pat)
    for i, pat in enumerate(patterns):
        addr = base_addr_word + i
        got = fpga_ddr_read32(ser, addr) & 0xFFFFFFFF
        print(f"  DDR[{addr}] -> 0x{got:08X} (exp 0x{pat & 0xFFFFFFFF:08X})")
        if got != (pat & 0xFFFFFFFF):
            raise RuntimeError(f"DDR smoke test mismatch at addr {addr}: got 0x{got:08X}")
    print("DDR smoke test passed.")


def fpga_ddr_zero32(ser: serial.Serial, base_addr_word: int, nwords: int) -> int:
    base = int(base_addr_word)
    rem = int(nwords)
    total_done = 0
    # HDL uses train_gen_count_total[15:0], so one request can zero at most 65535 words.
    max_chunk = 0xFFFF
    while rem > 0:
        chunk = rem if rem <= max_chunk else max_chunk
        status, result = send_request(ser, OP_DDR_ZERO32, [base, chunk])
        require_ok(status, f"DDR_ZERO32[base={base},n={chunk}]")
        total_done += int(result) & 0xFFFF
        base += chunk
        rem -= chunk
    return total_done


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


def read_raw1_image_u8(raw_bin_path: str, sample_idx: int = 0) -> tuple[list[int], int]:
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
        if sample_idx < 0 or sample_idx >= num_images:
            raise ValueError(f"RAW1 sample_idx out of range: {sample_idx} (max={num_images-1})")

        labels = f.read(num_images)
        if len(labels) != num_images:
            raise ValueError("RAW1 labels are truncated")
        if sample_idx:
            f.seek(sample_idx * bytes_per_image, 1)
        img = f.read(bytes_per_image)
        if len(img) != bytes_per_image:
            raise ValueError("RAW1 image is truncated")
        return list(img), int(labels[sample_idx])


def read_raw1_first_image_u8(raw_bin_path: str) -> tuple[list[int], int]:
    return read_raw1_image_u8(raw_bin_path, sample_idx=0)


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


def float_to_s32_q16(arr: np.ndarray) -> np.ndarray:
    q = np.rint(np.asarray(arr, dtype=np.float64) * float(1 << FXP_SHIFT)).astype(np.int64)
    return np.clip(q, np.iinfo(np.int32).min, np.iinfo(np.int32).max)


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


def fxp_mul_s16_16_py(a: int, b: int) -> int:
    p = int(a) * int(b)
    if p >= 0:
        return to_s32((p + (1 << 15)) >> 16)
    return to_s32((p - (1 << 15)) >> 16)


def trunc_div_toward_zero(num: int, den: int) -> int:
    if den == 0:
        raise ZeroDivisionError("den must be non-zero")
    s = -1 if (num < 0) ^ (den < 0) else 1
    q = (abs(int(num)) // abs(int(den)))
    return s * q


def kernel_stdp_update_tile_q16_phase1_python(
    W_q16: np.ndarray,
    A_q16: np.ndarray,
    B_T_q16: np.ndarray,
    row0: int,
    nrows: int,
) -> None:
    row1 = min(int(row0) + int(nrows), int(W_q16.shape[0]))
    if row1 <= int(row0):
        return
    for r in range(int(row0), row1):
        for c in range(N_IN):
            w = to_s32(int(W_q16[r, c]))
            a = to_s32(int(A_q16[r, c]))
            bt = to_s32(int(B_T_q16[c, r]))
            pot = fxp_mul_s16_16_py(fxp_mul_s16_16_py(TRAIN_LR_P_Q16, to_s32(TRAIN_WMAX_Q16 - w)), a)
            dep = fxp_mul_s16_16_py(fxp_mul_s16_16_py(TRAIN_LR_M_Q16, w), bt)
            dW = to_s32(pot - dep)
            dW_step = trunc_div_toward_zero(dW, 100)
            if dW_step > TRAIN_CLIP_DW_Q16:
                dW_step = TRAIN_CLIP_DW_Q16
            elif dW_step < -TRAIN_CLIP_DW_Q16:
                dW_step = -TRAIN_CLIP_DW_Q16
            w_new = to_s32(w + dW_step)
            if w_new > TRAIN_WMAX_Q16:
                w_new = TRAIN_WMAX_Q16
            elif w_new < TRAIN_WMIN_Q16:
                w_new = TRAIN_WMIN_Q16
            W_q16[r, c] = np.int64(w_new)


def kernel_stdp_update_tile_q16_python(
    W_q16: np.ndarray,
    A_q16: np.ndarray,
    B_T_q16: np.ndarray,
    row0: int,
    nrows: int,
) -> None:
    row1 = min(int(row0) + int(nrows), int(W_q16.shape[0]))
    if row1 <= int(row0):
        return
    for r in range(int(row0), row1):
        sum_abs = 0
        for c in range(N_IN):
            sum_abs += abs(to_s32(int(W_q16[r, c])))
        if sum_abs == 0:
            sum_abs = 1
        for c in range(N_IN):
            w = to_s32(int(W_q16[r, c]))
            a = to_s32(int(A_q16[r, c]))
            bt = to_s32(int(B_T_q16[c, r]))
            w_norm = trunc_div_toward_zero(int(w) * int(TRAIN_NORM_Q16), int(sum_abs))
            w_norm = to_s32(w_norm)
            pot = fxp_mul_s16_16_py(fxp_mul_s16_16_py(TRAIN_LR_P_Q16, to_s32(TRAIN_WMAX_Q16 - w_norm)), a)
            dep = fxp_mul_s16_16_py(fxp_mul_s16_16_py(TRAIN_LR_M_Q16, w_norm), bt)
            dW = to_s32(pot - dep)
            dW_step = trunc_div_toward_zero(dW, 100)
            if dW_step > TRAIN_CLIP_DW_Q16:
                dW_step = TRAIN_CLIP_DW_Q16
            elif dW_step < -TRAIN_CLIP_DW_Q16:
                dW_step = -TRAIN_CLIP_DW_Q16
            w_new = to_s32(w_norm + dW_step)
            if w_new > TRAIN_WMAX_Q16:
                w_new = TRAIN_WMAX_Q16
            elif w_new < TRAIN_WMIN_Q16:
                w_new = TRAIN_WMIN_Q16
            W_q16[r, c] = np.int64(w_new)


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


def fpga_read_train_debug(ser: serial.Serial) -> dict[str, int]:
    keys = [
        "trace_active",
        "trace_state",
        "trace_a_idx",
        "trace_pre_idx",
        "trace_b_col_idx",
        "stdp_active",
        "stdp_state",
        "stdp_row_idx",
        "stdp_col_idx",
        "gen_active",
        "gen_state",
        "ddr_req_pending_core",
        "xin_cache_valid",
        "xexc_cache_valid",
        "ddr_req_addr_word_core",
        "ddr_req_word_count_core",
        "ddr_req_we_core",
        "ddr_req_from_train_core",
        "trace_tmp_x_val",
        "trace_tmp_mem_val",
        "gen_count_total",
        "gen_idx",
        "ddr_bridge_state",
        "ddr_wb_stall",
        "ddr_wb_ack",
        "ddr_req_toggle_core",
        "ddr_req_toggle_ddr_sync2",
        "ddr_req_toggle_ddr_seen",
        "ddr_rsp_toggle_ddr",
        "ddr_rsp_toggle_core_sync2",
        "ddr_rsp_toggle_core_seen",
        "ddr_req_we_ddr",
        "ddr_req_addr_word_ddr",
        "ddr_req_from_sd_ddr",
        "ddr_lane_sel_ddr",
        "chunk_active",
        "chunk_state",
        "chunk_mode",
        "chunk_last_infer_spikes",
        "chunk_last_blank_spikes",
        "chunk_retry_curr_max_fr",
        "chunk_retry_accepted_max_fr",
    ]
    out: dict[str, int] = {}
    for idx, key in enumerate(keys):
        status, value = send_request(ser, OP_READ_TRAIN_DEBUG, [idx, 0], response_timeout=1.0)
        require_ok(status, f"READ_TRAIN_DEBUG[{idx}]")
        out[key] = int(value) & 0xFFFFFFFF
    return out


def fpga_set_poisson_max_fr(ser: serial.Serial, max_fr: int) -> int:
    status, value = send_request(ser, OP_SET_POISSON_MAX_FR, [int(max_fr), 0])
    require_ok(status, f"SET_POISSON_MAX_FR[{int(max_fr)}]")
    return int(value) & 0xFFFFFFFF


def fpga_trace_update_kernel(
    ser: serial.Serial,
    *,
    winner_idx: int,
    pre_count: int,
) -> int:
    """Kernel trigger only. Data搬入(x_in/x_exc/prelist)は別opcode実装前提。"""
    time.sleep(0.01)
    try:
        status, result = send_request(
            ser,
            OP_TRACE_UPDATE,
            [winner_idx, pre_count],
            response_timeout=TRAIN_KERNEL_TIMEOUT_SEC,
            transient_retry_max=0,
        )
    except TimeoutError as exc:
        print(f"TRACE_UPDATE timeout after {TRAIN_KERNEL_TIMEOUT_SEC:.1f}s; probing train debug...")
        try:
            dbg = fpga_read_train_debug(ser)
            print("Train debug:", ", ".join(f"{k}={v}" for k, v in dbg.items()))
        except Exception as dbg_exc:
            print(f"Train debug probe failed: {dbg_exc}")
        raise exc
    if status == STATUS_BAD_PACKET and (((int(result) >> 24) & 0xFF) == 0x31):
        print("TRACE_UPDATE returned TRAIN_BUSY; probing train debug...")
        try:
            dbg = fpga_read_train_debug(ser)
            print("Train debug:", ", ".join(f"{k}={v}" for k, v in dbg.items()))
        except Exception as dbg_exc:
            print(f"Train debug probe failed: {dbg_exc}")
    require_ok(status, "TRACE_UPDATE")
    return int(result)


def fpga_stdp_update_tile(
    ser: serial.Serial,
    *,
    row0: int,
    nrows: int,
) -> int:
    status, result = send_request(
        ser,
        OP_STDP_UPDATE_TILE,
        [row0, nrows],
        response_timeout=TRAIN_KERNEL_TIMEOUT_SEC,
    )
    require_ok(status, "STDP_UPDATE_TILE")
    return int(result)


def fpga_stdp_update_all(ser: serial.Serial, *, tile_rows: int) -> int:
    status, result = send_request(
        ser,
        OP_STDP_UPDATE_ALL,
        [int(tile_rows), 0],
        response_timeout=max(TRAIN_KERNEL_TIMEOUT_SEC, 120.0),
        transient_retry_max=0,
    )
    require_ok(status, "STDP_UPDATE_ALL")
    return int(result)


def fpga_train_run_chunk(ser: serial.Serial, *, nsamples: int, tile_rows: int) -> int:
    timeout_s = max(TRAIN_KERNEL_TIMEOUT_SEC, 60.0)
    try:
        status, result = send_request(
            ser,
            OP_TRAIN_RUN_CHUNK,
            [int(nsamples), int(tile_rows)],
            response_timeout=timeout_s,
            transient_retry_max=0,
        )
    except TimeoutError as exc:
        print(f"TRAIN_RUN_CHUNK timeout after {timeout_s:.1f}s; probing train/infer debug...")
        try:
            tdbg = fpga_read_train_debug(ser)
            print("Train debug:", ", ".join(f"{k}={v}" for k, v in tdbg.items()))
        except Exception as dbg_exc:
            print(f"Train debug probe failed: {dbg_exc}")
        try:
            idbg = fpga_read_infer_debug(ser)
            print("Infer debug:", ", ".join(f"{k}={v}" for k, v in idbg.items()))
        except Exception as dbg_exc:
            print(f"Infer debug probe failed: {dbg_exc}")
        raise exc
    if status == STATUS_BAD_PACKET and (((int(result) >> 24) & 0xFF) == 0x31):
        print("TRAIN_RUN_CHUNK returned TRAIN_BUSY; probing train/infer debug...")
        try:
            tdbg = fpga_read_train_debug(ser)
            print("Train debug:", ", ".join(f"{k}={v}" for k, v in tdbg.items()))
        except Exception as dbg_exc:
            print(f"Train debug probe failed: {dbg_exc}")
        try:
            idbg = fpga_read_infer_debug(ser)
            print("Infer debug:", ", ".join(f"{k}={v}" for k, v in idbg.items()))
        except Exception as dbg_exc:
            print(f"Infer debug probe failed: {dbg_exc}")
    require_ok(status, "TRAIN_RUN_CHUNK")
    return int(result)


def fpga_train_run_chunk_phase1_trace_stdp(ser: serial.Serial, *, nsteps: int, tile_rows: int) -> int:
    if int(nsteps) <= 0:
        raise ValueError("nsteps must be > 0")
    # HDL phase1 uses arg0<0,arg1>0 to mean internal synthetic trace loop + one STDP batch.
    return fpga_train_run_chunk(ser, nsamples=-int(nsteps), tile_rows=int(tile_rows))


def fpga_train_run_chunk_phase2_infer_trace_stdp(ser: serial.Serial, *, nsteps: int, tile_rows: int) -> int:
    if int(nsteps) <= 0:
        raise ValueError("nsteps must be > 0")
    if int(tile_rows) <= 0:
        raise ValueError("tile_rows must be > 0")
    # HDL phase2 uses arg0<0,arg1<0 to mean: infer (-arg0 steps), then synthetic trace loop + STDP batch.
    return fpga_train_run_chunk(ser, nsamples=-int(nsteps), tile_rows=-int(tile_rows))


def fpga_train_run_sample_phase3(ser: serial.Serial, *, inj_steps: int, tile_rows: int) -> int:
    timeout_s = max(TRAIN_KERNEL_TIMEOUT_SEC, 300.0)
    try:
        status, result = send_request(
            ser,
            OP_TRAIN_RUN_SAMPLE_PHASE3,
            [int(inj_steps), int(tile_rows)],
            response_timeout=timeout_s,
            transient_retry_max=0,
        )
    except TimeoutError as exc:
        print(f"TRAIN_RUN_SAMPLE_PHASE3 timeout after {timeout_s:.1f}s; probing train/infer debug...")
        try:
            print("Train debug:", ", ".join(f"{k}={v}" for k, v in fpga_read_train_debug(ser).items()))
        except Exception as dbg_exc:
            print(f"Train debug probe failed: {dbg_exc}")
        try:
            print("Infer debug:", ", ".join(f"{k}={v}" for k, v in fpga_read_infer_debug(ser).items()))
        except Exception as dbg_exc:
            print(f"Infer debug probe failed: {dbg_exc}")
        raise exc
    require_ok(status, "TRAIN_RUN_SAMPLE_PHASE3")
    return int(result)


def fpga_train_run_sample_phase4(ser: serial.Serial, *, inj_steps: int, tile_rows: int) -> int:
    timeout_s = max(TRAIN_KERNEL_TIMEOUT_SEC, 300.0)
    try:
        status, result = send_request(
            ser,
            OP_TRAIN_RUN_SAMPLE_PHASE4,
            [int(inj_steps), int(tile_rows)],
            response_timeout=timeout_s,
            transient_retry_max=0,
        )
    except TimeoutError as exc:
        print(f"TRAIN_RUN_SAMPLE_PHASE4 timeout after {timeout_s:.1f}s; probing train/infer debug...")
        try:
            print("Train debug:", ", ".join(f"{k}={v}" for k, v in fpga_read_train_debug(ser).items()))
        except Exception as dbg_exc:
            print(f"Train debug probe failed: {dbg_exc}")
        try:
            print("Infer debug:", ", ".join(f"{k}={v}" for k, v in fpga_read_infer_debug(ser).items()))
        except Exception as dbg_exc:
            print(f"Infer debug probe failed: {dbg_exc}")
        raise exc
    require_ok(status, "TRAIN_RUN_SAMPLE_PHASE4")
    return int(result)


def fpga_train_run_sample_phase3_verify_stats(
    ser: serial.Serial,
    *,
    sample_idx: int,
    inj_steps: int,
    tile_rows: int,
    start_lba: int,
    timeout_sec: float,
    image_source: str,
    raw_bin: str | None,
) -> None:
    if image_source != "fpga":
        raise ValueError("phase3 verify currently requires --image-source fpga")
    prepare_fpga_sample_image_via_streamed_load(
        ser, sample_idx=int(sample_idx), start_lba=int(start_lba), timeout_sec=float(timeout_sec)
    )
    # Python reference image is read from the original MNIST dataset (same source family as import_MNIST_raw.py),
    # not from RAW1 file or FPGA UART readback. This avoids UART overhead and RAW-file dependency.
    image_u8, py_label = read_mnist_image_u8(int(sample_idx))
    print(f"Using MNIST dataset image for phase3 Python reference (sample_idx={int(sample_idx)}, label={py_label}).")
    thresholds = build_poisson_thresholds_u11(image_u8)
    py_inj, py_blank = run_mine_style_python_inj_blank_stats_with_thresholds(
        thresholds,
        inj_steps=int(inj_steps),
        blank_steps=150,
        seed=0x12345678,
    )

    # Reload image because phase3 mutates internal inference state and requires raw_image0_u8 preloaded.
    prepare_fpga_sample_image_via_streamed_load(
        ser, sample_idx=int(sample_idx), start_lba=int(start_lba), timeout_sec=float(timeout_sec)
    )
    ret = fpga_train_run_sample_phase3(ser, inj_steps=int(inj_steps), tile_rows=int(tile_rows))
    fpga_inj = int(ret) & 0xFFFF
    fpga_blank = (int(ret) >> 16) & 0xFFFF

    print("Phase3 stats compare (Python mine-style simple ref vs FPGA phase3):")
    print(f"  inj_total_spikes:   python={py_inj}, fpga={fpga_inj}, diff={fpga_inj - py_inj}")
    print(f"  blank_total_spikes: python={py_blank}, fpga={fpga_blank}, diff={fpga_blank - py_blank}")
    if py_inj != fpga_inj or py_blank != fpga_blank:
        raise RuntimeError("phase3 verify failed: inj/blank spike totals mismatch")


def fpga_train_run_sample_phase3_retry_coarse(
    ser: serial.Serial,
    *,
    sample_idx: int,
    inj_steps: int,
    tile_rows: int,
    start_lba: int,
    timeout_sec: float,
    image_source: str,
    seed: int = 0x12345678,
    max_fr_start: int = 32,
    max_fr_step: int = 16,
    max_fr_limit: int = 256,
    min_inj_spikes: int = 5,
) -> tuple[int, int, int, int]:
    """Mine-like coarse retry: probe infer-only with increasing max_fr, then run phase3 once."""
    if image_source != "fpga":
        raise ValueError("--train-run-sample-phase3-retry currently requires --image-source fpga")
    if max_fr_start <= 0 or max_fr_step <= 0 or max_fr_limit < max_fr_start:
        raise ValueError("invalid max_fr retry parameters")
    prepare_fpga_sample_image_via_streamed_load(
        ser, sample_idx=int(sample_idx), start_lba=int(start_lba), timeout_sec=float(timeout_sec)
    )

    accepted_max_fr = None
    probe_total = None
    tried: list[tuple[int, int]] = []
    for max_fr in range(int(max_fr_start), int(max_fr_limit) + 1, int(max_fr_step)):
        scaled = fpga_set_poisson_max_fr(ser, max_fr)
        print(
            f"Probing infer-only for max_fr={max_fr} "
            f"(poisson_num_const_cfg={scaled})..."
        )
        total = int(
            fpga_run_sample_infer(
                ser=ser,
                seed=int(seed),
                n_steps=int(inj_steps),
                timeout_sec=max(float(timeout_sec), 60.0),
            )
        )
        tried.append((int(max_fr), total))
        if total >= int(min_inj_spikes):
            accepted_max_fr = int(max_fr)
            probe_total = total
            break
    if accepted_max_fr is None:
        accepted_max_fr, probe_total = tried[-1]
        print(
            f"No retry candidate reached min_inj_spikes={int(min_inj_spikes)}; "
            f"using last max_fr={accepted_max_fr} (probe_total={probe_total})."
        )
    else:
        print(
            f"Accepted max_fr={accepted_max_fr} from infer-only probe "
            f"(inj_total_spikes={probe_total}, threshold={int(min_inj_spikes)})."
        )

    fpga_set_poisson_max_fr(ser, accepted_max_fr)
    ret = fpga_train_run_sample_phase3(ser, inj_steps=int(inj_steps), tile_rows=int(tile_rows))
    fpga_inj = int(ret) & 0xFFFF
    fpga_blank = (int(ret) >> 16) & 0xFFFF
    print(
        "TRAIN_RUN_SAMPLE_PHASE3 retry coarse completed: "
        f"accepted_max_fr={accepted_max_fr}, probe_inj_total={probe_total}, "
        f"phase3_inj_total={fpga_inj}, phase3_blank_total={fpga_blank}"
    )
    if probe_total is not None and int(probe_total) != int(fpga_inj):
        print(
            "Warning: phase3 inj_total differs from accepted infer-only probe "
            f"(probe={int(probe_total)}, phase3={int(fpga_inj)})."
        )
    # Restore mine.py default max_fr=32 equivalent for later commands.
    try:
        fpga_set_poisson_max_fr(ser, 32)
    except Exception as exc:
        print(f"Warning: failed to restore Poisson max_fr=32: {exc}")
    return int(accepted_max_fr), int(probe_total), int(fpga_inj), int(fpga_blank)


def prepare_fpga_sample_image_via_streamed_load(
    ser: serial.Serial,
    *,
    sample_idx: int,
    start_lba: int,
    timeout_sec: float,
) -> None:
    if int(sample_idx) < 0:
        raise ValueError(f"sample_idx must be >=0, got {sample_idx}")
    img_byte_off = RAW1_HEADER_BYTES + RAW1_NUM_IMAGES_DEFAULT + (int(sample_idx) * N_IN)
    img_sector_off = img_byte_off // 512
    img_byte_in_sector = img_byte_off % 512
    sectors_needed = (img_byte_in_sector + N_IN + 511) // 512
    print(
        "Preparing input image via streamed FPGA image load path: "
        f"sample_idx={int(sample_idx)}, sector_off={img_sector_off}, sectors={sectors_needed}, "
        f"byte_in_sector={img_byte_in_sector}"
    )
    fpga_sd_sectors_to_ddr(
        ser=ser,
        start_lba=int(start_lba) + img_sector_off,
        num_sectors=sectors_needed,
        timeout_sec=timeout_sec,
    )
    fpga_load_image_from_ddr(
        ser=ser,
        base_addr_byte=img_byte_in_sector,
        n_bytes=N_IN,
        timeout_sec=timeout_sec,
    )


def fpga_train_run_chunk_phase2_verify_infer_stats(
    ser: serial.Serial,
    *,
    sample_idx: int,
    nsteps: int,
    tile_rows: int,
    start_lba: int,
    timeout_sec: float,
) -> None:
    print("Preparing image for standalone inference baseline...")
    prepare_fpga_sample_image_via_streamed_load(
        ser, sample_idx=sample_idx, start_lba=start_lba, timeout_sec=timeout_sec
    )
    baseline_total = fpga_run_sample_infer(ser, seed=0x12345678, n_steps=int(nsteps))
    baseline_dbg = fpga_read_infer_debug(ser)

    print("Preparing image for TRAIN_RUN_CHUNK phase2 run...")
    prepare_fpga_sample_image_via_streamed_load(
        ser, sample_idx=sample_idx, start_lba=start_lba, timeout_sec=timeout_sec
    )
    chunk_total = fpga_train_run_chunk_phase2_infer_trace_stdp(
        ser, nsteps=int(nsteps), tile_rows=int(tile_rows)
    )
    chunk_dbg = fpga_read_infer_debug(ser)

    keys = [
        "infer_total_spikes",
        "infer_steps_target",
        "raw_image0_sum_u8",
    ]
    print("Phase2 infer stats compare (standalone vs chunk phase2):")
    mismatches = 0
    for k in keys:
        a = int(baseline_dbg.get(k, 0))
        b = int(chunk_dbg.get(k, 0))
        if a != b:
            mismatches += 1
        print(f"  {k}: baseline={a}, chunk={b}, diff={b-a}")
    print(
        f"  returned_total: baseline={int(baseline_total)}, chunk_return={int(chunk_total)}, "
        f"diff={int(chunk_total)-int(baseline_total)}"
    )
    if int(chunk_total) != int(baseline_total):
        raise RuntimeError("phase2 verify failed: returned infer_total_spikes mismatch")
    if mismatches != 0:
        raise RuntimeError(f"phase2 verify failed: infer debug mismatches={mismatches}")


def fpga_train_gen_work(ser: serial.Serial, *, seed: int, region: str) -> int:
    mode = {"x_in": 0, "x_exc": 1}[region]
    status, result = send_request(ser, OP_TRAIN_GEN_WORK, [int(seed), int(mode)])
    require_ok(status, f"TRAIN_GEN_WORK[{region}]")
    return int(result)


def fpga_ddr_write_block32(
    ser: serial.Serial,
    base_addr_word: int,
    values: list[int] | np.ndarray,
    *,
    label: str | None = None,
    progress_every: int = 128,
) -> None:
    vals = np.asarray(values, dtype=np.int64).reshape(-1)
    n = int(vals.size)
    if label:
        print(f"DDR write block start: {label}, base_word=0x{int(base_addr_word):08X}, nwords={n}")
    for i, v in enumerate(vals):
        if i and (i % 32 == 0):
            time.sleep(0.002)
        try:
            fpga_ddr_write32(
                ser,
                base_addr_word + i,
                int(v),
                response_timeout=1.0,
                transient_retry_max=max(TRANSIENT_RETRY_MAX, 20),
            )
        except Exception as exc:
            print(
                "DDR write block failed: "
                f"{label or '<unnamed>'} at i={i}/{n}, "
                f"addr_word=0x{int(base_addr_word + i):08X}, value=0x{(int(v) & 0xFFFFFFFF):08X}, "
                f"exc={exc}"
            )
            raise
        if label and progress_every > 0 and (((i + 1) % progress_every) == 0 or (i + 1) == n):
            print(
                f"  DDR write progress [{label}]: {i+1}/{n} "
                f"(last_addr=0x{int(base_addr_word + i):08X})"
            )


def fpga_ddr_read_block32(
    ser: serial.Serial,
    base_addr_word: int,
    nwords: int,
    *,
    label: str | None = None,
    progress_every: int = 128,
) -> np.ndarray:
    out = np.zeros(int(nwords), dtype=np.int64)
    if label:
        print(f"DDR read block start: {label}, base_word=0x{int(base_addr_word):08X}, nwords={int(nwords)}")
    for i in range(int(nwords)):
        if i and (i % 32 == 0):
            time.sleep(0.002)
        try:
            out[i] = np.int64(
                np.int32(
                    fpga_ddr_read32(
                        ser,
                        base_addr_word + i,
                        response_timeout=1.0,
                        transient_retry_max=max(TRANSIENT_RETRY_MAX, 20),
                    )
                )
            )
        except Exception as exc:
            print(
                "DDR read block failed: "
                f"{label or '<unnamed>'} at i={i}/{int(nwords)}, "
                f"addr_word=0x{int(base_addr_word + i):08X}, exc={exc}"
            )
            raise
        if label and progress_every > 0 and (((i + 1) % progress_every) == 0 or (i + 1) == int(nwords)):
            print(
                f"  DDR read progress [{label}]: {i+1}/{int(nwords)} "
                f"(last_addr=0x{int(base_addr_word + i):08X})"
            )
    return out


def fpga_trace_update_kernel_selfcheck(ser: serial.Serial, seed: int = 0) -> None:
    rng = np.random.RandomState(seed)
    layout = build_train_ddr_layout()
    seed_xin = 0x13579BDF
    seed_xexc = 0x2468ACE1
    x_in = gen_train_work_values_from_seed(seed_xin, N_IN)
    x_exc = gen_train_work_values_from_seed(seed_xexc, N_NEURONS)

    # Test 1: A-row update only (winner valid, pre_count=0)
    winner = int(rng.randint(0, N_NEURONS))
    print("Preloading TRACE_UPDATE selfcheck (A-row only)...")
    print(
        "DDR zero fill: "
        f"A_row_zero[winner={winner}], base_word=0x{int(layout.base_a_q16 + winner * N_IN):08X}, nwords={N_IN}"
    )
    fpga_ddr_zero32(ser, layout.base_a_q16 + winner * N_IN, N_IN)
    print(f"Generating x_in_work on FPGA (seed=0x{seed_xin:08X})...")
    fpga_train_gen_work(ser, seed=seed_xin, region="x_in")
    print(f"Running FPGA TRACE_UPDATE kernel (A-row only): winner={winner}, pre_count=0")
    fpga_trace_update_kernel(ser, winner_idx=winner, pre_count=0)
    A_fpga_row = fpga_ddr_read_block32(
        ser,
        layout.base_a_q16 + winner * N_IN,
        N_IN,
        label=f"A_row_readback[winner={winner}]",
        progress_every=196,
    ).astype(np.int64)
    a_diff = int(np.max(np.abs(A_fpga_row - x_in.astype(np.int64))))

    # Test 2: B_T row update only (winner=-1, pre_count=1)
    pre_idx = int(rng.randint(0, N_IN))
    print("Preloading TRACE_UPDATE selfcheck (B_T-row only)...")
    print(
        "DDR zero fill: "
        f"BT_row_zero[pre={pre_idx}], base_word=0x{int(layout.base_bt_q16 + pre_idx * N_NEURONS):08X}, nwords={N_NEURONS}"
    )
    fpga_ddr_zero32(ser, layout.base_bt_q16 + pre_idx * N_NEURONS, N_NEURONS)
    print(f"Generating x_exc_work on FPGA (seed=0x{seed_xexc:08X})...")
    fpga_train_gen_work(ser, seed=seed_xexc, region="x_exc")
    fpga_ddr_write_block32(
        ser,
        layout.base_prelist_work,
        np.array([pre_idx], dtype=np.int64),
        label="prelist_work[1]",
        progress_every=1,
    )
    print(f"Running FPGA TRACE_UPDATE kernel (B_T-row only): winner=-1, pre_count=1 (pre={pre_idx})")
    fpga_trace_update_kernel(ser, winner_idx=-1, pre_count=1)
    B_fpga_row = fpga_ddr_read_block32(
        ser,
        layout.base_bt_q16 + pre_idx * N_NEURONS,
        N_NEURONS,
        label=f"BT_row_readback[pre={pre_idx}]",
        progress_every=50,
    ).astype(np.int64)
    b_diff = int(np.max(np.abs(B_fpga_row - x_exc.astype(np.int64))))

    print(f"TRACE_UPDATE selfcheck: max|A_fpga-A_ref|={a_diff}, max|B_fpga-B_ref|={b_diff}")
    if a_diff != 0 or b_diff != 0:
        raise RuntimeError("TRACE_UPDATE selfcheck failed")


def fpga_stdp_update_tile_selfcheck(ser: serial.Serial, seed: int = 1) -> None:
    rng = np.random.RandomState(seed)
    layout = build_train_ddr_layout()
    row0 = 3
    nrows = 4
    row1 = row0 + nrows

    W0 = rng.randint(0, TRAIN_WMAX_Q16 + 1, size=(N_NEURONS, N_IN), dtype=np.int64)
    A0 = rng.randint(0, 1 << 15, size=(N_NEURONS, N_IN), dtype=np.int64)
    B0 = rng.randint(0, 1 << 15, size=(N_IN, N_NEURONS), dtype=np.int64)
    W_ref = np.array(W0, copy=True)
    kernel_stdp_update_tile_q16_python(W_ref, A0, B0, row0, nrows)

    print(f"Preloading STDP tile selfcheck data into DDR (rows {row0}..{row1-1})...")
    for r in range(row0, row1):
        fpga_ddr_write_block32(ser, layout.base_w_q16 + r * N_IN, W0[r, :])
        fpga_ddr_write_block32(ser, layout.base_a_q16 + r * N_IN, A0[r, :])
    for c in range(N_IN):
        fpga_ddr_write_block32(ser, layout.base_bt_q16 + c * N_NEURONS + row0, B0[c, row0:row1])

    print(f"Running FPGA STDP_UPDATE_TILE kernel (row-normalized): row0={row0}, nrows={nrows}")
    fpga_stdp_update_tile(ser, row0=row0, nrows=nrows)

    print("Reading back W tile from DDR for comparison...")
    max_diff = 0
    for r in range(row0, row1):
        w_fpga = fpga_ddr_read_block32(ser, layout.base_w_q16 + r * N_IN, N_IN).astype(np.int64)
        row_diff = int(np.max(np.abs(w_fpga - W_ref[r, :])))
        max_diff = max(max_diff, row_diff)
    print(f"STDP_UPDATE_TILE selfcheck (row-normalized): max|W_fpga-W_ref|={max_diff}")
    if max_diff != 0:
        raise RuntimeError("STDP_UPDATE_TILE selfcheck failed (row-normalized)")


def fpga_train_kernels_selfcheck_all(ser: serial.Serial) -> None:
    fpga_trace_update_kernel_selfcheck(ser)
    fpga_stdp_update_tile_selfcheck(ser)


def fpga_train_one_sample_e2e_selfcheck(ser: serial.Serial, seed: int = 7) -> None:
    """Small end-to-end training loop (trace repeated -> STDP tile), FPGA vs Python.

    This is a "1-sample equivalent" kernel-chain check for development:
    it uses synthetic per-timestep events/seeds but preserves the same update order
    (trace accumulation repeated, then STDP tile update) used by mine.py.
    """
    rng = np.random.RandomState(seed)
    layout = build_train_ddr_layout()
    row0 = 0
    nrows = 2
    row1 = row0 + nrows
    nsteps = 16  # keep runtime practical while validating the end-to-end ordering

    # Build a sparse synthetic "one-sample" event stream.
    xin_seeds = [int(rng.randint(0, 0x7FFFFFFF)) for _ in range(nsteps)]
    xexc_seeds = [int(rng.randint(0, 0x7FFFFFFF)) for _ in range(nsteps)]
    winners: list[int] = []
    pre_indices: list[int] = []
    for _ in range(nsteps):
        winners.append(int(rng.randint(row0, row1)) if (rng.rand() < 0.6) else -1)
        pre_indices.append(int(rng.randint(0, N_IN)) if (rng.rand() < 0.5) else -1)

    # Python reference buffers (integer q16 values for kernel-level exactness).
    A_ref = np.zeros((N_NEURONS, N_IN), dtype=np.int64)
    B_ref = np.zeros((N_IN, N_NEURONS), dtype=np.int64)
    W0 = rng.randint(0, TRAIN_WMAX_Q16 + 1, size=(N_NEURONS, N_IN), dtype=np.int64)
    W_ref = np.array(W0, copy=True)

    print(f"Preloading end-to-end training selfcheck data into DDR (rows {row0}..{row1-1})...")
    # Zero A rows used by STDP tile.
    for r in range(row0, row1):
        fpga_ddr_zero32(ser, layout.base_a_q16 + r * N_IN, N_IN)
    # Zero B_T tile columns for all pres (B_T[c, row0:row1]); required because STDP reads all c.
    for c in range(N_IN):
        fpga_ddr_zero32(ser, layout.base_bt_q16 + c * N_NEURONS + row0, nrows)
    # Preload initial W tile rows only (STDP tile touches only these rows).
    for r in range(row0, row1):
        fpga_ddr_write_block32(
            ser,
            layout.base_w_q16 + r * N_IN,
            W0[r, :],
            label=f"W_init_row[{r}]",
            progress_every=196,
        )

    print(f"Running end-to-end kernel chain for {nsteps} timesteps (trace only), then STDP tile...")
    for t in range(nsteps):
        x_in = gen_train_work_values_from_seed(xin_seeds[t], N_IN)
        x_exc = gen_train_work_values_from_seed(xexc_seeds[t], N_NEURONS)
        fpga_train_gen_work(ser, seed=xin_seeds[t], region="x_in")
        fpga_train_gen_work(ser, seed=xexc_seeds[t], region="x_exc")

        pre = pre_indices[t]
        if pre >= 0:
            fpga_ddr_write_block32(
                ser,
                layout.base_prelist_work,
                np.array([pre], dtype=np.int64),
                label=f"prelist[t={t}]",
                progress_every=1,
            )
            pre_active = np.array([pre], dtype=np.int64)
            pre_count = 1
        else:
            pre_active = np.empty(0, dtype=np.int64)
            pre_count = 0

        winner = winners[t]
        kernel_trace_update_python(A_ref, B_ref, x_in, x_exc, winner, pre_active)
        fpga_trace_update_kernel(ser, winner_idx=winner, pre_count=pre_count)

    kernel_stdp_update_tile_q16_python(W_ref, A_ref, B_ref, row0, nrows)
    fpga_stdp_update_tile(ser, row0=row0, nrows=nrows)

    print("Reading back end-to-end W tile from DDR for comparison...")
    max_diff = 0
    for r in range(row0, row1):
        w_fpga = fpga_ddr_read_block32(
            ser,
            layout.base_w_q16 + r * N_IN,
            N_IN,
            label=f"W_e2e_readback[{r}]",
            progress_every=196,
        ).astype(np.int64)
        row_diff = int(np.max(np.abs(w_fpga - W_ref[r, :])))
        max_diff = max(max_diff, row_diff)
    print(f"Train one-sample e2e selfcheck: max|W_fpga-W_ref|={max_diff} (nsteps={nsteps}, rows={nrows})")
    if max_diff != 0:
        raise RuntimeError("train one-sample e2e selfcheck failed")


class _CallRecorder:
    def __init__(self, inner):
        self._inner = inner
        self.last = None

    def __call__(self, *args, **kwargs):
        out = self._inner(*args, **kwargs)
        self.last = np.array(out, copy=True)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _install_mine_models_shim() -> None:
    if "Models" in sys.modules:
        return
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "STDP_no_supervising",
        repo_root / "snn_cuda" / "STDP_no_supervising",
    ]
    src_dir = next((p for p in candidates if p.exists()), None)
    if src_dir is None:
        raise ModuleNotFoundError("Models (and STDP_no_supervising fallback) not found")
    sys.path.insert(0, str(src_dir))
    pkg = types.ModuleType("Models")
    sys.modules["Models"] = pkg
    for name in ("Neurons", "Synapses", "Connections"):
        mod = importlib.import_module(name)
        setattr(pkg, name, mod)
        sys.modules[f"Models.{name}"] = mod


def _import_mine_reference_module():
    mine_path = Path(__file__).resolve().parents[1] / "LIF_WTA_STDP_MNIST_mine.py"
    if not mine_path.exists():
        raise FileNotFoundError(f"mine.py not found: {mine_path}")
    for attempt in (0, 1):
        try:
            spec = importlib.util.spec_from_file_location("mine_ref_mod", mine_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to create import spec for {mine_path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["mine_ref_mod"] = mod
            sys.path.insert(0, str(mine_path.parent))
            spec.loader.exec_module(mod)
            return mod
        except ModuleNotFoundError as exc:
            if attempt == 0 and exc.name == "Models":
                _install_mine_models_shim()
                continue
            raise


def _load_image_u8_for_train_replay(ser: serial.Serial, args: argparse.Namespace) -> tuple[list[int], int]:
    if args.image_source == "raw":
        raw_bin_path = resolve_raw_bin_path(args.raw_bin)
        return read_raw1_image_u8(str(raw_bin_path), sample_idx=args.sample_idx)
    if args.image_source == "mnist":
        return read_mnist_image_u8(args.sample_idx)
    img_idx = int(args.sample_idx)
    if img_idx < 0:
        raise ValueError(f"--sample-idx must be >=0 for --image-source fpga, got {img_idx}")
    img_byte_off = RAW1_HEADER_BYTES + RAW1_NUM_IMAGES_DEFAULT + (img_idx * N_IN)
    img_sector_off = img_byte_off // 512
    img_byte_in_sector = img_byte_off % 512
    sectors_needed = (img_byte_in_sector + N_IN + 511) // 512
    print(
        "Using streamed FPGA image load path for mine replay: "
        f"sample_idx={img_idx}, sector_off={img_sector_off}, sectors={sectors_needed}, "
        f"byte_in_sector={img_byte_in_sector}"
    )
    fpga_sd_sectors_to_ddr(
        ser=ser,
        start_lba=args.start_lba + img_sector_off,
        num_sectors=sectors_needed,
        timeout_sec=args.timeout,
    )
    fpga_load_image_from_ddr(
        ser=ser,
        base_addr_byte=img_byte_in_sector,
        n_bytes=N_IN,
        timeout_sec=args.timeout,
    )
    image_u8 = fpga_read_raw_image_u8(ser)
    label = -1
    if args.raw_bin:
        try:
            _, label = read_raw1_image_u8(str(resolve_raw_bin_path(args.raw_bin)), sample_idx=args.sample_idx)
        except Exception:
            label = -1
    return image_u8, label


def _record_mine_one_sample_events(
    image_u8: list[int],
    *,
    sample_seed: int,
    dt: float = 1e-3,
    nt_inj: int = 350,
    nt_blank: int = 150,
    init_max_fr: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], dict[str, int]]:
    mine = _import_mine_reference_module()
    img_f = (np.asarray(image_u8, dtype=np.float32).reshape(1, N_IN) / 255.0)
    np_state = np.random.get_state()
    try:
        np.random.seed(int(sample_seed) & 0xFFFFFFFF)
        net = mine.DiehlAndCook2015Network(
            n_in=N_IN,
            n_neurons=N_NEURONS,
            wexc=2.25,
            winh=0.85,
            dt=dt,
            wmin=0.0,
            wmax=5e-2,
            lr=(1e-2, 1e-4),
            update_nt=nt_inj,
            profile_every=0,
        )
        net.initialize_states()
        xin_tap = _CallRecorder(net.input_synaptictrace)
        xexc_tap = _CallRecorder(net.exc_synaptictrace)
        net.input_synaptictrace = xin_tap
        net.exc_synaptictrace = xexc_tap

        w_init = np.array(net.input_conn.W, copy=True)
        blank_input = np.zeros(N_IN, dtype=np.uint8)
        max_fr = int(init_max_fr)
        all_events: list[dict[str, object]] = []
        attempts = 0
        total_spikes = 0
        accepted_max_fr = max_fr
        while True:
            attempts += 1
            input_spikes = mine.online_load_and_encoding_dataset(img_f, 0, dt, nt_inj, max_fr)
            attempt_events: list[dict[str, object]] = []
            spike_sum = 0
            for t in range(nt_inj):
                s_in = np.asarray(input_spikes[t], dtype=np.uint8)
                pre_active = np.flatnonzero(s_in).astype(np.int64)
                s_exc = np.asarray(net(s_in, stdp=True), dtype=np.uint8)
                spike_sum += int(np.sum(s_exc))
                x_in = np.asarray(xin_tap.last, dtype=np.float64)
                x_exc = np.asarray(xexc_tap.last, dtype=np.float64)
                p = int(np.argmax(s_exc))
                winner = p if int(s_exc[p]) != 0 else -1
                attempt_events.append(
                    {
                        "t": t,
                        "pre_active": pre_active,
                        "winner": winner,
                        "x_in": x_in,
                        "x_exc": x_exc,
                    }
                )
            for _ in range(nt_blank):
                _ = net(blank_input, stdp=False)
            all_events.extend(attempt_events)
            total_spikes += spike_sum
            if spike_sum >= 5:
                accepted_max_fr = max_fr
                break
            max_fr += 16
        w_final = np.array(net.input_conn.W, copy=True)
        meta = {
            "attempts": attempts,
            "accepted_max_fr": int(accepted_max_fr),
            "nt_inj": int(nt_inj),
            "nt_blank": int(nt_blank),
            "total_events": int(len(all_events)),
            "total_exc_spikes": int(total_spikes),
        }
        return w_init, w_final, all_events, meta
    finally:
        np.random.set_state(np_state)


def fpga_train_mine_one_sample_replay_selfcheck(
    ser: serial.Serial,
    args: argparse.Namespace,
    *,
    seed: int = 123,
    tile_rows: int = 10,
    verify_row0: int | None = None,
    verify_nrows: int | None = None,
    verify_mode: str = "exact",
    verify_sample_cols: int = 32,
) -> None:
    """Replay a real mine.py 1-sample training update at kernel granularity.

    mine.py itself is executed as the reference to generate the actual per-timestep
    event stream (`x_in`, `x_exc`, pre_active, winner) for one sample. To keep UART
    traffic tractable, the replay uses the sufficient statistics (A/B_T) aggregated
    from that event stream, then runs FPGA STDP tile kernels across all rows.
    """
    image_u8, label = _load_image_u8_for_train_replay(ser, args)
    print(
        f"Running mine.py one-sample reference and recording events "
        f"(sample_idx={int(args.sample_idx)}, label={label if label >= 0 else 'unknown'})..."
    )
    w0_f, w1_f, events, meta = _record_mine_one_sample_events(image_u8, sample_seed=seed)
    print(
        "mine.py replay source stats: "
        f"attempts={meta['attempts']}, accepted_max_fr={meta['accepted_max_fr']}, "
        f"events={meta['total_events']}, total_exc_spikes={meta['total_exc_spikes']}"
    )

    A_f = np.zeros((N_NEURONS, N_IN), dtype=np.float64)
    B_T_f = np.zeros((N_IN, N_NEURONS), dtype=np.float64)
    for ev in events:
        kernel_trace_update_python(
            A_f,
            B_T_f,
            np.asarray(ev["x_in"], dtype=np.float64),
            np.asarray(ev["x_exc"], dtype=np.float64),
            int(ev["winner"]),
            np.asarray(ev["pre_active"], dtype=np.int64),
        )

    W0_q16 = float_to_q16_clip(w0_f, 0, TRAIN_WMAX_Q16).astype(np.int64)
    A_q16 = float_to_s32_q16(A_f).astype(np.int64)
    B_T_q16 = float_to_s32_q16(B_T_f).astype(np.int64)
    W_ref_q16 = np.array(W0_q16, copy=True)
    if verify_row0 is None:
        row0_sel = 0
    else:
        row0_sel = max(0, min(N_NEURONS - 1, int(verify_row0)))
    if verify_nrows is None:
        nrows_sel = N_NEURONS - row0_sel
    else:
        nrows_sel = max(1, min(N_NEURONS - row0_sel, int(verify_nrows)))
    row1_sel = row0_sel + nrows_sel
    for row0 in range(row0_sel, row1_sel, int(tile_rows)):
        kernel_stdp_update_tile_q16_python(W_ref_q16, A_q16, B_T_q16, row0, min(int(tile_rows), row1_sel - row0))
    W_mine_final_q16 = float_to_q16_clip(w1_f, 0, TRAIN_WMAX_Q16).astype(np.int64)

    layout = build_train_ddr_layout()
    print(
        "Preloading mine-replay W/A/B_T into DDR "
        f"(verify rows {row0_sel}..{row1_sel-1}; this can still take time over UART)..."
    )
    fpga_ddr_zero32(ser, layout.base_a_q16 + row0_sel * N_IN, nrows_sel * N_IN)
    # Zero only the B_T columns corresponding to selected post rows: shape [N_IN, nrows_sel]
    for c in range(N_IN):
        fpga_ddr_zero32(ser, layout.base_bt_q16 + c * N_NEURONS + row0_sel, nrows_sel)
    for r in range(row0_sel, row1_sel):
        fpga_ddr_write_block32(
            ser,
            layout.base_w_q16 + r * N_IN,
            W0_q16[r, :],
            label=f"W0_row[{r}]",
            progress_every=196,
        )
    touched_a_rows = np.flatnonzero(np.any(A_q16[row0_sel:row1_sel, :] != 0, axis=1)) + row0_sel
    touched_b_rows = np.flatnonzero(np.any(B_T_q16 != 0, axis=1))
    print(
        f"Writing touched traces only (selected rows): A_rows={int(touched_a_rows.size)}/{nrows_sel}, "
        f"B_T_rows={int(touched_b_rows.size)}/{N_IN}"
    )
    for r in touched_a_rows:
        fpga_ddr_write_block32(
            ser,
            layout.base_a_q16 + int(r) * N_IN,
            A_q16[int(r), :],
            label=f"A_row[{int(r)}]",
            progress_every=196,
        )
    for c in touched_b_rows:
        fpga_ddr_write_block32(
            ser,
            layout.base_bt_q16 + int(c) * N_NEURONS + row0_sel,
            B_T_q16[int(c), row0_sel:row1_sel],
            label=f"BT_row[{int(c)}]",
            progress_every=50,
        )

    caps = fpga_train_query_caps(ser)
    if (row0_sel == 0) and (row1_sel == N_NEURONS) and (caps & (1 << 12)):
        print(f"Running FPGA TRAIN_RUN_CHUNK phase0 (nsamples=1, tile_rows={int(tile_rows)})...")
        fpga_train_run_chunk(ser, nsamples=1, tile_rows=int(tile_rows))
    elif (row0_sel == 0) and (row1_sel == N_NEURONS) and (caps & (1 << 11)):
        print(f"Running FPGA STDP_UPDATE_ALL (tile_rows={int(tile_rows)})...")
        fpga_stdp_update_all(ser, tile_rows=int(tile_rows))
    else:
        print(
            f"Running FPGA STDP_UPDATE_TILE over selected rows "
            f"{row0_sel}..{row1_sel-1} (tile_rows={int(tile_rows)})..."
        )
        for row0 in range(row0_sel, row1_sel, int(tile_rows)):
            fpga_stdp_update_tile(ser, row0=row0, nrows=min(int(tile_rows), row1_sel - row0))

    mode = str(verify_mode).lower()
    if mode == "none":
        max_diff_fpga_vs_ref = -1
        max_diff_ref_vs_mine = int(np.max(np.abs(W_ref_q16[row0_sel:row1_sel, :] - W_mine_final_q16[row0_sel:row1_sel, :])))
        max_diff_fpga_vs_mine = -1
        print(
            "Mine one-sample replay selfcheck (no readback): "
            f"rows={row0_sel}..{row1_sel-1}, "
            f"max|W_ref_q16-W_mine_q16|={max_diff_ref_vs_mine}"
        )
        return

    if mode == "sampled":
        nsamp = max(1, min(N_IN, int(verify_sample_cols)))
        cols = np.linspace(0, N_IN - 1, nsamp, dtype=np.int64)
        cols = np.unique(cols)
        print(
            f"Reading back sampled W entries for rows {row0_sel}..{row1_sel-1} "
            f"(sampled_cols={int(cols.size)})..."
        )
        max_diff_fpga_vs_ref = 0
        max_diff_fpga_vs_mine = 0
        for r in range(row0_sel, row1_sel):
            for c in cols:
                got = np.int64(np.int32(fpga_ddr_read32(ser, layout.base_w_q16 + r * N_IN + int(c))))
                d1 = int(abs(int(got) - int(W_ref_q16[r, int(c)])))
                d2 = int(abs(int(got) - int(W_mine_final_q16[r, int(c)])))
                if d1 > max_diff_fpga_vs_ref:
                    max_diff_fpga_vs_ref = d1
                if d2 > max_diff_fpga_vs_mine:
                    max_diff_fpga_vs_mine = d2
        max_diff_ref_vs_mine = int(np.max(np.abs(W_ref_q16[row0_sel:row1_sel, cols] - W_mine_final_q16[row0_sel:row1_sel, cols])))
        print(
            "Mine one-sample replay selfcheck (sampled): "
            f"rows={row0_sel}..{row1_sel-1}, cols={int(cols.size)}, "
            f"max|W_fpga-W_ref_q16|={max_diff_fpga_vs_ref}, "
            f"max|W_ref_q16-W_mine_q16|={max_diff_ref_vs_mine}, "
            f"max|W_fpga-W_mine_q16|={max_diff_fpga_vs_mine}"
        )
        if max_diff_fpga_vs_ref != 0:
            raise RuntimeError("mine one-sample replay sampled selfcheck failed (FPGA vs q16 replay ref)")
        return

    print(f"Reading back selected W rows {row0_sel}..{row1_sel-1} for mine replay comparison...")
    W_fpga_q16 = np.array(W0_q16, copy=True)
    for r in range(row0_sel, row1_sel):
        W_fpga_q16[r, :] = fpga_ddr_read_block32(
            ser,
            layout.base_w_q16 + r * N_IN,
            N_IN,
            label=f"W_replay_readback[{r}]",
            progress_every=196,
        ).astype(np.int64)

    max_diff_fpga_vs_ref = int(np.max(np.abs(W_fpga_q16[row0_sel:row1_sel, :] - W_ref_q16[row0_sel:row1_sel, :])))
    max_diff_ref_vs_mine = int(np.max(np.abs(W_ref_q16[row0_sel:row1_sel, :] - W_mine_final_q16[row0_sel:row1_sel, :])))
    max_diff_fpga_vs_mine = int(np.max(np.abs(W_fpga_q16[row0_sel:row1_sel, :] - W_mine_final_q16[row0_sel:row1_sel, :])))
    print(
        "Mine one-sample replay selfcheck: "
        f"rows={row0_sel}..{row1_sel-1}, "
        f"max|W_fpga-W_ref_q16|={max_diff_fpga_vs_ref}, "
        f"max|W_ref_q16-W_mine_q16|={max_diff_ref_vs_mine}, "
        f"max|W_fpga-W_mine_q16|={max_diff_fpga_vs_mine}"
    )
    if max_diff_fpga_vs_ref != 0:
        raise RuntimeError("mine one-sample replay selfcheck failed (FPGA vs q16 replay ref)")


def lcg_next_u32(state: int) -> int:
    return (state * LCG_A + LCG_C) & 0xFFFFFFFF


def gen_train_work_values_from_seed(seed: int, n: int) -> np.ndarray:
    out = np.zeros(int(n), dtype=np.int64)
    state = int(seed) & 0xFFFFFFFF
    for i in range(int(n)):
        out[i] = np.int64((state >> 8) & 0x7FFF)
        state = lcg_next_u32(state)
    return out


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


def run_fixed_point_python_poisson_with_thresholds_event_driven(
    thresholds: list[int], n_steps: int, seed: int
) -> list[int]:
    """Same simple fixed-point reference as run_fixed_point_python_poisson_with_thresholds(),
    but accumulates per neuron using a per-step active-input prelist (event-driven style).

    This mirrors the intended HDL direction: generate input spikes once, store active indices,
    then scan weights only for active presynaptic events.
    """
    v = [0] * N_NEURONS
    v_inh = [0] * N_NEURONS
    c_inh = [0] * N_NEURONS
    g_inh = [0] * N_NEURONS
    g_exc_delay0 = [0] * N_NEURONS
    g_exc_delay1 = [0] * N_NEURONS
    spike_count = [0] * N_NEURONS
    rng_state = seed & 0xFFFFFFFF

    for _ in range(n_steps):
        pre_active: list[int] = []
        for i in range(N_IN):
            rng_state = lcg_next_u32(rng_state)
            rand11 = (rng_state >> 21) & 0x7FF
            if rand11 < thresholds[i]:
                pre_active.append(i)

        s_exc = [0] * N_NEURONS

        for n in range(N_NEURONS):
            accum = 0
            for i in pre_active:
                if (((i + n) & 0x3) == 0):
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


def validate_fixed_point_event_driven_inference_equivalence(
    thresholds: list[int] | None = None,
    *,
    n_steps: int = 100,
    seed: int = 0x12345678,
) -> tuple[int, int]:
    """Returns (mismatched_neurons, max_abs_diff) for dense vs event-driven fixed-point ref."""
    if thresholds is None:
        rng = np.random.RandomState(123)
        thresholds = [int(x) for x in rng.randint(0, RNG_MAX + 1, size=N_IN)]
    dense = run_fixed_point_python_poisson_with_thresholds(thresholds, int(n_steps), int(seed))
    sparse = run_fixed_point_python_poisson_with_thresholds_event_driven(thresholds, int(n_steps), int(seed))
    diffs = [abs(int(a) - int(b)) for a, b in zip(dense, sparse)]
    mismatched = int(sum(1 for d in diffs if d != 0))
    max_abs_diff = int(max(diffs) if diffs else 0)
    print(
        "Fixed-point infer event-driven equivalence: "
        f"mismatched_neurons={mismatched}, max_abs_diff={max_abs_diff}"
    )
    return mismatched, max_abs_diff


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


def run_mine_style_python_inj_blank_stats_with_thresholds(
    thresholds: list[int],
    inj_steps: int,
    blank_steps: int,
    seed: int,
    w_in: np.ndarray | None = None,
) -> tuple[int, int]:
    """Mine-style inference statistics for injection then blank (no-input), continuous state.

    Returns:
        (inj_total_spikes, blank_total_spikes)
    """
    dt = MINE_DT
    n = N_NEURONS
    n_in = N_IN

    rng_state = seed & 0xFFFFFFFF
    if w_in is None:
        w_in = _build_fixed_w_in_for_mine_like()
    inh_coeff = MINE_WINH / (n - 1)

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

    v_exc = np.full(n, -65.0, dtype=np.float64)
    tlast_exc = np.zeros(n, dtype=np.float64)
    theta = np.zeros(n, dtype=np.float64)
    vthr_exc = np.full(n, -52.0, dtype=np.float64)
    exc_tcount = 0

    v_inh = np.full(n, -45.0, dtype=np.float64)
    tlast_inh = np.zeros(n, dtype=np.float64)
    vthr_inh = np.full(n, -40.0, dtype=np.float64)
    inh_tcount = 0

    thresholds_arr = np.asarray(thresholds, dtype=np.uint16)

    def _run_steps(n_steps: int, *, force_no_input: bool) -> int:
        nonlocal rng_state, c_in_state, g_in_state, exc_syn_r, inh_syn_r, delay_input, delay_exc2inh, g_inh
        nonlocal v_exc, tlast_exc, theta, vthr_exc, exc_tcount, v_inh, tlast_inh, vthr_inh, inh_tcount
        total_spikes = 0
        for _ in range(int(n_steps)):
            s_in = np.zeros(n_in, dtype=np.uint8)
            if not force_no_input:
                for i in range(n_in):
                    rng_state = lcg_next_u32(rng_state)
                    rand11 = (rng_state >> 21) & 0x7FF
                    s_in[i] = 1 if rand11 < int(thresholds_arr[i]) else 0
            pre_active = np.flatnonzero(s_in)

            c_in_state = c_in_state * input_decay + input_scale * s_in.astype(np.float64)
            g_in_state *= input_decay
            if pre_active.size > 0:
                g_in_state += input_scale * np.sum(w_in[:, pre_active], axis=1)
            delayed_g_in, delay_input = _delay_step(delay_input, g_in_state)

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
            total_spikes += int(np.sum(s_exc, dtype=np.int64))

            exc_syn_r = _single_exp_step(exc_syn_r, s_exc.astype(np.float64), dt, exc_td)
            g_exc = MINE_WEXC * exc_syn_r
            delayed_g_exc, delay_exc2inh = _delay_step(delay_exc2inh, g_exc)

            v_inh, tlast_inh, s_inh = _conductance_lif_step(
                v_inh, tlast_inh, inh_tcount, delayed_g_exc, np.zeros(n, dtype=np.float64),
                dt=dt, tref=2e-3, tc_m=1e-2,
                vrest=-60.0, vreset=-45.0, vthr=vthr_inh, vpeak=20.0,
                e_exc=0.0, e_inh=-85.0,
            )
            inh_tcount += 1

            inh_syn_r = _single_exp_step(inh_syn_r, s_inh.astype(np.float64), dt, inh_td)
            sum_c_inh = float(np.sum(inh_syn_r))
            g_inh = inh_coeff * (sum_c_inh - inh_syn_r)
        return total_spikes

    inj_total = _run_steps(int(inj_steps), force_no_input=False)
    blank_total = _run_steps(int(blank_steps), force_no_input=True)
    return int(inj_total), int(blank_total)


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
    parser.add_argument(
        "--full-sd-copy",
        action="store_true",
        help="force legacy full RAW1 copy before inference (default fpga image path streams only required sectors)",
    )
    parser.add_argument("--n-steps", type=int, default=350)
    parser.add_argument("--seed", type=lambda x: int(x, 0), default=0x12345678)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--ddr-smoke", action="store_true", help="run DDR read/write smoke test before SD/inference")
    parser.add_argument("--ddr-smoke-addr", type=lambda x: int(x, 0), default=0x100, help="base word address for DDR smoke test")
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
    parser.add_argument(
        "--train-trace-fpga-selfcheck",
        action="store_true",
        help="run FPGA TRACE_UPDATE kernel self-check using DDR preload/readback and exit",
    )
    parser.add_argument(
        "--train-stdp-fpga-selfcheck",
        action="store_true",
        help="run FPGA STDP_UPDATE_TILE kernel self-check (phase1, no row normalization) and exit",
    )
    parser.add_argument(
        "--train-fpga-selfcheck-all",
        action="store_true",
        help="run FPGA trace+STDP kernel self-checks and exit",
    )
    parser.add_argument(
        "--train-one-sample-e2e-selfcheck",
        action="store_true",
        help="run a small end-to-end training kernel-chain selfcheck (trace loop + STDP tile) and exit",
    )
    parser.add_argument(
        "--train-mine-one-sample-replay-selfcheck",
        action="store_true",
        help="run a real mine.py one-sample replay selfcheck (aggregated A/B_T -> FPGA STDP tiles) and exit",
    )
    parser.add_argument(
        "--train-run-chunk-phase0",
        action="store_true",
        help="run phase0 coarse-grained FPGA train chunk (repeats STDP all-rows on current DDR-resident buffers) and exit",
    )
    parser.add_argument(
        "--train-run-chunk-phase1",
        action="store_true",
        help="run phase1 coarse-grained FPGA train chunk (internal synthetic trace loop + one STDP batch) and exit",
    )
    parser.add_argument(
        "--train-run-chunk-phase2",
        action="store_true",
        help="run phase2 coarse-grained FPGA train chunk (infer + synthetic trace loop + one STDP batch) and exit",
    )
    parser.add_argument(
        "--train-run-chunk-phase2-verify",
        action="store_true",
        help="compare standalone inference vs TRAIN_RUN_CHUNK phase2 using final infer statistics only",
    )
    parser.add_argument(
        "--train-run-sample-phase3",
        action="store_true",
        help="run phase3 coarse-grained FPGA sample flow (infer inj + synthetic trace/STDP + infer blank) and exit",
    )
    parser.add_argument(
        "--train-run-sample-phase3-verify",
        action="store_true",
        help="compare phase3 inj/blank spike totals against a Python mine-style simple reference",
    )
    parser.add_argument(
        "--train-run-sample-phase4",
        action="store_true",
        help="run phase4 coarse-grained FPGA sample flow (in-FPGA max_fr retry + synthetic trace/STDP + blank) and exit",
    )
    parser.add_argument(
        "--train-run-sample-phase3-retry",
        action="store_true",
        help="mine-like coarse retry: infer-only probes with increasing max_fr, then run one phase3 sample",
    )
    parser.add_argument("--chunk-nsamples", type=int, default=1, help="sample count for --train-run-chunk-phase0")
    parser.add_argument("--chunk-nsteps", type=int, default=16, help="step count for --train-run-chunk-phase1/phase2")
    parser.add_argument("--infer-max-fr", type=int, default=32, help="runtime max_fr for Poisson threshold scaling (OP_SET_POISSON_MAX_FR)")
    parser.add_argument("--train-retry-max-fr-start", type=int, default=32, help="starting max_fr for coarse phase3 retry")
    parser.add_argument("--train-retry-max-fr-step", type=int, default=16, help="max_fr increment for coarse phase3 retry")
    parser.add_argument("--train-retry-max-fr-limit", type=int, default=256, help="max_fr upper limit for coarse phase3 retry")
    parser.add_argument("--train-retry-min-inj-spikes", type=int, default=5, help="acceptance threshold on inj_total_spikes for coarse phase3 retry")
    parser.add_argument(
        "--train-mine-seed",
        type=int,
        default=123,
        help="seed for mine.py one-sample replay selfcheck (weight init + Poisson encoding)",
    )
    parser.add_argument(
        "--train-tile-rows",
        type=int,
        default=10,
        help="row tile size for FPGA STDP_UPDATE_TILE loops in training selfchecks",
    )
    parser.add_argument(
        "--train-verify-row0",
        type=int,
        default=0,
        help="start row for lightweight mine replay verification (default 0)",
    )
    parser.add_argument(
        "--train-verify-nrows",
        type=int,
        default=0,
        help="number of rows for lightweight mine replay verification (0 means all rows)",
    )
    parser.add_argument(
        "--train-verify-mode",
        choices=["exact", "sampled", "none"],
        default="exact",
        help="mine replay verification readback mode: exact rows, sampled columns, or none",
    )
    parser.add_argument(
        "--train-verify-sample-cols",
        type=int,
        default=32,
        help="number of sampled columns when --train-verify-mode sampled",
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
        print(f"  base_x_in_work  = {layout.base_x_in_work}")
        print(f"  base_x_exc_work = {layout.base_x_exc_work}")
        print(f"  base_prelist    = {layout.base_prelist_work}")
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
        # Let the USB-UART/FPGA side settle after open and drop any stale bytes from a prior run.
        time.sleep(0.05)
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            pass
        caps_ok = False
        try:
            caps = fpga_train_query_caps(ser)
            print(f"Train kernel caps: 0x{caps:08X}")
            caps_ok = True
        except Exception as exc:
            print(f"Train kernel caps query skipped/failed: {exc}")
            # One short retry after a small gap; this often recovers from port-open timing.
            try:
                time.sleep(0.05)
                ser.reset_input_buffer()
            except Exception:
                pass
            try:
                caps = fpga_train_query_caps(ser)
                print(f"Train kernel caps (retry): 0x{caps:08X}")
                caps_ok = True
            except Exception as exc2:
                print(f"Train kernel caps retry failed: {exc2}")

        needs_reliable_link = any([
            args.ddr_smoke,
            args.train_trace_fpga_selfcheck,
            args.train_stdp_fpga_selfcheck,
            args.train_fpga_selfcheck_all,
            args.train_one_sample_e2e_selfcheck,
            args.train_run_chunk_phase0,
            args.train_run_chunk_phase1,
            args.train_run_chunk_phase2,
            args.train_run_chunk_phase2_verify,
            args.train_run_sample_phase3,
            args.train_run_sample_phase3_verify,
            args.train_run_sample_phase4,
            args.train_run_sample_phase3_retry,
            args.train_mine_one_sample_replay_selfcheck,
        ])
        if needs_reliable_link and not caps_ok:
            raise RuntimeError(
                "FPGA UART link is not responding (TRAIN_QUERY_CAPS timeout). "
                "This often happens if a prior run left the FPGA busy/stuck. "
                "Reset/power-cycle the FPGA board and retry."
            )

        if args.train_trace_fpga_selfcheck:
            fpga_trace_update_kernel_selfcheck(ser)
            raise SystemExit(0)
        if args.train_stdp_fpga_selfcheck:
            fpga_stdp_update_tile_selfcheck(ser)
            raise SystemExit(0)
        if args.train_fpga_selfcheck_all:
            fpga_train_kernels_selfcheck_all(ser)
            raise SystemExit(0)
        if args.train_one_sample_e2e_selfcheck:
            fpga_train_one_sample_e2e_selfcheck(ser)
            raise SystemExit(0)
        if args.train_run_chunk_phase0:
            ret = fpga_train_run_chunk(
                ser,
                nsamples=max(1, int(args.chunk_nsamples)),
                tile_rows=max(1, int(args.train_tile_rows)),
            )
            print(
                "TRAIN_RUN_CHUNK phase0 completed: "
                f"result=0x{(int(ret) & 0xFFFFFFFF):08X}, nsamples={max(1,int(args.chunk_nsamples))}, "
                f"tile_rows={max(1,int(args.train_tile_rows))}"
            )
            raise SystemExit(0)
        if args.train_run_chunk_phase1:
            ret = fpga_train_run_chunk_phase1_trace_stdp(
                ser,
                nsteps=max(1, int(args.chunk_nsteps)),
                tile_rows=max(1, int(args.train_tile_rows)),
            )
            print(
                "TRAIN_RUN_CHUNK phase1 completed: "
                f"result=0x{(int(ret) & 0xFFFFFFFF):08X}, nsteps={max(1,int(args.chunk_nsteps))}, "
                f"tile_rows={max(1,int(args.train_tile_rows))}"
            )
            raise SystemExit(0)
        if args.train_run_chunk_phase2:
            if args.image_source != "fpga":
                raise ValueError(
                    "--train-run-chunk-phase2 currently requires --image-source fpga "
                    "(it expects raw_image0_u8 to be loaded on FPGA from SD/DDR)"
                )
            prepare_fpga_sample_image_via_streamed_load(
                ser,
                sample_idx=int(args.sample_idx),
                start_lba=int(args.start_lba),
                timeout_sec=float(args.timeout),
            )
            ret = fpga_train_run_chunk_phase2_infer_trace_stdp(
                ser,
                nsteps=max(1, int(args.chunk_nsteps)),
                tile_rows=max(1, int(args.train_tile_rows)),
            )
            print(
                "TRAIN_RUN_CHUNK phase2 completed: "
                f"result=0x{(int(ret) & 0xFFFFFFFF):08X}, nsteps={max(1,int(args.chunk_nsteps))}, "
                f"tile_rows={max(1,int(args.train_tile_rows))}, infer_total_spikes={int(ret) & 0xFFFFFFFF}"
            )
            raise SystemExit(0)
        if args.train_run_chunk_phase2_verify:
            if args.image_source != "fpga":
                raise ValueError("--train-run-chunk-phase2-verify currently requires --image-source fpga")
            fpga_train_run_chunk_phase2_verify_infer_stats(
                ser,
                sample_idx=int(args.sample_idx),
                nsteps=max(1, int(args.chunk_nsteps)),
                tile_rows=max(1, int(args.train_tile_rows)),
                start_lba=int(args.start_lba),
                timeout_sec=float(args.timeout),
            )
            raise SystemExit(0)
        if args.train_run_sample_phase3:
            if args.image_source != "fpga":
                raise ValueError("--train-run-sample-phase3 currently requires --image-source fpga")
            prepare_fpga_sample_image_via_streamed_load(
                ser,
                sample_idx=int(args.sample_idx),
                start_lba=int(args.start_lba),
                timeout_sec=float(args.timeout),
            )
            ret = fpga_train_run_sample_phase3(
                ser,
                inj_steps=max(1, int(args.chunk_nsteps)),
                tile_rows=max(1, int(args.train_tile_rows)),
            )
            inj_spikes = int(ret) & 0xFFFF
            blank_spikes = (int(ret) >> 16) & 0xFFFF
            print(
                "TRAIN_RUN_SAMPLE_PHASE3 completed: "
                f"result=0x{(int(ret) & 0xFFFFFFFF):08X}, inj_steps={max(1,int(args.chunk_nsteps))}, "
                f"tile_rows={max(1,int(args.train_tile_rows))}, inj_total_spikes={inj_spikes}, "
                f"blank_total_spikes={blank_spikes}"
            )
            raise SystemExit(0)
        if args.train_run_sample_phase4:
            if args.image_source != "fpga":
                raise ValueError("--train-run-sample-phase4 currently requires --image-source fpga")
            prepare_fpga_sample_image_via_streamed_load(
                ser,
                sample_idx=int(args.sample_idx),
                start_lba=int(args.start_lba),
                timeout_sec=float(args.timeout),
            )
            ret = fpga_train_run_sample_phase4(
                ser,
                inj_steps=max(1, int(args.chunk_nsteps)),
                tile_rows=max(1, int(args.train_tile_rows)),
            )
            inj_spikes = int(ret) & 0xFFFF
            blank_spikes = (int(ret) >> 16) & 0xFFFF
            tdbg = {}
            try:
                tdbg = fpga_read_train_debug(ser)
            except Exception:
                tdbg = {}
            accepted_max_fr = int(tdbg.get("chunk_retry_accepted_max_fr", 0))
            print(
                "TRAIN_RUN_SAMPLE_PHASE4 completed: "
                f"result=0x{(int(ret) & 0xFFFFFFFF):08X}, inj_steps={max(1,int(args.chunk_nsteps))}, "
                f"tile_rows={max(1,int(args.train_tile_rows))}, inj_total_spikes={inj_spikes}, "
                f"blank_total_spikes={blank_spikes}, accepted_max_fr={accepted_max_fr}"
            )
            raise SystemExit(0)
        if args.train_run_sample_phase3_retry:
            if args.image_source != "fpga":
                raise ValueError("--train-run-sample-phase3-retry currently requires --image-source fpga")
            accepted_max_fr, probe_inj, phase3_inj, phase3_blank = fpga_train_run_sample_phase3_retry_coarse(
                ser,
                sample_idx=int(args.sample_idx),
                inj_steps=max(1, int(args.chunk_nsteps)),
                tile_rows=max(1, int(args.train_tile_rows)),
                start_lba=int(args.start_lba),
                timeout_sec=float(args.timeout),
                image_source=str(args.image_source),
                seed=int(args.seed),
                max_fr_start=max(1, int(args.train_retry_max_fr_start)),
                max_fr_step=max(1, int(args.train_retry_max_fr_step)),
                max_fr_limit=max(1, int(args.train_retry_max_fr_limit)),
                min_inj_spikes=max(0, int(args.train_retry_min_inj_spikes)),
            )
            print(
                "TRAIN_RUN_SAMPLE_PHASE3 retry coarse summary: "
                f"accepted_max_fr={accepted_max_fr}, probe_inj_total={probe_inj}, "
                f"phase3_inj_total={phase3_inj}, phase3_blank_total={phase3_blank}"
            )
            raise SystemExit(0)
        if args.train_run_sample_phase3_verify:
            fpga_train_run_sample_phase3_verify_stats(
                ser,
                sample_idx=int(args.sample_idx),
                inj_steps=max(1, int(args.chunk_nsteps)),
                tile_rows=max(1, int(args.train_tile_rows)),
                start_lba=int(args.start_lba),
                timeout_sec=float(args.timeout),
                image_source=str(args.image_source),
                raw_bin=args.raw_bin,
            )
            raise SystemExit(0)
        if args.train_mine_one_sample_replay_selfcheck:
            fpga_train_mine_one_sample_replay_selfcheck(
                ser,
                args,
                seed=int(args.train_mine_seed),
                tile_rows=max(1, int(args.train_tile_rows)),
                verify_row0=max(0, int(args.train_verify_row0)),
                verify_nrows=(None if int(args.train_verify_nrows) <= 0 else int(args.train_verify_nrows)),
                verify_mode=str(args.train_verify_mode),
                verify_sample_cols=max(1, int(args.train_verify_sample_cols)),
            )
            raise SystemExit(0)

        if args.ddr_smoke:
            fpga_ddr_smoke_test(ser, base_addr_word=args.ddr_smoke_addr)

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

        if args.image_source == "fpga" and args.num_sectors == 0 and not args.full_sd_copy:
            img_idx = int(args.sample_idx)
            if img_idx < 0:
                raise ValueError(f"--sample-idx must be >=0 for --image-source fpga, got {img_idx}")
            img_byte_off = RAW1_HEADER_BYTES + RAW1_NUM_IMAGES_DEFAULT + (img_idx * N_IN)
            img_sector_off = img_byte_off // 512
            img_byte_in_sector = img_byte_off % 512
            sectors_needed = (img_byte_in_sector + N_IN + 511) // 512
            ddr_image_base_byte = img_byte_in_sector
            print(
                "Using streamed FPGA image load path: "
                f"sample_idx={img_idx}, sector_off={img_sector_off}, sectors={sectors_needed}, "
                f"byte_in_sector={img_byte_in_sector}, ddr_image_base_byte={ddr_image_base_byte}"
            )
            fpga_sd_sectors_to_ddr(
                ser=ser,
                start_lba=args.start_lba + img_sector_off,
                num_sectors=sectors_needed,
                timeout_sec=args.timeout,
            )
            fpga_load_image_from_ddr(
                ser=ser,
                base_addr_byte=ddr_image_base_byte,
                n_bytes=N_IN,
                timeout_sec=args.timeout,
            )
        else:
            if args.num_sectors > 0:
                fpga_sd_sectors_to_ddr(
                    ser=ser,
                    start_lba=args.start_lba,
                    num_sectors=args.num_sectors,
                    timeout_sec=args.timeout,
                )
            elif args.full_sd_copy:
                print(
                    "Using full sector DMA copy (RAW1 parser on FPGA disabled): "
                    f"sectors={RAW1_TOTAL_SECTORS_DEFAULT}"
                )
                fpga_sd_sectors_to_ddr(
                    ser=ser,
                    start_lba=args.start_lba,
                    num_sectors=RAW1_TOTAL_SECTORS_DEFAULT,
                    timeout_sec=args.timeout,
                )
            else:
                print(
                    "Using host-side RAW1 size assumption for full sector DMA copy "
                    f"(sectors={RAW1_TOTAL_SECTORS_DEFAULT})"
                )
                fpga_sd_sectors_to_ddr(
                    ser=ser,
                    start_lba=args.start_lba,
                    num_sectors=RAW1_TOTAL_SECTORS_DEFAULT,
                    timeout_sec=args.timeout,
                )

        if args.image_source == "fpga":
            image0_u8 = fpga_read_raw_image_u8(ser)
            label0 = -1
            if args.raw_bin:
                raw_bin_path = resolve_raw_bin_path(args.raw_bin)
                ref_u8, ref_label = read_raw1_image_u8(str(raw_bin_path), sample_idx=args.sample_idx)
                img_mismatch = sum(1 for a, b in zip(image0_u8, ref_u8) if int(a) != int(b))
                label0 = ref_label
                print(
                    "Loaded comparison image from FPGA raw_image0_u8 "
                    f"(sample_idx={args.sample_idx}, label={label0}, mismatched_vs_raw={img_mismatch})"
                )
            else:
                print(
                    "Loaded comparison image from FPGA raw_image0_u8 "
                    f"(sample_idx={args.sample_idx}, label=unknown)"
                )
        elif args.image_source == "mnist":
            image0_u8, label0 = read_mnist_image_u8(args.sample_idx)
            print(f"Loaded comparison image from MNIST: sample_idx={args.sample_idx} (label={label0})")
        else:
            raw_bin_path = resolve_raw_bin_path(args.raw_bin)
            image0_u8, label0 = read_raw1_image_u8(str(raw_bin_path), sample_idx=args.sample_idx)
            print(f"Loaded comparison image from RAW1: {raw_bin_path} (sample_idx={args.sample_idx}, label={label0})")

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
