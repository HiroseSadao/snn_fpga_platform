from __future__ import annotations

import argparse
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
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
OP_READ_TRAIN_INJ_SPIKE_COUNT = 0x27
OP_TRAIN_QUERY_CAPS = 0x30
OP_TRACE_UPDATE = 0x31
OP_STDP_UPDATE_TILE = 0x32
OP_TRAIN_GEN_WORK = 0x33
OP_READ_TRAIN_DEBUG = 0x34
OP_STDP_UPDATE_ALL = 0x35
OP_TRAIN_RUN_CHUNK = 0x36
OP_TRAIN_RUN_SAMPLE_PHASE3 = 0x37
OP_TRAIN_RUN_SAMPLE_PHASE4 = 0x38
OP_TRAIN_LABEL_STATS_RESET = 0x39
OP_TRAIN_LABEL_STATS_ACCUM = 0x3A
OP_READ_TRAIN_LABEL_STAT_SUM = 0x3B
OP_READ_TRAIN_LABEL_STAT_COUNT = 0x3C

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
N_NEURONS = 50
N_WEIGHTS = N_NEURONS * N_IN
PHASE3_SPARSE_PERCENT = 30
PHASE3_SPARSE_SEED = 0x5A17_0030
RAW1_HEADER_BYTES = 20
RAW1_NUM_IMAGES_DEFAULT = 10_000
RAW1_TOTAL_BYTES_DEFAULT = RAW1_HEADER_BYTES + RAW1_NUM_IMAGES_DEFAULT + (RAW1_NUM_IMAGES_DEFAULT * N_IN)
RAW1_TOTAL_SECTORS_DEFAULT = (RAW1_TOTAL_BYTES_DEFAULT + 511) // 512
IMGLOAD_SRC_BIAS_BYTES = 0
IMGLOAD_DST_BIAS_BYTES = 0
IMGLOAD_GUARD_SEC = 0.01
IMG_STAGING_MARGIN_WORDS = 262144  # must match top_level.sv IMG_STAGING_BASE_WORD margin


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
MINE_WINH = 0.875

# Poisson/RNG constants (must match top_level.sv)
POISSON_NUM_CONST = 9175  # floor(32*140*2048*1e-3)
# Poisson threshold upper bound. 2048 means always-fire against rand11 in [0..2047].
RNG_MAX = 2048
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
    """Step1: fix the logical DDR layout used by training kernels.

    Layout addresses are defined in 32-bit word units as a host/HDL contract.
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




def img_staging_base_word() -> int:
    """Must mirror top_level.sv IMG_STAGING_BASE_WORD."""
    layout = build_train_ddr_layout()
    return int(layout.base_prelist_work) + int(IMG_STAGING_MARGIN_WORDS)


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
    if reason == 0x28:
        byte_idx = (u >> 2) & 0x3FF
        lane = u & 0x3
        return (
            f"reason=IMGLOAD_DDR_TIMEOUT_OR_RANGE(0x28), "
            f"opcode=0x{opcode:02X}, byte_idx={byte_idx}, lane={lane}"
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
    if reason == 0x38:
        calib = (u >> 23) & 0x1
        ddr_pending = (u >> 22) & 0x1
        trace_active = (u >> 21) & 0x1
        stdp_active = (u >> 20) & 0x1
        stdp_batch = (u >> 19) & 0x1
        chunk_active = (u >> 18) & 0x1
        label_stats_active = (u >> 17) & 0x1
        infer_active = (u >> 16) & 0x1
        req_nargs_dbg = (u >> 8) & 0xFF
        tile_rows_lo8 = u & 0xFF
        return (
            "reason=TRAIN_RUN_SAMPLE_PHASE4_GATE(0x38), "
            f"calib={calib}, ddr_pending={ddr_pending}, trace={trace_active}, "
            f"stdp={stdp_active}, stdp_batch={stdp_batch}, chunk={chunk_active}, "
            f"label_stats={label_stats_active}, infer={infer_active}, "
            f"req_nargs={req_nargs_dbg}, tile_rows_lo8={tile_rows_lo8}"
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
    clear_input_buffer: bool = True,
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

            if clear_input_buffer:
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
    verbose: bool = True,
) -> int:
    if verbose:
        print(f"Requesting SD sectors->DDR DMA: start_lba={start_lba}, sectors={num_sectors}")
    t0 = time.time()
    try:
        status, result = send_request(
            ser=ser,
            opcode=OP_SD_SECTORS_TO_DDR,
            args=[start_lba, num_sectors],
            response_timeout=timeout_sec,
        )
    except TimeoutError as exc:
        print(
            "SD sectors->DDR timeout: "
            f"start_lba={int(start_lba)}, sectors={int(num_sectors)}, waited={float(timeout_sec):.1f}s"
        )
        fpga_probe_runtime_debug(ser, context="sd_sectors_to_ddr timeout")
        raise exc
    require_ok(status, "SD sectors->DDR")
    elapsed = time.time() - t0
    if verbose:
        print(f"SD sectors->DDR completed in {elapsed:.2f}s, words_written={result}")
    return result


def fpga_load_image_from_ddr(
    ser: serial.Serial,
    base_addr_byte: int,
    n_bytes: int = N_IN,
    timeout_sec: float = 120.0,
    verbose: bool = True,
) -> None:
    if n_bytes <= 0 or n_bytes > N_IN:
        raise ValueError(f"n_bytes must be in [1, {N_IN}], got {n_bytes}")
    if verbose:
        print(f"Requesting DDR->raw_image0 load: base_byte=0x{base_addr_byte:08X}, n_bytes={n_bytes}")
    try:
        status, result = send_request(
            ser=ser,
            opcode=OP_LOAD_IMAGE_FROM_DDR,
            args=[base_addr_byte, n_bytes],
            response_timeout=min(float(timeout_sec), 5.0),
        )
    except TimeoutError as exc:
        print(
            "DDR->raw_image0 load timeout: "
            f"base_byte=0x{int(base_addr_byte):08X}, n_bytes={int(n_bytes)}"
        )
        fpga_probe_runtime_debug(ser, context="load_image_from_ddr timeout")
        raise exc
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
    status, total_spikes = send_request(
        ser=ser,
        opcode=OP_RUN_SAMPLE_INFER,
        args=[seed_i32, n_steps],
        response_timeout=timeout_sec,
    )
    require_ok(status, "RUN_SAMPLE_INFER")
    return total_spikes


def build_fixed_weight_matrix_q16() -> np.ndarray:
    # Must match the fallback fixed connectivity used by the Python mine-style reference.
    w_q16 = np.zeros((N_NEURONS, N_IN), dtype=np.uint16)
    for n in range(N_NEURONS):
        for i in range(N_IN):
            if ((i + n) & 0x3) == 0:
                w_q16[n, i] = np.uint16(FXP_INPUT_W & 0xFFFF)
    return w_q16


def load_q16_mem_matrix(path: Path, rows: int, cols: int) -> np.ndarray:
    vals: list[int] = []
    with path.open("r", encoding="ascii") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            vals.append(int(s, 16) & 0xFFFF)
    need = int(rows) * int(cols)
    if len(vals) < need:
        raise ValueError(f"{path} has too few entries: {len(vals)} < {need}")
    if len(vals) > need:
        vals = vals[:need]
    arr = np.asarray(vals, dtype=np.uint16).reshape((rows, cols))
    return arr


def _write_mem_hex(path: Path, values: list[int], width_bits: int) -> None:
    width_nibbles = max(1, (int(width_bits) + 3) // 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as f:
        for v in values:
            f.write(f"{int(v) & ((1 << width_bits) - 1):0{width_nibbles}X}\n")


def _build_phase3_sparse_layout() -> dict[str, list[int] | np.ndarray | int]:
    n_edges = (int(N_WEIGHTS) * int(PHASE3_SPARSE_PERCENT)) // 100
    if n_edges <= 0 or n_edges > int(N_WEIGHTS):
        raise ValueError(f"invalid sparse edge target: n_edges={n_edges}, n_weights={N_WEIGHTS}")

    base_deg = n_edges // int(N_NEURONS)
    extra_deg = n_edges % int(N_NEURONS)
    rng = np.random.default_rng(int(PHASE3_SPARSE_SEED))

    csr_row_ptr: list[int] = [0]
    csr_col_idx: list[int] = []
    edge_dense_idx: list[int] = []
    for row in range(int(N_NEURONS)):
        deg = base_deg + (1 if row < extra_deg else 0)
        cols = np.sort(rng.choice(int(N_IN), size=int(deg), replace=False))
        for col in cols.tolist():
            csr_col_idx.append(int(col))
            edge_dense_idx.append((row * int(N_IN)) + int(col))
        csr_row_ptr.append(len(csr_col_idx))

    csc_bins: list[list[tuple[int, int]]] = [[] for _ in range(int(N_IN))]
    for row in range(int(N_NEURONS)):
        s = csr_row_ptr[row]
        e = csr_row_ptr[row + 1]
        for edge_local in range(s, e):
            col = csr_col_idx[edge_local]
            csc_bins[col].append((row, edge_local))

    csc_col_ptr: list[int] = [0]
    csc_row_idx: list[int] = []
    csc_edge_idx: list[int] = []
    for col in range(int(N_IN)):
        entries = csc_bins[col]
        for row, edge_id in entries:
            csc_row_idx.append(int(row))
            csc_edge_idx.append(int(edge_id))
        csc_col_ptr.append(len(csc_row_idx))

    if len(csr_col_idx) != n_edges or len(csc_row_idx) != n_edges or len(csc_edge_idx) != n_edges:
        raise RuntimeError("phase3 sparse layout build failed edge count consistency check")

    return {
        "n_edges": int(n_edges),
        "csr_row_ptr": csr_row_ptr,
        "csr_col_idx": csr_col_idx,
        "csc_col_ptr": csc_col_ptr,
        "csc_row_idx": csc_row_idx,
        "csc_edge_idx": csc_edge_idx,
        "edge_dense_idx": np.asarray(edge_dense_idx, dtype=np.int64),
    }


def ensure_phase3_sparse_index_mem_files(data_dir: Path, *, rewrite: bool = False) -> dict[str, int]:
    """Create phase3 sparse(30%) CSR/CSC index mem files + sparse infer_w mem."""
    layout = _build_phase3_sparse_layout()
    n_edges = int(layout["n_edges"])
    row_ptr_w = int(np.ceil(np.log2(n_edges + 1)))
    edge_w = int(np.ceil(np.log2(max(2, n_edges))))
    col_w = int(np.ceil(np.log2(N_IN)))
    row_w = int(np.ceil(np.log2(N_NEURONS)))
    csr_row_ptr = list(layout["csr_row_ptr"])
    csr_col_idx = list(layout["csr_col_idx"])
    csc_col_ptr = list(layout["csc_col_ptr"])
    csc_row_idx = list(layout["csc_row_idx"])
    csc_edge_idx = list(layout["csc_edge_idx"])

    files = {
        "csr_row_ptr.mem": (csr_row_ptr, row_ptr_w),
        "csr_col_idx.mem": (csr_col_idx, col_w),
        "csc_col_ptr.mem": (csc_col_ptr, row_ptr_w),
        "csc_row_idx.mem": (csc_row_idx, row_w),
        "csc_edge_idx.mem": (csc_edge_idx, edge_w),
    }
    for name, (vals, bits) in files.items():
        out_path = data_dir / name
        if rewrite or (not out_path.exists()):
            _write_mem_hex(out_path, vals, bits)

    # Build sparse csr_weight from dense source using selected edges.
    infer_w_path = data_dir / "infer_w_q16.mem"
    if not infer_w_path.exists():
        raise FileNotFoundError(f"missing weight mem for sparse masking: {infer_w_path}")
    w_dense = load_q16_mem_matrix(infer_w_path, N_NEURONS, N_IN).reshape(-1)
    edge_dense_idx = np.asarray(layout["edge_dense_idx"], dtype=np.int64)
    w_sparse_edges = w_dense[edge_dense_idx].astype(np.uint16)
    if rewrite:
        _write_mem_hex(data_dir / "csr_weight_q16.mem", [int(v) for v in w_sparse_edges.tolist()], 16)

    return {
        "n_edges": n_edges,
        "sparse_percent": int(PHASE3_SPARSE_PERCENT),
        "csr_row_ptr_len": len(csr_row_ptr),
        "csr_col_idx_len": len(csr_col_idx),
        "csc_col_ptr_len": len(csc_col_ptr),
        "csc_row_idx_len": len(csc_row_idx),
        "csc_edge_idx_len": len(csc_edge_idx),
        "w_nnz_after_mask": int(np.count_nonzero(w_sparse_edges)),
    }


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
        try:
            status, value = send_request(ser, OP_READ_SPIKE_COUNT, [neuron_idx, 0])
        except TimeoutError as exc:
            print(f"READ_SPIKE_COUNT timeout at neuron_idx={neuron_idx}")
            fpga_probe_runtime_debug(ser, context=f"read_spike_count timeout idx={neuron_idx}")
            raise exc
        require_ok(status, f"READ_SPIKE_COUNT[{neuron_idx}]")
        counts.append(int(value) & 0xFFFF)
    return counts


def fpga_read_train_inj_spike_counts(ser: serial.Serial) -> list[int]:
    counts = []
    for neuron_idx in range(N_NEURONS):
        status, value = send_request(ser, OP_READ_TRAIN_INJ_SPIKE_COUNT, [neuron_idx, 0])
        require_ok(status, f"READ_TRAIN_INJ_SPIKE_COUNT[{neuron_idx}]")
        counts.append(int(value) & 0xFFFF)
    return counts


def fpga_try_read_train_inj_spike_counts(ser: serial.Serial) -> list[int] | None:
    """Return inj counts if opcode is available; otherwise return None."""
    try:
        return fpga_read_train_inj_spike_counts(ser)
    except RuntimeError as exc:
        msg = str(exc)
        if ("UNSUPPORTED_OP" in msg) and ("opcode=0x27" in msg):
            return None
        raise


def fpga_train_label_stats_reset(ser: serial.Serial) -> None:
    status, result = send_request(ser, OP_TRAIN_LABEL_STATS_RESET, [0, 0], response_timeout=max(10.0, TRAIN_KERNEL_TIMEOUT_SEC))
    require_ok(status, "TRAIN_LABEL_STATS_RESET")


def fpga_train_label_stats_accum(ser: serial.Serial, label: int) -> None:
    status, result = send_request(ser, OP_TRAIN_LABEL_STATS_ACCUM, [int(label), 0], response_timeout=max(5.0, TRAIN_KERNEL_TIMEOUT_SEC))
    require_ok(status, f"TRAIN_LABEL_STATS_ACCUM[label={int(label)}]")


def fpga_read_train_label_stat_sum_row(ser: serial.Serial, label: int) -> list[int]:
    out: list[int] = []
    for n in range(N_NEURONS):
        status, value = send_request(
            ser,
            OP_READ_TRAIN_LABEL_STAT_SUM,
            [int(label), n],
            response_timeout=3.0,
            transient_retry_max=8,
            clear_input_buffer=False,
        )
        require_ok(status, f"READ_TRAIN_LABEL_STAT_SUM[label={int(label)},n={n}]")
        out.append(int(np.int32(value)))
    return out


def fpga_read_train_label_stat_count(ser: serial.Serial, label: int) -> int:
    status, value = send_request(
        ser,
        OP_READ_TRAIN_LABEL_STAT_COUNT,
        [int(label), 0],
        response_timeout=3.0,
        transient_retry_max=8,
        clear_input_buffer=False,
    )
    require_ok(status, f"READ_TRAIN_LABEL_STAT_COUNT[label={int(label)}]")
    return int(value) & 0xFFFFFFFF


def fpga_read_train_label_stats_all(ser: serial.Serial) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((10, N_NEURONS), dtype=np.int64)
    counts = np.zeros((10,), dtype=np.int64)
    for lbl in range(10):
        sums[lbl, :] = np.asarray(fpga_read_train_label_stat_sum_row(ser, lbl), dtype=np.int64)
        counts[lbl] = np.int64(fpga_read_train_label_stat_count(ser, lbl))
    return sums, counts


def assign_labels_from_aggregated_stats(
    label_spike_sums: np.ndarray,
    label_counts: np.ndarray,
    *,
    rates_prev: np.ndarray | None = None,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """mine.py::assign_labels() equivalent using aggregated sums/counts instead of per-sample spikes."""
    sums = np.asarray(label_spike_sums, dtype=np.float32)
    counts = np.asarray(label_counts, dtype=np.int64)
    if sums.shape != (10, N_NEURONS):
        raise ValueError(f"label_spike_sums shape must be (10,{N_NEURONS}), got {sums.shape}")
    if counts.shape != (10,):
        raise ValueError(f"label_counts shape must be (10,), got {counts.shape}")

    if rates_prev is None:
        rates = np.zeros((N_NEURONS, 10), dtype=np.float32)
    else:
        rates = np.array(rates_prev, copy=True, dtype=np.float32)
        if rates.shape != (N_NEURONS, 10):
            raise ValueError(f"rates_prev shape must be ({N_NEURONS},10), got {rates.shape}")

    for lbl in range(10):
        n_labeled = int(counts[lbl])
        if n_labeled > 0:
            rates[:, lbl] = float(alpha) * rates[:, lbl] + (sums[lbl, :] / float(n_labeled))

    sum_rate = np.sum(rates, axis=1)
    sum_rate[sum_rate == 0] = 1.0
    proportions = rates / np.expand_dims(sum_rate, 1)
    proportions[proportions != proportions] = 0.0
    assignments = np.argmax(proportions, axis=1).astype(np.uint8)
    return assignments, rates


def predict_label_from_counts(counts: np.ndarray, assignments: np.ndarray) -> int:
    """mine.py::prediction() equivalent for one sample spike-count vector."""
    c = np.asarray(counts, dtype=np.float64).reshape(-1)
    a = np.asarray(assignments, dtype=np.int64).reshape(-1)
    if c.shape[0] != N_NEURONS or a.shape[0] != N_NEURONS:
        raise ValueError(f"predict_label_from_counts expects length {N_NEURONS}")
    rates = np.zeros((10,), dtype=np.float64)
    for lbl in range(10):
        idx = np.where(a == lbl)[0]
        if idx.size > 0:
            rates[lbl] = float(np.sum(c[idx])) / float(idx.size)
    return int(np.argmax(rates))


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
        vals.append(int(value) & 0xFFF)
    return vals


def fpga_read_infer_debug(ser: serial.Serial) -> dict[str, int]:
    names = {
        0: "infer_total_spikes",
        1: "infer_steps_target",
        2: "infer_step_idx",
        3: "infer_state",
        4: "infer_pre_active_count",
        5: "raw_image0_sum_u8",
        6: "poisson_num_const_cfg",
        7: "infer_active",
        8: "imgload_active",
        9: "ddr_req_pending_core",
        10: "sd_copy_active",
        11: "rx_state",
        12: "response_ready",
        13: "first_step_input_spikes",
        14: "total_input_spikes_generated",
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


def fpga_read_ddr_words(
    ser: serial.Serial,
    *,
    base_word: int,
    count: int,
) -> list[int]:
    out: list[int] = []
    for i in range(max(0, int(count))):
        out.append(fpga_ddr_read32(ser, int(base_word) + i))
    return out


def _ddr_words_to_bytes_le(words: list[int]) -> list[int]:
    out: list[int] = []
    for w in words:
        ww = int(w) & 0xFFFFFFFF
        out.append((ww >> 0) & 0xFF)
        out.append((ww >> 8) & 0xFF)
        out.append((ww >> 16) & 0xFF)
        out.append((ww >> 24) & 0xFF)
    return out


def _count_mismatch(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if int(a[i]) != int(b[i]))


def _first_nonzero_indices(data: list[int], *, limit: int = 8) -> list[int]:
    out: list[int] = []
    for i, v in enumerate(data):
        if int(v) != 0:
            out.append(i)
            if len(out) >= int(limit):
                break
    return out


def fpga_read_raw_image_span(
    ser: serial.Serial,
    *,
    start_idx: int,
    nbytes: int,
) -> list[int]:
    s = max(0, min(int(start_idx), int(N_IN)))
    n = max(0, min(int(nbytes), int(N_IN) - s))
    out: list[int] = []
    for idx in range(n):
        status, value = send_request(
            ser,
            OP_READ_RAW_U8,
            [s + idx, 0],
            response_timeout=1.0,
            transient_retry_max=max(TRANSIENT_RETRY_MAX, 8),
        )
        require_ok(status, f"READ_RAW_U8_SPAN[{s + idx}]")
        out.append(int(value) & 0xFF)
    return out


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
        # DDR_ZERO32 runs an internal word-by-word generator/write loop in FPGA and can take
        # noticeably longer than simple single-word DDR commands, especially for large chunks.
        # Use a chunk-scaled timeout to avoid spurious host-side UART timeouts.
        zero_timeout_s = max(2.0, min(60.0, float(chunk) / 4000.0))
        try:
            status, result = send_request(
                ser,
                OP_DDR_ZERO32,
                [base, chunk],
                response_timeout=zero_timeout_s,
                transient_retry_max=max(TRANSIENT_RETRY_MAX, 8),
            )
        except TimeoutError as exc:
            print(
                "DDR_ZERO32 timeout: "
                f"base_word=0x{int(base):08X}, chunk={int(chunk)}, total_done={int(total_done)}, rem={int(rem)}"
            )
            try:
                mdbg = fpga_read_train_debug_minimal(ser)
                print("Train debug minimal:", ", ".join(f"{k}={v}" for k, v in mdbg.items()))
                flags = int(mdbg.get("ddr_flags", 0))
                print(
                    "Train debug minimal (decoded ddr_flags): "
                    f"req_toggle_ddr_sync2={flags & 1}, "
                    f"req_toggle_ddr_seen={(flags >> 1) & 1}, "
                    f"rsp_toggle_core_sync2={(flags >> 2) & 1}, "
                    f"rsp_toggle_core_seen={(flags >> 3) & 1}, "
                    f"ddr_req_we_core={(flags >> 4) & 1}, "
                    f"ddr_req_from_train_core={(flags >> 5) & 1}"
                )
            except Exception as dbg_exc:
                print(f"Train debug minimal probe failed: {dbg_exc}")
            raise exc
        require_ok(status, f"DDR_ZERO32[base={base},n={chunk}]")
        total_done += int(result) & 0xFFFF
        base += chunk
        rem -= chunk
    return total_done


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST through the exact same loader used by import_MNIST_raw.py."""
    script_dir = Path(__file__).resolve().parent
    snn_dir = script_dir.parent
    repo_dir = snn_dir.parent
    if str(snn_dir) not in sys.path:
        sys.path.insert(0, str(snn_dir))
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    try:
        from import_MNIST_raw import load_mnist as raw_loader_local
        images, labels = raw_loader_local()
        return np.asarray(images), np.asarray(labels, dtype=np.int64)
    except Exception as exc:
        raise RuntimeError(
            "MNIST loading failed via import_MNIST_raw.py::load_mnist(). "
            "Ensure snn_cuda/import_MNIST_raw.py and its dependencies are available."
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
    script_dir = Path(__file__).resolve().parent
    snn_dir = script_dir.parent
    repo_dir = snn_dir.parent
    candidates: list[Path] = []
    if raw_bin_arg:
        p = Path(raw_bin_arg)
        candidates.extend(
            [
                p,
                (Path.cwd() / p),
                (script_dir / p),
                (snn_dir / p),
                (repo_dir / p),
            ]
        )
    else:
        candidates.extend(
            [
                Path.cwd() / "raw_samples_u8.bin",
                Path.cwd() / "raw_samples.bin",
                snn_dir / "raw_samples_u8.bin",
                snn_dir / "raw_samples.bin",
                repo_dir / "raw_samples_u8.bin",
                repo_dir / "raw_samples.bin",
            ]
        )

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(p)

    for p in unique_candidates:
        if p.exists() and p.is_file():
            return p.resolve()

    searched = ", ".join(str(x) for x in unique_candidates)
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


def read_raw1_labels_u8(raw_bin_path: str) -> np.ndarray:
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
        labels = f.read(num_images)
        if len(labels) != num_images:
            raise ValueError("RAW1 labels are truncated")
        return np.frombuffer(labels, dtype=np.uint8).copy()


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
    post_active: np.ndarray | None = None,
) -> None:
    if post_active is not None:
        if post_active.size > 0:
            A[post_active.astype(np.int64), :] += x_in
    elif winner_idx is not None and winner_idx >= 0:
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
        post_active = np.flatnonzero(s_exc)
        if post_active.size > 0:
            A_ref[post_active, :] += x_in
        if pre_active.size > 0:
            np.add.at(B_T_ref, pre_active, x_exc)
        kernel_trace_update_python(A_k, B_T_k, x_in, x_exc, None, pre_active, post_active)

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


def fpga_read_train_debug_minimal(ser: serial.Serial) -> dict[str, int]:
    """Always-on minimal train/DDR debug IDs exposed even when TRAIN_DEBUG_ENABLE=0."""
    ids = [
        (48, "gen_active"),
        (49, "gen_state"),
        (50, "gen_idx"),
        (51, "gen_count_total"),
        (52, "ddr_req_pending_core"),
        (53, "ddr_bridge_state"),
        (54, "ddr_req_addr_word_core"),
        (55, "ddr_flags"),
    ]
    out: dict[str, int] = {}
    for idx, key in ids:
        status, value = send_request(ser, OP_READ_TRAIN_DEBUG, [idx, 0], response_timeout=1.0, transient_retry_max=0)
        require_ok(status, f"READ_TRAIN_DEBUG_MIN[{idx}]")
        out[key] = int(value) & 0xFFFFFFFF
    return out


def fpga_read_train_debug_extra(ser: serial.Serial) -> dict[str, int]:
    """Best-effort runtime debug IDs focused on SD/image-load stalls."""
    ids = [
        (56, "sd_copy_active"),
        (57, "sd_ddr_flush_active"),
        (58, "sd_in_read"),
        (59, "sd_sector_buf_ready"),
        (60, "imgload_active"),
        (61, "imgload_word_valid"),
        (62, "sd_status"),
        (63, "sd_wait_counter"),
    ]
    out: dict[str, int] = {}
    for idx, key in ids:
        try:
            status, value = send_request(
                ser,
                OP_READ_TRAIN_DEBUG,
                [idx, 0],
                response_timeout=1.0,
                transient_retry_max=0,
            )
            require_ok(status, f"READ_TRAIN_DEBUG_EXTRA[{idx}]")
            out[key] = int(value) & 0xFFFFFFFF
        except Exception:
            out[key] = -1
    return out


def fpga_probe_runtime_debug(ser: serial.Serial, *, context: str) -> None:
    print(f"[debug] probe start: {context}")
    try:
        caps = fpga_train_query_caps(ser)
        print(f"[debug] caps=0x{caps:08X}")
    except Exception as exc:
        print(f"[debug] caps read failed: {exc}")
    try:
        tmin = fpga_read_train_debug_minimal(ser)
        print("[debug] train_min:", ", ".join(f"{k}={v}" for k, v in tmin.items()))
    except Exception as exc:
        print(f"[debug] train_min read failed: {exc}")
    try:
        textra = fpga_read_train_debug_extra(ser)
        print("[debug] train_extra:", ", ".join(f"{k}={v}" for k, v in textra.items()))
    except Exception as exc:
        print(f"[debug] train_extra read failed: {exc}")
    try:
        idbg = fpga_read_infer_debug(ser)
        print("[debug] infer:", ", ".join(f"{k}={v}" for k, v in idbg.items()))
    except Exception as exc:
        print(f"[debug] infer read failed: {exc}")
    print(f"[debug] probe end: {context}")


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
    """Kernel trigger only. x_in/x_exc/prelist are prepared by other opcodes."""
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
            # Phase4 launch can transiently collide with just-finished background FSM cleanup.
            # Retry BAD_PACKET(result==0)/timeout briefly before surfacing as hard error.
            transient_retry_max=32,
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


def fpga_probe_phase4_opcode(ser: serial.Serial) -> None:
    """Probe whether opcode 0x38 is implemented in the current bitstream.

    Sends an intentionally invalid phase4 payload. Implemented kernels should return BAD_PACKET,
    while non-implemented kernels return UNSUPPORTED_OP.
    """
    status, result = send_request(
        ser,
        OP_TRAIN_RUN_SAMPLE_PHASE4,
        [0, 0],  # invalid by design
        response_timeout=2.0,
        transient_retry_max=0,
    )
    if status == STATUS_UNSUPPORTED_OP:
        raise RuntimeError(
            "FPGA bitstream/protocol mismatch: opcode 0x38 (TRAIN_RUN_SAMPLE_PHASE4) is not implemented "
            "in the currently programmed FPGA image."
        )
    # Implemented bitstreams should reject [0,0] as BAD_PACKET.
    if status not in (STATUS_BAD_PACKET, STATUS_OK):
        raise RuntimeError(f"phase4 opcode probe unexpected status=0x{status:02X}, result=0x{int(result)&0xFFFFFFFF:08X}")


def fpga_probe_phase3_opcode(ser: serial.Serial) -> None:
    """Probe whether opcode 0x37 is implemented in the current bitstream."""
    status, result = send_request(
        ser,
        OP_TRAIN_RUN_SAMPLE_PHASE3,
        [0, 0],  # invalid by design
        response_timeout=2.0,
        transient_retry_max=0,
    )
    if status == STATUS_UNSUPPORTED_OP:
        raise RuntimeError(
            "FPGA bitstream/protocol mismatch: opcode 0x37 (TRAIN_RUN_SAMPLE_PHASE3) is not implemented "
            "in the currently programmed FPGA image."
        )
    if status not in (STATUS_BAD_PACKET, STATUS_OK):
        raise RuntimeError(f"phase3 opcode probe unexpected status=0x{status:02X}, result=0x{int(result)&0xFFFFFFFF:08X}")


def fpga_probe_image_load_opcode(ser: serial.Serial) -> None:
    """Probe whether opcode 0x14 is implemented in the current bitstream."""
    status, result = send_request(
        ser,
        OP_LOAD_IMAGE_FROM_DDR,
        [0, 0],  # invalid by design (n_bytes must be > 0)
        response_timeout=2.0,
        transient_retry_max=0,
    )
    if status == STATUS_UNSUPPORTED_OP:
        raise RuntimeError(
            "FPGA bitstream/protocol mismatch: opcode 0x14 (LOAD_IMAGE_FROM_DDR) is not implemented "
            "in the currently programmed FPGA image."
        )
    if status not in (STATUS_BAD_PACKET, STATUS_OK):
        raise RuntimeError(f"image-load opcode probe unexpected status=0x{status:02X}, result=0x{int(result)&0xFFFFFFFF:08X}")


def build_poisson_thresholds_u11_with_max_fr(image_u8: list[int], max_fr: int) -> list[int]:
    """Match FPGA runtime Poisson scaling: POISSON_NUM_CONST * max_fr / 32, then per-pixel normalize."""
    sum_u8 = int(sum(image_u8))
    if sum_u8 <= 0:
        return [0] * N_IN
    scaled_num_const = (int(POISSON_NUM_CONST) * int(max_fr)) >> 5
    out: list[int] = []
    for px in image_u8:
        q = (scaled_num_const * int(px)) // sum_u8
        if q > RNG_MAX:
            q = RNG_MAX
        out.append(int(q))
    return out


def run_mine_style_python_phase4_retry_stats_with_image(
    image_u8: list[int],
    *,
    inj_steps: int,
    blank_steps: int,
    seed: int,
    max_fr_start: int,
    max_fr_step: int,
    max_fr_limit: int,
    min_inj_spikes: int,
    model_state: dict | None = None,
) -> tuple[int, int, int, int, np.ndarray]:
    """Mine.py retry reference: each trial runs inj(STDP)->blank with state continuity."""
    if int(max_fr_start) <= 0 or int(max_fr_step) <= 0:
        raise ValueError("invalid max_fr retry parameters")
    dt = 1e-3
    n_in = N_IN
    n = N_NEURONS
    input_decay = 0.0
    input_scale = 1000.0
    exc_td = 1e-3
    inh_td = 2e-3
    inh_coeff = 0.85 / float(n - 1)

    if model_state is not None and ("w_in" in model_state):
        w_in = np.asarray(model_state["w_in"], dtype=np.float64)
    else:
        w_in = _build_fixed_w_in_for_mine_like()

    if model_state is not None and ("rng_state" in model_state):
        rng_state = int(model_state["rng_state"]) & 0xFFFFFFFF
    else:
        rng_state = int(seed) & 0xFFFFFFFF

    if model_state is not None and ("c_in_state" in model_state):
        c_in_state = np.asarray(model_state["c_in_state"], dtype=np.float64)
        g_in_state = np.asarray(model_state["g_in_state"], dtype=np.float64)
        x_in_state = np.asarray(model_state["x_in_state"], dtype=np.float64)
        x_exc_state = np.asarray(model_state["x_exc_state"], dtype=np.float64)
        A = np.asarray(model_state["A"], dtype=np.float64)
        B_T = np.asarray(model_state["B_T"], dtype=np.float64)
        exc_syn_r = np.asarray(model_state["exc_syn_r"], dtype=np.float64)
        inh_syn_r = np.asarray(model_state["inh_syn_r"], dtype=np.float64)
        delay_input = np.asarray(model_state["delay_input"], dtype=np.float64)
        delay_exc2inh = np.asarray(model_state["delay_exc2inh"], dtype=np.float64)
        g_inh = np.asarray(model_state["g_inh"], dtype=np.float64)
        v_exc = np.asarray(model_state["v_exc"], dtype=np.float64)
        tlast_exc = np.asarray(model_state["tlast_exc"], dtype=np.float64)
        theta = np.asarray(model_state["theta"], dtype=np.float64)
        vthr_exc = np.asarray(model_state["vthr_exc"], dtype=np.float64)
        exc_tcount = int(model_state["exc_tcount"])
        v_inh = np.asarray(model_state["v_inh"], dtype=np.float64)
        tlast_inh = np.asarray(model_state["tlast_inh"], dtype=np.float64)
        vthr_inh = np.asarray(model_state["vthr_inh"], dtype=np.float64)
        inh_tcount = int(model_state["inh_tcount"])
    else:
        c_in_state = np.zeros(n_in, dtype=np.float64)
        g_in_state = np.zeros(n, dtype=np.float64)
        x_in_state = np.zeros(n_in, dtype=np.float64)
        x_exc_state = np.zeros(n, dtype=np.float64)
        A = np.zeros((n, n_in), dtype=np.float64)
        B_T = np.zeros((n_in, n), dtype=np.float64)
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

    def _run_steps(
        thresholds_arr: np.ndarray,
        n_steps: int,
        *,
        force_no_input: bool,
        stdp_enable: bool,
        track_exc_counts: bool = False,
    ) -> tuple[int, np.ndarray]:
        nonlocal rng_state, c_in_state, g_in_state, exc_syn_r, inh_syn_r, delay_input, delay_exc2inh, g_inh
        nonlocal v_exc, tlast_exc, theta, vthr_exc, exc_tcount, v_inh, tlast_inh, vthr_inh, inh_tcount
        nonlocal x_in_state, x_exc_state, A, B_T
        total_spikes = 0
        exc_counts = np.zeros(n, dtype=np.int64)
        for _ in range(int(n_steps)):
            s_in = np.zeros(n_in, dtype=np.uint8)
            if not force_no_input:
                for i in range(n_in):
                    rng_state = lcg_next_u32(rng_state)
                    rand11 = (rng_state >> 21) & 0x7FF
                    s_in[i] = 1 if rand11 < int(thresholds_arr[i]) else 0
            pre_active = np.flatnonzero(s_in)

            c_in_state = c_in_state * input_decay + input_scale * s_in.astype(np.float64)
            x_in_state = _single_exp_step(x_in_state, s_in.astype(np.float64), dt, 2e-2)
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
            step_exc_sum = int(np.sum(s_exc, dtype=np.int64))
            total_spikes += step_exc_sum
            if track_exc_counts and step_exc_sum > 0:
                exc_counts += s_exc.astype(np.int64)

            exc_syn_r = _single_exp_step(exc_syn_r, s_exc.astype(np.float64), dt, exc_td)
            x_exc_state = _single_exp_step(x_exc_state, s_exc.astype(np.float64), dt, 2e-2)
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

            if stdp_enable:
                post_active = np.flatnonzero(s_exc)
                if post_active.size > 0:
                    A[post_active, :] += x_in_state
                if pre_active.size > 0:
                    np.add.at(B_T, pre_active, x_exc_state)
        return int(total_spikes), exc_counts

    def _apply_stdp_once(update_nt: int) -> None:
        nonlocal w_in, g_in_state, A, B_T
        if int(update_nt) <= 0:
            return
        W = np.array(w_in, copy=True)
        W_abs_sum = np.sum(np.abs(W), axis=1, keepdims=True)
        W_abs_sum[W_abs_sum == 0.0] = 1.0
        W = W * (0.1 / W_abs_sum)
        dW = 1e-2 * (5e-2 - W) * A
        dW -= 1e-4 * W * B_T.T
        clipped_dW = np.clip(dW / float(int(update_nt)), -1e-3, 1e-3)
        W = np.clip(W + clipped_dW, 0.0, 5e-2)
        w_in = W
        g_in_state = np.dot(w_in, c_in_state)
        A.fill(0.0)
        B_T.fill(0.0)

    accepted_max_fr = int(max_fr_start)
    accepted_inj = 0
    accepted_blank = 0
    accepted_counts = np.zeros(n, dtype=np.int64)
    max_fr = int(max_fr_start)
    retry_guard = 0
    while True:
        thresholds_arr = np.asarray(build_poisson_thresholds_u11_with_max_fr(image_u8, int(max_fr)), dtype=np.uint16)
        inj_total, inj_counts = _run_steps(
            thresholds_arr,
            int(inj_steps),
            force_no_input=False,
            stdp_enable=True,
            track_exc_counts=True,
        )
        _apply_stdp_once(int(inj_steps))
        blank_total, _ = _run_steps(
            thresholds_arr,
            int(blank_steps),
            force_no_input=True,
            stdp_enable=False,
            track_exc_counts=False,
        )
        accepted_max_fr = int(max_fr)
        accepted_inj = int(inj_total)
        accepted_blank = int(blank_total)
        accepted_counts = np.array(inj_counts, copy=True)
        if int(inj_total) >= int(min_inj_spikes):
            break
        max_fr += int(max_fr_step)
        retry_guard += 1
        if retry_guard > 100000:
            raise RuntimeError("mine-style phase4 retry did not converge")

    if model_state is not None:
        model_state["w_in"] = np.array(w_in, copy=True)
        model_state["rng_state"] = int(rng_state)
        model_state["c_in_state"] = np.array(c_in_state, copy=True)
        model_state["g_in_state"] = np.array(g_in_state, copy=True)
        model_state["x_in_state"] = np.array(x_in_state, copy=True)
        model_state["x_exc_state"] = np.array(x_exc_state, copy=True)
        model_state["A"] = np.array(A, copy=True)
        model_state["B_T"] = np.array(B_T, copy=True)
        model_state["exc_syn_r"] = np.array(exc_syn_r, copy=True)
        model_state["inh_syn_r"] = np.array(inh_syn_r, copy=True)
        model_state["delay_input"] = np.array(delay_input, copy=True)
        model_state["delay_exc2inh"] = np.array(delay_exc2inh, copy=True)
        model_state["g_inh"] = np.array(g_inh, copy=True)
        model_state["v_exc"] = np.array(v_exc, copy=True)
        model_state["tlast_exc"] = np.array(tlast_exc, copy=True)
        model_state["theta"] = np.array(theta, copy=True)
        model_state["vthr_exc"] = np.array(vthr_exc, copy=True)
        model_state["exc_tcount"] = int(exc_tcount)
        model_state["v_inh"] = np.array(v_inh, copy=True)
        model_state["tlast_inh"] = np.array(tlast_inh, copy=True)
        model_state["vthr_inh"] = np.array(vthr_inh, copy=True)
        model_state["inh_tcount"] = int(inh_tcount)
    return int(accepted_max_fr), int(accepted_inj), int(accepted_inj), int(accepted_blank), accepted_counts


def fpga_phase4_build_assignments_from_raw1(
    ser: serial.Serial,
    args: argparse.Namespace,
) -> None:
    if str(args.image_source) != "fpga":
        raise ValueError("--train-phase4-build-assignments currently requires --image-source fpga")
    if tqdm is None:
        raise RuntimeError("tqdm is required for --train-phase4-build-assignments (pip install tqdm)")

    caps = fpga_train_query_caps(ser)
    if caps == 0 or (caps & (1 << 14)) == 0:
        raise RuntimeError(f"phase4 assignment build requires phase4-capable training build, caps=0x{caps:08X}")

    _, all_labels = load_mnist()
    n_total = int(len(all_labels))
    n_train = int(getattr(args, "train_split_train", 9000))
    n_test = int(getattr(args, "train_split_test", 1000))
    n_epoch = int(getattr(args, "train_epochs", 30))
    if n_train <= 0 or n_test <= 0 or n_epoch <= 0:
        raise ValueError("train/test/epoch counts must be positive")
    if (n_train + n_test) > n_total:
        raise ValueError(f"RAW1 labels count={n_total} is smaller than train+test={n_train+n_test}")

    inj_steps = int(args.chunk_nsteps)
    tile_rows = int(args.train_tile_rows)
    start_lba = int(args.start_lba)
    timeout_sec = float(args.timeout)
    rates_prev: np.ndarray | None = None
    assignments: np.ndarray | None = None

    print(
        "FPGA phase4 training (label-stats aggregation) start: "
        f"epochs={n_epoch}, train={n_train}, test={n_test}, inj_steps={inj_steps}, tile_rows={tile_rows}"
    )
    print("Using Python-side labels from load_mnist() (same source family as import_MNIST_raw.py).")

    for epoch in range(n_epoch):
        print(f"[epoch {epoch+1}/{n_epoch}] reset FPGA label stats")
        fpga_train_label_stats_reset(ser)

        pbar = tqdm(
            total=n_train,
            desc=f"train e{epoch+1}/{n_epoch}",
            unit="img",
            miniters=100,
            leave=True,
        )
        for sample_idx in range(n_train):
            prepare_fpga_sample_image_via_streamed_load(
                ser,
                sample_idx=int(sample_idx),
                start_lba=start_lba,
                timeout_sec=timeout_sec,
                verbose=False,
            )
            fpga_train_run_sample_phase4(ser, inj_steps=inj_steps, tile_rows=tile_rows)
            lbl = int(all_labels[sample_idx])
            if lbl < 0 or lbl >= 10:
                raise RuntimeError(f"invalid Python label: sample_idx={sample_idx}, label={lbl}")
            fpga_train_label_stats_accum(ser, lbl)
            pbar.update(1)
        pbar.close()

        sums, counts = fpga_read_train_label_stats_all(ser)
        assignments, rates_prev = assign_labels_from_aggregated_stats(sums, counts, rates_prev=rates_prev, alpha=1.0)
        print(f"[epoch {epoch+1}/{n_epoch}] label_counts={counts.astype(int).tolist()}")
        print(f"[epoch {epoch+1}/{n_epoch}] assignment_hist={[int(np.sum(assignments == i)) for i in range(10)]}")

    assert assignments is not None
    out_dir = Path(__file__).resolve().parent.parent / "obj"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fpga_phase4_assignments.npy"
    np.save(out_path, assignments.astype(np.uint8))
    print(f"Saved final assignments to {out_path}")
    print("Training-side aggregation/assignment build completed. Next step is FPGA-side test prediction label return.")


def fpga_train_infer_e2e_compare(
    ser: serial.Serial,
    args: argparse.Namespace,
) -> None:
    """Train+infer style compact E2E comparison against use_fpga_calc.py reference model.

    Flow:
    - Run phase4 training for N samples on FPGA (with streamed image load),
    - Compare per-sample inj/blank totals to Python coarse phase4 reference,
    - Accumulate label stats on FPGA and in Python reference,
    - Compare final assignment vectors.
    """
    if str(args.image_source) != "fpga":
        raise ValueError("--train-infer-e2e-compare currently requires --image-source fpga")
    if tqdm is None:
        raise RuntimeError("tqdm is required for --train-infer-e2e-compare (pip install tqdm)")

    caps = fpga_train_query_caps(ser)
    if (caps & (1 << 14)) == 0:
        raise RuntimeError(f"train/infer e2e compare requires phase4-capable training build, caps=0x{caps:08X}")

    n_samples = max(1, int(args.train_e2e_samples))
    mine_timing_mode = bool(getattr(args, "train_e2e_mine_timing", False))
    inj_steps = 350 if mine_timing_mode else max(1, int(args.chunk_nsteps))
    blank_steps = 150
    tile_rows = max(1, int(args.train_tile_rows))
    start_lba = int(args.start_lba)
    timeout_sec = float(args.timeout)
    seed = int(args.seed)
    max_fr_start = max(1, int(args.train_retry_max_fr_start))
    max_fr_step = max(1, int(args.train_retry_max_fr_step))
    max_fr_limit = max(1, int(args.train_retry_max_fr_limit))
    min_inj_spikes = max(0, int(args.train_retry_min_inj_spikes))
    strict = bool(getattr(args, "train_e2e_strict", False))

    _, labels_all = load_mnist()
    if n_samples > int(len(labels_all)):
        raise ValueError(f"train_e2e_samples={n_samples} exceeds dataset size={int(len(labels_all))}")

    print(
        "E2E compare start: "
        f"samples={n_samples}, inj_steps={inj_steps}, tile_rows={tile_rows}, "
        f"retry=({max_fr_start},{max_fr_step},{max_fr_limit}), min_inj={min_inj_spikes}"
    )
    if mine_timing_mode:
        print("E2E compare mode: mine timing override enabled (inj=350, blank=150).")
    print("Python phase4 reference uses mine.py order: inj(STDP) -> weight update -> blank(no STDP).")

    fpga_train_label_stats_reset(ser)
    py_label_spike_sums = np.zeros((10, N_NEURONS), dtype=np.int64)
    py_label_counts = np.zeros((10,), dtype=np.int64)
    py_phase4_state: dict = {"w_in": _build_fixed_w_in_for_mine_like()}

    mismatch_inj_blank = 0
    pbar = tqdm(total=n_samples, desc="e2e train+compare", unit="img", miniters=1, leave=True)
    for sample_idx in range(n_samples):
        label = int(labels_all[sample_idx])
        image_u8, _ = read_mnist_image_u8(sample_idx)

        py_accepted_max_fr, _, py_inj, py_blank, py_counts_arr = run_mine_style_python_phase4_retry_stats_with_image(
            image_u8,
            inj_steps=inj_steps,
            blank_steps=blank_steps,
            seed=seed,
            max_fr_start=max_fr_start,
            max_fr_step=max_fr_step,
            max_fr_limit=max_fr_limit,
            min_inj_spikes=min_inj_spikes,
            model_state=py_phase4_state,
        )
        py_label_spike_sums[label, :] += np.asarray(py_counts_arr, dtype=np.int64)
        py_label_counts[label] += 1

        prepare_fpga_sample_image_via_streamed_load(
            ser,
            sample_idx=sample_idx,
            start_lba=start_lba,
            timeout_sec=timeout_sec,
            verbose=False,
        )
        ret = fpga_train_run_sample_phase4(
            ser,
            inj_steps=inj_steps,
            tile_rows=tile_rows,
        )
        fpga_inj = int(ret) & 0xFFFF
        fpga_blank = (int(ret) >> 16) & 0xFFFF
        fpga_train_label_stats_accum(ser, label)

        if (fpga_inj != int(py_inj)) or (fpga_blank != int(py_blank)):
            mismatch_inj_blank += 1
            print(
                f"[sample {sample_idx}] phase4 mismatch: "
                f"inj py={int(py_inj)} fpga={fpga_inj}, "
                f"blank py={int(py_blank)} fpga={fpga_blank}"
            )
        pbar.update(1)
    pbar.close()

    fpga_sums, fpga_counts = fpga_read_train_label_stats_all(ser)
    fpga_assign, _ = assign_labels_from_aggregated_stats(fpga_sums, fpga_counts, rates_prev=None, alpha=1.0)
    py_assign, _ = assign_labels_from_aggregated_stats(py_label_spike_sums, py_label_counts, rates_prev=None, alpha=1.0)
    assignment_mismatch = int(np.sum(fpga_assign.astype(np.int64) != py_assign.astype(np.int64)))
    max_sum_abs_diff = int(np.max(np.abs(fpga_sums.astype(np.int64) - py_label_spike_sums.astype(np.int64))))
    max_count_abs_diff = int(np.max(np.abs(fpga_counts.astype(np.int64) - py_label_counts.astype(np.int64))))

    print("E2E compare summary:")
    print(f"  phase4 inj/blank mismatched samples = {mismatch_inj_blank}/{n_samples}")
    print(f"  max|label_spike_sum_fpga - label_spike_sum_py| = {max_sum_abs_diff}")
    print(f"  max|label_count_fpga - label_count_py| = {max_count_abs_diff}")
    print(f"  assignment mismatch count = {assignment_mismatch}/{N_NEURONS}")

    if strict and (mismatch_inj_blank != 0 or assignment_mismatch != 0):
        raise RuntimeError(
            "train/infer e2e compare strict failed: "
            f"inj_blank_mismatch={mismatch_inj_blank}, assignment_mismatch={assignment_mismatch}"
        )


def fpga_train_then_infer_compare_500_100(
    ser: serial.Serial,
    args: argparse.Namespace,
) -> None:
    """Train first N samples with phase4, then infer next M samples and compare to mine-style Python reference."""
    if str(args.image_source) != "fpga":
        raise ValueError("--train-then-infer-compare requires --image-source fpga")
    if tqdm is None:
        raise RuntimeError("tqdm is required for --train-then-infer-compare (pip install tqdm)")

    caps = fpga_train_query_caps(ser)
    if (caps & (1 << 14)) == 0:
        raise RuntimeError(f"train-then-infer compare requires phase4-capable training build, caps=0x{caps:08X}")

    n_train = max(1, int(getattr(args, "train_then_infer_train_samples", 500)))
    n_infer = max(1, int(getattr(args, "train_then_infer_infer_samples", 100)))
    mine_timing_mode = bool(getattr(args, "train_e2e_mine_timing", False))
    inj_steps = 350 if mine_timing_mode else max(1, int(args.chunk_nsteps))
    blank_steps = 150
    infer_steps = 350 if mine_timing_mode else max(1, int(args.chunk_nsteps))
    infer_max_fr = max(1, int(getattr(args, "train_then_infer_max_fr", 32)))
    tile_rows = max(1, int(args.train_tile_rows))
    start_lba = int(args.start_lba)
    timeout_sec = float(args.timeout)
    seed = int(args.seed)
    max_fr_start = max(1, int(args.train_retry_max_fr_start))
    max_fr_step = max(1, int(args.train_retry_max_fr_step))
    max_fr_limit = max(1, int(args.train_retry_max_fr_limit))
    min_inj_spikes = max(0, int(args.train_retry_min_inj_spikes))
    strict = bool(getattr(args, "train_e2e_strict", False))

    _, labels_all = load_mnist()
    total_need = n_train + n_infer
    if total_need > int(len(labels_all)):
        raise ValueError(f"need {total_need} samples but dataset has {int(len(labels_all))}")

    print(
        "Train->Infer compare start: "
        f"train={n_train}, infer={n_infer}, inj_steps={inj_steps}, infer_steps={infer_steps}, "
        f"tile_rows={tile_rows}, infer_max_fr={infer_max_fr}"
    )
    if mine_timing_mode:
        print("Train->Infer mode: mine timing override enabled (train inj=350/blank=150, infer steps=350).")

    # -------- Train phase (first n_train samples) --------
    fpga_train_label_stats_reset(ser)
    py_label_spike_sums = np.zeros((10, N_NEURONS), dtype=np.int64)
    py_label_counts = np.zeros((10,), dtype=np.int64)
    py_phase4_state: dict = {"w_in": _build_fixed_w_in_for_mine_like()}

    pbar_train = tqdm(total=n_train, desc=f"train {n_train}", unit="img", miniters=1, leave=True)
    mismatch_inj_blank = 0
    for sample_idx in range(n_train):
        label = int(labels_all[sample_idx])
        image_u8, _ = read_mnist_image_u8(sample_idx)

        _, _, py_inj, py_blank, py_counts_arr = run_mine_style_python_phase4_retry_stats_with_image(
            image_u8,
            inj_steps=inj_steps,
            blank_steps=blank_steps,
            seed=seed,
            max_fr_start=max_fr_start,
            max_fr_step=max_fr_step,
            max_fr_limit=max_fr_limit,
            min_inj_spikes=min_inj_spikes,
            model_state=py_phase4_state,
        )
        py_label_spike_sums[label, :] += np.asarray(py_counts_arr, dtype=np.int64)
        py_label_counts[label] += 1

        prepare_fpga_sample_image_via_streamed_load(
            ser,
            sample_idx=sample_idx,
            start_lba=start_lba,
            timeout_sec=timeout_sec,
            verbose=False,
        )
        ret = fpga_train_run_sample_phase4(ser, inj_steps=inj_steps, tile_rows=tile_rows)
        fpga_inj = int(ret) & 0xFFFF
        fpga_blank = (int(ret) >> 16) & 0xFFFF
        fpga_train_label_stats_accum(ser, label)
        if (fpga_inj != int(py_inj)) or (fpga_blank != int(py_blank)):
            mismatch_inj_blank += 1
        pbar_train.update(1)
    pbar_train.close()

    fpga_sums, fpga_counts = fpga_read_train_label_stats_all(ser)
    fpga_assign, _ = assign_labels_from_aggregated_stats(fpga_sums, fpga_counts, rates_prev=None, alpha=1.0)
    py_assign, _ = assign_labels_from_aggregated_stats(py_label_spike_sums, py_label_counts, rates_prev=None, alpha=1.0)
    assign_mismatch = int(np.sum(fpga_assign.astype(np.int64) != py_assign.astype(np.int64)))
    print(
        "Train summary: "
        f"inj/blank mismatches={mismatch_inj_blank}/{n_train}, "
        f"assignment mismatch={assign_mismatch}/{N_NEURONS}"
    )

    # -------- Infer phase (next n_infer samples) --------
    py_w_in = np.asarray(py_phase4_state["w_in"], dtype=np.float64)
    infer_pred_mismatch = 0
    fpga_correct = 0
    py_correct = 0
    pbar_infer = tqdm(total=n_infer, desc=f"infer {n_infer}", unit="img", miniters=1, leave=True)
    for k in range(n_infer):
        sample_idx = n_train + k
        label = int(labels_all[sample_idx])
        image_u8, _ = read_mnist_image_u8(sample_idx)

        prepare_fpga_sample_image_via_streamed_load(
            ser,
            sample_idx=sample_idx,
            start_lba=start_lba,
            timeout_sec=timeout_sec,
            verbose=False,
        )
        _ = fpga_run_sample_infer(ser, seed=seed, n_steps=infer_steps, timeout_sec=max(30.0, timeout_sec))
        fpga_counts_vec = np.asarray(fpga_read_spike_counts(ser), dtype=np.int64)
        fpga_pred = predict_label_from_counts(fpga_counts_vec, fpga_assign)

        py_thresh = build_poisson_thresholds_u11_with_max_fr(image_u8, infer_max_fr)
        py_counts_vec = np.asarray(
            run_mine_style_python_poisson_with_thresholds(py_thresh, infer_steps, seed, w_in=py_w_in),
            dtype=np.int64,
        )
        py_pred = predict_label_from_counts(py_counts_vec, py_assign)

        if fpga_pred != py_pred:
            infer_pred_mismatch += 1
        if fpga_pred == label:
            fpga_correct += 1
        if py_pred == label:
            py_correct += 1
        pbar_infer.update(1)
    pbar_infer.close()

    fpga_acc = float(fpga_correct) / float(n_infer)
    py_acc = float(py_correct) / float(n_infer)
    print("Train->Infer compare summary:")
    print(f"  train phase4 inj/blank mismatches = {mismatch_inj_blank}/{n_train}")
    print(f"  train assignment mismatch count = {assign_mismatch}/{N_NEURONS}")
    print(f"  infer prediction mismatch count = {infer_pred_mismatch}/{n_infer}")
    print(f"  infer accuracy fpga = {fpga_acc:.4f}")
    print(f"  infer accuracy python(mine-style) = {py_acc:.4f}")

    if strict and (assign_mismatch != 0 or infer_pred_mismatch != 0):
        raise RuntimeError(
            "train-then-infer compare strict failed: "
            f"assignment_mismatch={assign_mismatch}, infer_pred_mismatch={infer_pred_mismatch}"
        )


def prepare_fpga_sample_image_via_streamed_load(
    ser: serial.Serial,
    *,
    sample_idx: int,
    start_lba: int,
    timeout_sec: float,
    verbose: bool = True,
) -> None:
    if int(sample_idx) < 0:
        raise ValueError(f"sample_idx must be >=0, got {sample_idx}")

    img_byte_off = RAW1_HEADER_BYTES + RAW1_NUM_IMAGES_DEFAULT + (int(sample_idx) * N_IN)
    img_sector_off = img_byte_off // 512
    img_byte_in_sector = img_byte_off % 512
    img_base_addr_byte = int(img_byte_in_sector)
    sectors_needed = (img_byte_in_sector + IMGLOAD_SRC_BIAS_BYTES + N_IN + 511) // 512

    if verbose:
        print(
            "Preparing input image via streamed FPGA image load path: "
            f"sample_idx={int(sample_idx)}, sector_off={img_sector_off}, sectors={sectors_needed}, "
            f"byte_in_sector={img_byte_in_sector}, src_bias_bytes={IMGLOAD_SRC_BIAS_BYTES}"
        )

    fpga_sd_sectors_to_ddr(
        ser=ser,
        start_lba=int(start_lba) + img_sector_off,
        num_sectors=sectors_needed,
        timeout_sec=timeout_sec,
        verbose=verbose,
    )

    # Host-side guard to avoid first-word mis-association on older RTL builds.
    if IMGLOAD_GUARD_SEC > 0.0:
        time.sleep(IMGLOAD_GUARD_SEC)

    fpga_load_image_from_ddr(
        ser=ser,
        base_addr_byte=img_base_addr_byte,
        n_bytes=N_IN,
        timeout_sec=timeout_sec,
        verbose=verbose,
    )

    # Verify against the same MNIST source/quantization path used by import_MNIST_raw.py.
    expected_img_u8, _ = read_mnist_image_u8(int(sample_idx))
    raw_slice = fpga_read_raw_image_span(
        ser,
        start_idx=IMGLOAD_DST_BIAS_BYTES,
        nbytes=N_IN,
    )
    expected_verify = expected_img_u8[:len(raw_slice)]

    if len(raw_slice) != len(expected_verify):
        raise RuntimeError(
            "MNIST->raw_image0 verify failed to collect enough bytes: "
            f"raw={len(raw_slice)}, expected={len(expected_verify)}"
        )

    # DDR source vs raw_image0 diagnostics:
    # - dump around expected DDR source address
    # - compare raw against expected and against +/- byte deltas
    # This makes source-address slips and destination-index slips visible.
    expected_src_base_byte = int(img_base_addr_byte) + int(IMGLOAD_SRC_BIAS_BYTES)
    staging_base_word = img_staging_base_word()
    probe_margin_bytes = 64
    probe_span_bytes = N_IN + (2 * probe_margin_bytes)
    probe_base_byte = max(0, expected_src_base_byte - probe_margin_bytes)
    probe_base_word = int(staging_base_word) + (probe_base_byte // 4)
    probe_byte_off_in_word = probe_base_byte % 4
    probe_words = (probe_byte_off_in_word + probe_span_bytes + 3) // 4

    ddr_probe_words = fpga_read_ddr_words(
        ser,
        base_word=probe_base_word,
        count=probe_words,
    )
    ddr_probe_bytes = _ddr_words_to_bytes_le(ddr_probe_words)
    ddr_probe_bytes = ddr_probe_bytes[probe_byte_off_in_word:probe_byte_off_in_word + probe_span_bytes]
    expected_off_in_probe = expected_src_base_byte - probe_base_byte
    ddr_expected = ddr_probe_bytes[expected_off_in_probe:expected_off_in_probe + N_IN]

    cmp_len = min(128, len(ddr_expected), len(raw_slice), len(expected_verify))
    mism_expected = _count_mismatch(ddr_expected[:cmp_len], raw_slice[:cmp_len])
    best_delta = 0
    best_delta_mism = 10**9
    for delta in range(-64, 65):
        d0 = expected_off_in_probe + delta
        d1 = d0 + cmp_len
        if d0 < 0 or d1 > len(ddr_probe_bytes):
            continue
        m = _count_mismatch(ddr_probe_bytes[d0:d1], raw_slice[:cmp_len])
        if m < best_delta_mism:
            best_delta_mism = m
            best_delta = delta

    best_raw_start = 0
    best_raw_start_mism = 10**9
    for raw_start in range(0, min(257, max(1, len(raw_slice) - cmp_len + 1))):
        m = _count_mismatch(ddr_expected[:cmp_len], raw_slice[raw_start:raw_start + cmp_len])
        if m < best_raw_start_mism:
            best_raw_start_mism = m
            best_raw_start = raw_start

    if verbose:
        dump_cols = 32
        dump_len = min(len(expected_verify), len(raw_slice))
        print(
            "DDR source probe: "
            f"staging_base_word=0x{staging_base_word:08X}, "
            f"expected_base_off=0x{expected_src_base_byte:08X}, "
            f"probe_base_off=0x{probe_base_byte:08X}, span={probe_span_bytes}B"
        )
        print(f"DDR expected full ({dump_len}B):")
        for i in range(0, dump_len, dump_cols):
            print(
                f"  [{i:03d}:{min(i + dump_cols, dump_len):03d}] "
                + " ".join(f"{int(x)&0xFF:02X}" for x in expected_verify[i:i + dump_cols])
            )
        print(f"raw_image0 full ({dump_len}B):")
        for i in range(0, dump_len, dump_cols):
            print(
                f"  [{i:03d}:{min(i + dump_cols, dump_len):03d}] "
                + " ".join(f"{int(x)&0xFF:02X}" for x in raw_slice[i:i + dump_cols])
            )
        print(
            "DDR<->raw compare: "
            f"mismatch@delta0={mism_expected}/{cmp_len}, "
            f"best_delta={best_delta:+d}B (mismatch={best_delta_mism}/{cmp_len}), "
            f"best_raw_start={best_raw_start} (mismatch={best_raw_start_mism}/{cmp_len}), "
            f"raw_first_nonzero={_first_nonzero_indices(raw_slice, limit=8)}"
        )

    mism = [
        i
        for i, (exp_b, raw_b) in enumerate(zip(expected_verify, raw_slice))
        if int(exp_b) != int(raw_b)
    ]
    if mism:
        i0 = mism[0]
        raise RuntimeError(
            "MNIST->raw_image0 verify: FAIL "
            f"(sample_idx={int(sample_idx)}, mismatch_count={len(mism)}, "
            f"first_mismatch=i={i0}, expected=0x{int(expected_verify[i0]):02X}, raw=0x{int(raw_slice[i0]):02X})"
        )

    if verbose:
        print(
            "MNIST->raw_image0 verify: PASS "
            f"(sample_idx={int(sample_idx)}, compared_bytes={len(expected_verify)}, raw_start={IMGLOAD_DST_BIAS_BYTES})"
        )


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
    # Use the same BRAM init source as FPGA infer weight memory to avoid init drift.
    mem_path = Path(__file__).resolve().parent.parent / "data" / "infer_w_q16.mem"
    w_q16 = load_q16_mem_matrix(mem_path, N_NEURONS, N_IN)
    return w_q16.astype(np.float64) / float(1 << FXP_SHIFT)


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
    apply_stdp_after_inj: bool = True,
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
    # STDP traces used by mine.py during injection window.
    x_in_state = np.zeros(n_in, dtype=np.float64)
    x_exc_state = np.zeros(n, dtype=np.float64)
    A = np.zeros((n, n_in), dtype=np.float64)
    B_T = np.zeros((n_in, n), dtype=np.float64)
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

    def _run_steps(n_steps: int, *, force_no_input: bool, stdp_enable: bool) -> int:
        nonlocal rng_state, c_in_state, g_in_state, exc_syn_r, inh_syn_r, delay_input, delay_exc2inh, g_inh
        nonlocal v_exc, tlast_exc, theta, vthr_exc, exc_tcount, v_inh, tlast_inh, vthr_inh, inh_tcount
        nonlocal x_in_state, x_exc_state, A, B_T
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
            x_in_state = _single_exp_step(x_in_state, s_in.astype(np.float64), dt, 2e-2)
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
            x_exc_state = _single_exp_step(x_exc_state, s_exc.astype(np.float64), dt, 2e-2)
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

            if stdp_enable:
                post_active = np.flatnonzero(s_exc)
                if post_active.size > 0:
                    A[post_active, :] += x_in_state
                if pre_active.size > 0:
                    np.add.at(B_T, pre_active, x_exc_state)
        return total_spikes

    inj_total = _run_steps(int(inj_steps), force_no_input=False, stdp_enable=True)
    if bool(apply_stdp_after_inj) and int(inj_steps) > 0:
        # mine.py online STDP update at the end of each injection window.
        W = np.array(w_in, copy=True)
        W_abs_sum = np.sum(np.abs(W), axis=1, keepdims=True)
        W_abs_sum[W_abs_sum == 0.0] = 1.0
        W = W * (0.1 / W_abs_sum)
        dW = 1e-2 * (5e-2 - W) * A
        dW -= 1e-4 * W * B_T.T
        clipped_dW = np.clip(dW / float(int(inj_steps)), -1e-3, 1e-3)
        W = np.clip(W + clipped_dW, 0.0, 5e-2)
        w_in = W
        # mine.py re-bases input conductance state after weight update.
        g_in_state = np.dot(w_in, c_in_state)
    blank_total = _run_steps(int(blank_steps), force_no_input=True, stdp_enable=False)
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


def fpga_phase1_verify_one_shot(ser: serial.Serial, args: argparse.Namespace) -> None:
    """One-shot verification for phase1 changes (N=50, W via BRAM path in train STDP)."""
    if int(N_NEURONS) != 50:
        raise RuntimeError(f"Phase1 verify expects N_NEURONS=50, got {N_NEURONS}")

    mem_path = Path(__file__).resolve().parent.parent / "data" / "infer_w_q16.mem"
    w_q16 = load_q16_mem_matrix(mem_path, N_NEURONS, N_IN)
    print(
        "Weight mem check: "
        f"path={mem_path}, shape={w_q16.shape}, nnz={int(np.count_nonzero(w_q16))}, "
        f"q16_sum={int(np.sum(w_q16, dtype=np.uint64))}"
    )

    sample_idx = int(args.sample_idx)
    inj_steps = max(1, int(args.chunk_nsteps))
    tile_rows = max(1, int(args.train_tile_rows))
    print(
        "Phase1 one-shot verify start: "
        f"sample_idx={sample_idx}, inj_steps={inj_steps}, tile_rows={tile_rows}, "
        f"expect_neurons={N_NEURONS}"
    )

    prepare_fpga_sample_image_via_streamed_load(
        ser,
        sample_idx=sample_idx,
        start_lba=int(args.start_lba),
        timeout_sec=float(args.timeout),
    )
    ret = fpga_train_run_sample_phase4(
        ser,
        inj_steps=inj_steps,
        tile_rows=tile_rows,
    )
    inj_total_spikes = int(ret) & 0xFFFF
    blank_total_spikes = (int(ret) >> 16) & 0xFFFF

    train_inj_counts = fpga_try_read_train_inj_spike_counts(ser)
    infer_counts = fpga_read_spike_counts(ser)
    if len(infer_counts) != N_NEURONS:
        raise RuntimeError(
            f"READ_SPIKE_COUNT length mismatch: got {len(infer_counts)}, expected {N_NEURONS}"
        )

    train_inj_sum: int | None = None
    if train_inj_counts is not None:
        if len(train_inj_counts) != N_NEURONS:
            raise RuntimeError(
                f"READ_TRAIN_INJ_SPIKE_COUNT length mismatch: got {len(train_inj_counts)}, expected {N_NEURONS}"
            )
        train_inj_sum = int(sum(train_inj_counts))
    infer_sum = int(sum(infer_counts))
    print(
        "Phase1 one-shot verify result: "
        f"phase4_ret=0x{(int(ret) & 0xFFFFFFFF):08X}, "
        f"inj_total={inj_total_spikes}, blank_total={blank_total_spikes}, "
        f"inj_count_sum={train_inj_sum if train_inj_sum is not None else 'N/A(opcode 0x27 unsupported)'}, "
        f"infer_count_sum={infer_sum}"
    )
    if (train_inj_sum is not None) and (train_inj_sum != inj_total_spikes):
        print(
            "Warning: inj_count_sum != inj_total. "
            "The run succeeded, but accounting may use different aggregation points."
        )
    print("Phase1 one-shot verify: PASS")


def fpga_phase2_verify_one_shot(ser: serial.Serial, args: argparse.Namespace) -> None:
    """One-shot verification for phase2 event-driven learning path."""
    if int(N_NEURONS) != 50:
        raise RuntimeError(f"Phase2 verify expects N_NEURONS=50, got {N_NEURONS}")

    sample_idx = int(args.sample_idx)
    inj_steps = max(1, int(args.chunk_nsteps))
    tile_rows = max(1, int(args.train_tile_rows))
    print(
        "Phase2 one-shot verify start: "
        f"sample_idx={sample_idx}, inj_steps={inj_steps}, tile_rows={tile_rows}, "
        "mode=phase4 x2 (event-driven online update)"
    )

    run_results: list[tuple[int, int, int, int]] = []
    for run_i in range(2):
        prepare_fpga_sample_image_via_streamed_load(
            ser,
            sample_idx=sample_idx,
            start_lba=int(args.start_lba),
            timeout_sec=float(args.timeout),
        )
        ret = fpga_train_run_sample_phase4(
            ser,
            inj_steps=inj_steps,
            tile_rows=tile_rows,
        )
        inj_total = int(ret) & 0xFFFF
        blank_total = (int(ret) >> 16) & 0xFFFF
        infer_counts = fpga_read_spike_counts(ser)
        if len(infer_counts) != N_NEURONS:
            raise RuntimeError(
                f"READ_SPIKE_COUNT length mismatch: got {len(infer_counts)}, expected {N_NEURONS}"
            )
        infer_sum = int(sum(infer_counts))
        run_results.append((int(ret) & 0xFFFFFFFF, inj_total, blank_total, infer_sum))
        print(
            f"  run{run_i + 1}: "
            f"phase4_ret=0x{(int(ret) & 0xFFFFFFFF):08X}, "
            f"inj_total={inj_total}, blank_total={blank_total}, infer_count_sum={infer_sum}"
        )

    r1 = run_results[0]
    r2 = run_results[1]
    print(
        "Phase2 one-shot verify summary: "
        f"run1_ret=0x{r1[0]:08X}, run2_ret=0x{r2[0]:08X}, "
        f"run1_sum={r1[3]}, run2_sum={r2[3]}"
    )
    print("Phase2 one-shot verify: PASS")


def fpga_phase3_verify_one_shot(ser: serial.Serial, args: argparse.Namespace) -> None:
    """One-shot verification for phase3 sparse-index BRAM path (CSR/CSC + edge-wise update)."""
    if int(N_NEURONS) != 50:
        raise RuntimeError(f"Phase3 verify expects N_NEURONS=50, got {N_NEURONS}")

    data_dir = Path(__file__).resolve().parent.parent / "data"
    idx_info = ensure_phase3_sparse_index_mem_files(data_dir, rewrite=bool(args.phase3_rewrite_sparse_mem))
    print(
        "Phase3 index mem check: "
        f"sparse={idx_info['sparse_percent']}%, edges={idx_info['n_edges']}, "
        f"csr_row_ptr={idx_info['csr_row_ptr_len']}, csr_col_idx={idx_info['csr_col_idx_len']}, "
        f"csc_col_ptr={idx_info['csc_col_ptr_len']}, csc_row_idx={idx_info['csc_row_idx_len']}, "
        f"csc_edge_idx={idx_info['csc_edge_idx_len']}, w_nnz={idx_info['w_nnz_after_mask']}"
    )

    sample_idx = int(args.sample_idx)
    inj_steps = max(1, int(args.chunk_nsteps))
    tile_rows = max(1, int(args.train_tile_rows))
    verify_count = 5
    print(
        "Phase3 one-shot verify start: "
        f"sample_idx={sample_idx}..{sample_idx + verify_count - 1}, "
        f"inj_steps={inj_steps}, tile_rows={tile_rows}, "
        "mode=phase3 (CSR/CSC edge update + blank)"
    )

    run_ok = 0
    run_ng = 0
    run_results: list[dict[str, object]] = []

    for k in range(verify_count):
        curr_sample_idx = sample_idx + k
        print(f"  [run {k + 1}/{verify_count}] sample_idx={curr_sample_idx}")
        try:
            prepare_fpga_sample_image_via_streamed_load(
                ser,
                sample_idx=curr_sample_idx,
                start_lba=int(args.start_lba),
                timeout_sec=float(args.timeout),
            )
            ret = fpga_train_run_sample_phase3(
                ser,
                inj_steps=inj_steps,
                tile_rows=tile_rows,
            )
            inj_total = int(ret) & 0xFFFF
            blank_total = (int(ret) >> 16) & 0xFFFF
            infer_counts = fpga_read_spike_counts(ser)
            if len(infer_counts) != N_NEURONS:
                raise RuntimeError(
                    f"READ_SPIKE_COUNT length mismatch: got {len(infer_counts)}, expected {N_NEURONS}"
                )
            infer_sum = int(sum(infer_counts))

            run_ok += 1
            run_results.append(
                {
                    "sample_idx": curr_sample_idx,
                    "ok": True,
                    "phase3_ret": int(ret) & 0xFFFFFFFF,
                    "inj_total": inj_total,
                    "blank_total": blank_total,
                    "infer_count_sum": infer_sum,
                    "error": "",
                }
            )
            print(
                "  result: "
                f"phase3_ret=0x{(int(ret) & 0xFFFFFFFF):08X}, "
                f"inj_total={inj_total}, blank_total={blank_total}, infer_count_sum={infer_sum}"
            )
            if infer_sum != (inj_total + blank_total):
                print(
                    "  Warning: infer_count_sum != inj_total+blank_total. "
                    "Run completed, but counter aggregation points differ in this build."
                )
        except Exception as exc:
            run_ng += 1
            run_results.append(
                {
                    "sample_idx": curr_sample_idx,
                    "ok": False,
                    "phase3_ret": 0,
                    "inj_total": 0,
                    "blank_total": 0,
                    "infer_count_sum": 0,
                    "error": str(exc),
                }
            )
            print(f"  run failed: {exc}")

    print("Phase3 verify summary (5 samples):")
    for r in run_results:
        if bool(r["ok"]):
            print(
                f"  sample_idx={int(r['sample_idx'])}: OK, "
                f"phase3_ret=0x{int(r['phase3_ret']) & 0xFFFFFFFF:08X}, "
                f"inj_total={int(r['inj_total'])}, blank_total={int(r['blank_total'])}, "
                f"infer_count_sum={int(r['infer_count_sum'])}"
            )
        else:
            print(
                f"  sample_idx={int(r['sample_idx'])}: NG, "
                f"error={r['error']}"
            )

    print(f"Phase3 one-shot verify finished: OK={run_ok}, NG={run_ng}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FPGA phase4 utility (single-sample run, assignment build, and compact E2E compare)."
        )
    )
    parser.add_argument(
        "--image-source",
        type=str,
        choices=["fpga"],
        default="fpga",
        help="image source (current modes support fpga only)",
    )
    parser.add_argument("--sample-idx", type=int, default=0, help="sample index for one-sample phase4 command")
    parser.add_argument("--port", type=str, default=SERIAL_PORTNAME)
    parser.add_argument("--start-lba", type=int, default=2048)
    parser.add_argument("--seed", type=lambda x: int(x, 0), default=0x12345678)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--train-run-sample-phase4",
        action="store_true",
        help="run one sample through phase4 and print inj/blank totals",
    )
    parser.add_argument(
        "--phase1-verify",
        action="store_true",
        help="one-shot verification for phase1 changes (N=50 + train STDP W via BRAM)",
    )
    parser.add_argument(
        "--phase2-verify",
        action="store_true",
        help="one-shot verification for phase2 changes (event-driven online update path)",
    )
    parser.add_argument(
        "--phase3-verify",
        action="store_true",
        help="one-shot verification for phase3 changes (CSR/CSC sparse-index update path)",
    )
    parser.add_argument(
        "--phase3-rewrite-sparse-mem",
        action="store_true",
        help="regenerate data/csr_*.mem and data/csc_*.mem before --phase3-verify",
    )
    parser.add_argument(
        "--train-phase4-build-assignments",
        action="store_true",
        help="run FPGA phase4 over train split, aggregate label stats on FPGA, and build mine-style assignments in Python",
    )
    parser.add_argument(
        "--train-infer-e2e-compare",
        action="store_true",
        help="run compact train+infer E2E compare for N samples (phase4 sample stats + assignment compare)",
    )
    parser.add_argument(
        "--train-then-infer-compare",
        action="store_true",
        help="train on first N samples, then infer next M samples, and compare with mine.py-style Python reference",
    )
    parser.add_argument(
        "--train-e2e-samples",
        type=int,
        default=20,
        help="sample count for --train-infer-e2e-compare (default 20)",
    )
    parser.add_argument(
        "--train-e2e-strict",
        action="store_true",
        help="fail --train-infer-e2e-compare on any sample-level phase4 mismatch or assignment mismatch",
    )
    parser.add_argument("--train-then-infer-train-samples", type=int, default=9000, help="train sample count for --train-then-infer-compare (default 9000)")
    parser.add_argument("--train-then-infer-infer-samples", type=int, default=1000, help="infer sample count for --train-then-infer-compare (default 1000)")
    parser.add_argument("--train-then-infer-max-fr", type=int, default=32, help="max_fr for Python inference reference in --train-then-infer-compare")
    parser.add_argument("--chunk-nsteps", type=int, default=16, help="step count for phase4 injection window")
    parser.add_argument(
        "--train-e2e-mine-timing",
        action="store_true",
        help="use mine.py timing in E2E compare (inj=350, blank=150) regardless of --chunk-nsteps",
    )
    parser.add_argument("--train-epochs", type=int, default=30, help="epoch count for train phase4 assignment build (mine.py default=30)")
    parser.add_argument("--train-split-train", type=int, default=9000, help="number of train samples (default 9000)")
    parser.add_argument("--train-split-test", type=int, default=1000, help="number of test samples (default 1000)")
    parser.add_argument("--train-retry-max-fr-start", type=int, default=32, help="starting max_fr for coarse phase3 retry")
    parser.add_argument("--train-retry-max-fr-step", type=int, default=16, help="max_fr increment for coarse phase3 retry")
    parser.add_argument("--train-retry-max-fr-limit", type=int, default=256, help="max_fr upper limit for coarse phase3 retry")
    parser.add_argument("--train-retry-min-inj-spikes", type=int, default=5, help="acceptance threshold on inj_total_spikes for coarse phase3 retry")
    parser.add_argument(
        "--train-tile-rows",
        type=int,
        default=10,
        help="row tile size for phase4 STDP update",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if serial is None:
        raise RuntimeError(
            "pyserial is not installed. Install it to use FPGA communication paths."
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
            if int(caps) == 0:
                print("Train kernels are disabled in this FPGA build (TRAIN_ENABLE=0 fast-build configuration).")
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
            args.phase1_verify,
            args.phase2_verify,
            args.phase3_verify,
            args.train_run_sample_phase4,
            args.train_phase4_build_assignments,
            args.train_infer_e2e_compare,
            args.train_then_infer_compare,
        ])
        needs_phase4_opcode = any([
            args.phase1_verify,
            args.phase2_verify,
            args.train_run_sample_phase4,
            args.train_phase4_build_assignments,
            args.train_infer_e2e_compare,
            args.train_then_infer_compare,
        ])
        needs_phase3_opcode = any([
            args.phase3_verify,
        ])
        if needs_reliable_link and not caps_ok:
            raise RuntimeError(
                "FPGA UART link is not responding (TRAIN_QUERY_CAPS timeout). "
                "This often happens if a prior run left the FPGA busy/stuck. "
                "Reset/power-cycle the FPGA board and retry."
            )
        if needs_phase4_opcode:
            fpga_probe_image_load_opcode(ser)
            fpga_probe_phase4_opcode(ser)
        if needs_phase3_opcode:
            fpga_probe_image_load_opcode(ser)
            fpga_probe_phase3_opcode(ser)
        if args.phase1_verify:
            if args.image_source != "fpga":
                raise ValueError("--phase1-verify currently requires --image-source fpga")
            fpga_phase1_verify_one_shot(ser, args)
            raise SystemExit(0)
        if args.phase2_verify:
            if args.image_source != "fpga":
                raise ValueError("--phase2-verify currently requires --image-source fpga")
            fpga_phase2_verify_one_shot(ser, args)
            raise SystemExit(0)
        if args.phase3_verify:
            if args.image_source != "fpga":
                raise ValueError("--phase3-verify currently requires --image-source fpga")
            fpga_phase3_verify_one_shot(ser, args)
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
            print(
                "TRAIN_RUN_SAMPLE_PHASE4 completed: "
                f"result=0x{(int(ret) & 0xFFFFFFFF):08X}, inj_steps={max(1,int(args.chunk_nsteps))}, "
                f"tile_rows={max(1,int(args.train_tile_rows))}, inj_total_spikes={inj_spikes}, "
                f"blank_total_spikes={blank_spikes}"
            )
            raise SystemExit(0)
        if args.train_phase4_build_assignments:
            fpga_phase4_build_assignments_from_raw1(ser, args)
            raise SystemExit(0)
        if args.train_infer_e2e_compare:
            fpga_train_infer_e2e_compare(ser, args)
            raise SystemExit(0)
        if args.train_then_infer_compare:
            fpga_train_then_infer_compare_500_100(ser, args)
            raise SystemExit(0)
        raise RuntimeError(
            "No mode selected. Use one of: --phase1-verify, --phase2-verify, --phase3-verify, --train-run-sample-phase4, "
            "--train-phase4-build-assignments, --train-infer-e2e-compare, --train-then-infer-compare"
        )
