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

OP_SD_SECTORS_TO_DDR = 0x13
OP_LOAD_IMAGE_FROM_DDR = 0x14
OP_RUN_SAMPLE_INFER = 0x20
OP_READ_SPIKE_COUNT = 0x21
OP_TRAIN_QUERY_CAPS = 0x30
OP_TRAIN_RUN_SAMPLE_PHASE4 = 0x38
OP_TRAIN_LABEL_STATS_RESET = 0x39
OP_TRAIN_LABEL_STATS_ACCUM = 0x3A
OP_READ_TRAIN_LABEL_STAT_SUM = 0x3B
OP_READ_TRAIN_LABEL_STAT_COUNT = 0x3C
OP_BATCH_CONFIG0 = 0x40
OP_BATCH_CONFIG1 = 0x41
OP_BATCH_START = 0x42
OP_BATCH_STATUS = 0x43
OP_BATCH_READ_SUMMARY = 0x44
OP_BATCH_CONFIG2 = 0x45

STATUS_OK = 0x00
STATUS_BAD_PACKET = 0xE1
STATUS_UNSUPPORTED_OP = 0xE2

# Fixed-point/address constants needed for the training DDR map.
FXP_SHIFT = 16
N_IN = 784
N_NEURONS = 50
N_WEIGHTS = N_NEURONS * N_IN
RAW1_HEADER_BYTES = 20
RAW1_NUM_IMAGES_DEFAULT = 10_000
IMGLOAD_SRC_BIAS_BYTES = 0
IMGLOAD_GUARD_SEC = 0.01
IMG_STAGING_MARGIN_WORDS = 262144  # must match top_level.sv IMG_STAGING_BASE_WORD margin

LAST_IO: dict[str, object] = {}


@dataclass(frozen=True)
class BatchStatus:
    phase: int
    error_code: int
    has_error: bool
    done: bool
    active: bool
    cfg1_valid: bool
    cfg0_valid: bool


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
        return (
            "reason=TRAIN_RUN_SAMPLE_PHASE4_GATE(0x38), "
            f"calib={calib}, ddr_pending={ddr_pending}, trace={trace_active}, "
            f"stdp={stdp_active}, stdp_batch={stdp_batch}, chunk={chunk_active}, "
            f"label_stats={label_stats_active}, infer={infer_active}, "
            f"req_nargs={req_nargs_dbg}"
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


def fpga_read_spike_counts(ser: serial.Serial) -> list[int]:
    counts = []
    for neuron_idx in range(N_NEURONS):
        try:
            status, value = send_request(ser, OP_READ_SPIKE_COUNT, [neuron_idx, 0])
        except TimeoutError as exc:
            print(f"READ_SPIKE_COUNT timeout at neuron_idx={neuron_idx}")
            raise exc
        require_ok(status, f"READ_SPIKE_COUNT[{neuron_idx}]")
        counts.append(int(value) & 0xFFFF)
    return counts


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


def fpga_train_query_caps(ser: serial.Serial) -> int:
    status, result = send_request(ser, OP_TRAIN_QUERY_CAPS, [0, 0])
    require_ok(status, "TRAIN_QUERY_CAPS")
    return int(result) & 0xFFFFFFFF


def fpga_batch_config0(
    ser: serial.Serial,
    *,
    mode_train: bool,
    start_sample_idx: int,
) -> None:
    mode_flags = 1 if mode_train else 0
    status, result = send_request(
        ser,
        OP_BATCH_CONFIG0,
        [int(mode_flags), int(start_sample_idx)],
    )
    require_ok(status, "BATCH_CONFIG0")


def fpga_batch_config1(
    ser: serial.Serial,
    *,
    num_samples: int,
    seed_value: int,
) -> None:
    status, result = send_request(
        ser,
        OP_BATCH_CONFIG1,
        [int(num_samples), int(seed_value)],
    )
    require_ok(status, "BATCH_CONFIG1")


def fpga_batch_start(ser: serial.Serial) -> tuple[int, int]:
    return send_request(ser, OP_BATCH_START, [0, 0], response_timeout=max(5.0, TRAIN_KERNEL_TIMEOUT_SEC))


def fpga_batch_config2(
    ser: serial.Serial,
    *,
    start_lba: int,
    reserved: int = 0,
) -> None:
    status, result = send_request(
        ser,
        OP_BATCH_CONFIG2,
        [int(start_lba), int(reserved)],
    )
    require_ok(status, "BATCH_CONFIG2")


def fpga_batch_read_status(ser: serial.Serial) -> BatchStatus:
    status, value = send_request(ser, OP_BATCH_STATUS, [0, 0])
    require_ok(status, "BATCH_STATUS")
    u = int(value) & 0xFFFFFFFF
    return BatchStatus(
        phase=(u >> 19) & 0xFF,
        error_code=(u >> 11) & 0xFF,
        has_error=bool((u >> 10) & 0x1),
        done=bool((u >> 9) & 0x1),
        active=bool((u >> 8) & 0x1),
        cfg1_valid=bool((u >> 7) & 0x1),
        cfg0_valid=bool((u >> 6) & 0x1),
    )


def fpga_batch_read_summary_field(ser: serial.Serial, field_idx: int) -> int:
    status, value = send_request(ser, OP_BATCH_READ_SUMMARY, [int(field_idx), 0])
    require_ok(status, f"BATCH_READ_SUMMARY[{int(field_idx)}]")
    return int(np.int32(value))


def fpga_batch_wait_done(
    ser: serial.Serial,
    *,
    timeout_sec: float,
    poll_interval_sec: float = 0.05,
) -> BatchStatus:
    t0 = time.time()
    while True:
        st = fpga_batch_read_status(ser)
        if st.done:
            return st
        if (time.time() - t0) > float(timeout_sec):
            raise TimeoutError(f"BATCH_STATUS timeout after {float(timeout_sec):.1f}s")
        time.sleep(poll_interval_sec)


def fpga_train_run_sample_phase4(ser: serial.Serial, *, inj_steps: int) -> int:
    timeout_s = max(TRAIN_KERNEL_TIMEOUT_SEC, 300.0)
    status, result = send_request(
        ser,
        OP_TRAIN_RUN_SAMPLE_PHASE4,
        [int(inj_steps), 0],
        response_timeout=timeout_s,
        transient_retry_max=32,
    )
    require_ok(status, "TRAIN_RUN_SAMPLE_PHASE4")
    return int(result)


def fpga_train_then_infer_compare_500_100(
    ser: serial.Serial,
    args: argparse.Namespace,
) -> None:
    """Train on 500 samples, infer on 100 samples, and report FPGA accuracy."""
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
    infer_steps = 350 if mine_timing_mode else max(1, int(args.chunk_nsteps))
    start_lba = int(args.start_lba)
    timeout_sec = float(args.timeout)
    seed = int(args.seed)

    _, labels_all = load_mnist()
    total_need = n_train + n_infer
    if total_need > int(len(labels_all)):
        raise ValueError(f"need {total_need} samples but dataset has {int(len(labels_all))}")

    print(
        "FPGA train/infer start: "
        f"train={n_train}, infer={n_infer}, inj_steps={inj_steps}, infer_steps={infer_steps}"
    )
    if mine_timing_mode:
        print("Train->Infer mode: mine timing override enabled (train inj=350/blank=150, infer steps=350).")

    fpga_train_label_stats_reset(ser)
    pbar_train = tqdm(total=n_train, desc=f"train {n_train}", unit="img", miniters=1, leave=True)
    for sample_idx in range(n_train):
        label = int(labels_all[sample_idx])
        prepare_fpga_sample_image_via_streamed_load(
            ser,
            sample_idx=sample_idx,
            start_lba=start_lba,
            timeout_sec=timeout_sec,
            verbose=False,
        )
        fpga_train_run_sample_phase4(ser, inj_steps=inj_steps)
        fpga_train_label_stats_accum(ser, label)
        pbar_train.update(1)
        if ((sample_idx + 1) % 100) == 0:
            pbar_train.write(f"train checkpoint: {sample_idx + 1}/{n_train} samples completed over UART")
    pbar_train.close()

    fpga_sums, fpga_counts = fpga_read_train_label_stats_all(ser)
    fpga_assign, _ = assign_labels_from_aggregated_stats(fpga_sums, fpga_counts, rates_prev=None, alpha=1.0)

    fpga_correct = 0
    pbar_infer = tqdm(total=n_infer, desc=f"infer {n_infer}", unit="img", miniters=1, leave=True)
    for k in range(n_infer):
        sample_idx = n_train + k
        label = int(labels_all[sample_idx])

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
        if fpga_pred == label:
            fpga_correct += 1
        pbar_infer.update(1)
    pbar_infer.close()

    fpga_acc = float(fpga_correct) / float(n_infer)
    print("FPGA train/infer summary:")
    print(f"  trained samples = {n_train}")
    print(f"  inferred samples = {n_infer}")
    print(f"  correct = {fpga_correct}")
    print(f"  accuracy = {fpga_acc:.4f}")


def prepare_fpga_sample_image_via_streamed_load(
    ser: serial.Serial,
    *,
    sample_idx: int,
    start_lba: int,
    timeout_sec: float,
    verbose: bool = True,
    raw_bin_path: str | None = None,
) -> None:
    if int(sample_idx) < 0:
        raise ValueError(f"sample_idx must be >=0, got {sample_idx}")

    raw1_header_bytes = RAW1_HEADER_BYTES
    raw1_num_images = RAW1_NUM_IMAGES_DEFAULT
    raw1_bytes_per_image = N_IN
    raw_layout_src = "reconstructed(import_MNIST_raw.py)"

    if int(sample_idx) >= int(raw1_num_images):
        raise ValueError(
            f"sample_idx out of range for RAW1 layout: {sample_idx} (max={int(raw1_num_images)-1})"
        )

    img_byte_off = int(raw1_header_bytes) + int(raw1_num_images) + (int(sample_idx) * int(raw1_bytes_per_image))
    img_sector_off = img_byte_off // 512
    img_byte_in_sector = img_byte_off % 512
    img_base_addr_byte = int(img_byte_in_sector)
    sectors_needed = (img_byte_in_sector + IMGLOAD_SRC_BIAS_BYTES + int(raw1_bytes_per_image) + 511) // 512

    if verbose:
        print(
            "Preparing input image via streamed FPGA image load path: "
            f"sample_idx={int(sample_idx)}, sector_off={img_sector_off}, sectors={sectors_needed}, "
            f"byte_in_sector={img_byte_in_sector}, src_bias_bytes={IMGLOAD_SRC_BIAS_BYTES}, "
            f"raw1_num_images={int(raw1_num_images)}, raw_layout_src={raw_layout_src}"
        )

    try:
        fpga_sd_sectors_to_ddr(
            ser=ser,
            start_lba=int(start_lba) + img_sector_off,
            num_sectors=sectors_needed,
            timeout_sec=timeout_sec,
            verbose=verbose,
        )
    except TimeoutError as exc:
        raise TimeoutError(
            f"SD sectors->DDR timeout: sample_idx={int(sample_idx)}, "
            f"start_lba={int(start_lba) + img_sector_off}, sectors={sectors_needed}"
        ) from exc

    if IMGLOAD_GUARD_SEC > 0.0:
        time.sleep(IMGLOAD_GUARD_SEC)

    try:
        fpga_load_image_from_ddr(
            ser=ser,
            base_addr_byte=img_base_addr_byte,
            n_bytes=N_IN,
            timeout_sec=timeout_sec,
            verbose=verbose,
        )
    except TimeoutError as exc:
        raise TimeoutError(
            f"DDR->raw_image0 timeout: sample_idx={int(sample_idx)}, "
            f"base_addr_byte=0x{int(img_base_addr_byte):08X}, n_bytes={N_IN}"
        ) from exc


def fpga_batch_single_smoke(
    ser: serial.Serial,
    *,
    sample_idx: int,
    num_samples: int,
    start_lba: int,
    timeout_sec: float,
    seed: int,
    mode_train: bool,
) -> None:
    fpga_batch_config0(ser, mode_train=mode_train, start_sample_idx=sample_idx)
    fpga_batch_config1(ser, num_samples=num_samples, seed_value=seed)
    fpga_batch_config2(ser, start_lba=start_lba)
    before = fpga_batch_read_status(ser)
    print(f"Batch status before start: {before}")
    start_status, start_result = fpga_batch_start(ser)
    require_ok(start_status, "BATCH_START")
    print(f"Batch start accepted: result=0x{int(start_result) & 0xFFFFFFFF:08X}")
    after = fpga_batch_wait_done(ser, timeout_sec=max(30.0, timeout_sec))
    print(f"Batch status after done: {after}")
    for idx in range(8):
        print(f"Batch summary[{idx}] = {fpga_batch_read_summary_field(ser, idx)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 500 MNIST samples on FPGA, infer 100 samples, and report accuracy."
    )
    parser.add_argument("--image-source", type=str, choices=["fpga"], default="fpga")
    parser.add_argument("--port", type=str, default=SERIAL_PORTNAME)
    parser.add_argument("--start-lba", type=int, default=2048)
    parser.add_argument("--seed", type=lambda x: int(x, 0), default=0x12345678)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--train-then-infer-train-samples", type=int, default=500)
    parser.add_argument("--train-then-infer-infer-samples", type=int, default=100)
    parser.add_argument("--chunk-nsteps", type=int, default=16)
    parser.add_argument(
        "--train-e2e-mine-timing",
        action="store_true",
    )
    parser.add_argument(
        "--batch-control-smoke",
        action="store_true",
        help="Exercise the new batch control-plane opcodes without changing the legacy flow.",
    )
    parser.add_argument(
        "--batch-single-infer-smoke",
        action="store_true",
        help="Load one image, then run one-sample batch infer through the new batch start path.",
    )
    parser.add_argument(
        "--batch-single-train-smoke",
        action="store_true",
        help="Load one image, then run one-sample batch train through the new batch start path.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Sample index used by the batch single-sample smoke modes.",
    )
    parser.add_argument(
        "--batch-num-samples",
        type=int,
        default=1,
        help="Number of samples for the batch smoke modes.",
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
        try:
            caps = fpga_train_query_caps(ser)
            print(f"Train kernel caps: 0x{caps:08X}")
        except Exception as exc:
            raise RuntimeError(f"TRAIN_QUERY_CAPS failed: {exc}") from exc

        if args.batch_control_smoke:
            fpga_batch_config0(ser, mode_train=True, start_sample_idx=0)
            fpga_batch_config1(ser, num_samples=1, seed_value=int(args.seed))
            batch_status = fpga_batch_read_status(ser)
            print(f"Batch status before start: {batch_status}")
            start_status, start_result = fpga_batch_start(ser)
            print(
                "Batch start response: "
                f"status=0x{int(start_status):02X}, result=0x{int(start_result) & 0xFFFFFFFF:08X}"
            )
            batch_status = fpga_batch_read_status(ser)
            print(f"Batch status after start: {batch_status}")
            for idx in range(4):
                print(f"Batch summary[{idx}] = {fpga_batch_read_summary_field(ser, idx)}")
            sys.exit(0)

        if args.batch_single_infer_smoke:
            fpga_batch_single_smoke(
                ser,
                sample_idx=int(args.sample_idx),
                num_samples=int(args.batch_num_samples),
                start_lba=int(args.start_lba),
                timeout_sec=float(args.timeout),
                seed=int(args.seed),
                mode_train=False,
            )
            sys.exit(0)

        if args.batch_single_train_smoke:
            fpga_batch_single_smoke(
                ser,
                sample_idx=int(args.sample_idx),
                num_samples=int(args.batch_num_samples),
                start_lba=int(args.start_lba),
                timeout_sec=float(args.timeout),
                seed=int(args.seed),
                mode_train=True,
            )
            sys.exit(0)

        fpga_train_then_infer_compare_500_100(ser, args)

