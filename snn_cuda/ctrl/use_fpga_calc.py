import argparse
import struct
import time
from pathlib import Path

import numpy as np
import serial

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

STATUS_OK = 0x00
STATUS_BAD_PACKET = 0xE1
STATUS_UNSUPPORTED_OP = 0xE2

# Fixed-point S16.16 model constants (must match top_level.sv)
FXP_SHIFT = 16
FXP_ALPHA = 62259
FXP_INPUT_W = 8192
FXP_THRESH = 65536
FXP_BIAS_LSB = 512
FXP_WTA_INH = 55706
N_IN = 784
N_NEURONS = 100

# Poisson/RNG constants (must match top_level.sv)
POISSON_NUM_CONST = 9175  # floor(32*140*2048*1e-3)
RNG_MAX = 2047
LCG_A = 1664525
LCG_C = 1013904223

LAST_IO: dict[str, object] = {}


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
            resp_raw = read_exact(ser, 7)
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
    spike_count = [0] * N_NEURONS
    rng_state = seed & 0xFFFFFFFF

    for _ in range(n_steps):
        s_in = [0] * N_IN
        for i in range(N_IN):
            rng_state = lcg_next_u32(rng_state)
            rand11 = (rng_state >> 21) & 0x7FF
            s_in[i] = 1 if rand11 < thresholds[i] else 0

        v_next = [0] * N_NEURONS
        any_spike = False
        winner_idx = 0
        winner_v_next = 0

        for n in range(N_NEURONS):
            accum = neuron_bias(n)
            for i in range(N_IN):
                if s_in[i] and (((i + n) & 0x3) == 0):
                    accum = to_s32(accum + FXP_INPUT_W)
            v_n_next = to_s32(((to_s32(v[n]) * FXP_ALPHA) >> FXP_SHIFT) + accum)
            v_next[n] = v_n_next
            if v_n_next >= FXP_THRESH:
                if (not any_spike) or (v_n_next > winner_v_next) or (
                    v_n_next == winner_v_next and n < winner_idx
                ):
                    winner_idx = n
                    winner_v_next = v_n_next
                any_spike = True

        if any_spike:
            for n in range(N_NEURONS):
                if n == winner_idx:
                    v[n] = to_s32(v_next[n] - FXP_THRESH)
                    spike_count[n] += 1
                else:
                    inhibited = to_s32(v_next[n] - FXP_WTA_INH)
                    v[n] = inhibited if inhibited > 0 else 0
        else:
            v = v_next
    return spike_count


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
    parser.add_argument("a", type=int, help="integer add operand A")
    parser.add_argument("b", type=int, help="integer add operand B")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Opening serial port {args.port}")

    with serial.Serial(
        args.port,
        BAUD,
        timeout=TIMEOUT_SEC,
        write_timeout=WRITE_TIMEOUT_SEC
    ) as ser:
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
        print(f"WTA inhibition (S16.16): FXP_WTA_INH={FXP_WTA_INH}")

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
