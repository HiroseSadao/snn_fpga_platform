import struct
import sys
import time

import serial

# Communication parameters
SERIAL_PORTNAME = "COM7"  # CHANGE ME
BAUD = 115200
TIMEOUT_SEC = 2.0
WRITE_TIMEOUT_SEC = 2.0

# Protocol constants
REQ_SYNC = 0xA5
RESP_SYNC = 0x5A
PROTO_VER = 0x01

OP_ADD_I32 = 0x01
OP_SD_TO_DDR_COPY = 0x11
OP_RUN_SAMPLE_INFER = 0x20
OP_READ_SPIKE_COUNT = 0x21

STATUS_OK = 0x00
STATUS_BAD_PACKET = 0xE1
STATUS_UNSUPPORTED_OP = 0xE2

# Fixed-point S16.16 model constants (must match top_level.sv)
FXP_SHIFT = 16
FXP_ALPHA = 62259
FXP_INPUT_W = 8192
FXP_THRESH = 65536
FXP_BIAS_LSB = 512
N_IN = 784
N_NEURONS = 100


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


def send_request(
    ser: serial.Serial,
    opcode: int,
    args: list[int],
    response_timeout: float = TIMEOUT_SEC
) -> tuple[int, int]:
    req = build_request(opcode, args)
    old_timeout = ser.timeout
    ser.timeout = response_timeout
    try:
        ser.reset_input_buffer()
        ser.write(req)
        ser.flush()
        resp_raw = read_exact(ser, 7)
        return parse_response(resp_raw)
    finally:
        ser.timeout = old_timeout


def require_ok(status: int, context: str) -> None:
    if status == STATUS_OK:
        return
    if status == STATUS_BAD_PACKET:
        raise RuntimeError(f"{context}: FPGA rejected packet (BAD_PACKET)")
    if status == STATUS_UNSUPPORTED_OP:
        raise RuntimeError(f"{context}: FPGA rejected packet (UNSUPPORTED_OP)")
    raise RuntimeError(f"{context}: FPGA returned unknown status 0x{status:02X}")


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


def fpga_run_sample_infer(ser: serial.Serial, sample_idx: int, n_steps: int, timeout_sec: float = 120.0) -> int:
    print(f"Requesting FPGA inference: sample={sample_idx}, steps={n_steps}")
    status, total_spikes = send_request(
        ser=ser,
        opcode=OP_RUN_SAMPLE_INFER,
        args=[sample_idx, n_steps],
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


def neuron_bias(neuron_idx: int) -> int:
    return ((neuron_idx & 0x7) + 1) * FXP_BIAS_LSB


def to_s32(v: int) -> int:
    v &= 0xFFFFFFFF
    if v & 0x80000000:
        v -= 0x100000000
    return v


def run_fixed_point_python(bits_784: list[int], n_steps: int) -> list[int]:
    v = [0] * N_NEURONS
    spike_count = [0] * N_NEURONS

    for _ in range(n_steps):
        for n in range(N_NEURONS):
            accum = neuron_bias(n)
            for i in range(N_IN):
                if bits_784[i] and (((i + n) & 0x3) == 0):
                    accum = to_s32(accum + FXP_INPUT_W)

            v_next = to_s32(((to_s32(v[n]) * FXP_ALPHA) >> FXP_SHIFT) + accum)
            if v_next >= FXP_THRESH:
                v[n] = to_s32(v_next - FXP_THRESH)
                spike_count[n] += 1
            else:
                v[n] = v_next

    return spike_count


def compare_counts(fpga_counts: list[int], py_counts: list[int]) -> None:
    diffs = [abs(a - b) for a, b in zip(fpga_counts, py_counts)]
    max_diff = max(diffs)
    mismatch = sum(1 for d in diffs if d != 0)
    print(f"Compare spike_count[100]: mismatched={mismatch}, max_diff={max_diff}")
    if mismatch > 0:
        for i, (f, p, d) in enumerate(zip(fpga_counts, py_counts, diffs)):
            if d != 0:
                print(f"  neuron {i}: fpga={f}, python={p}, diff={d}")
                if i >= 20:
                    break


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
        images = [ds[i][0].numpy().squeeze() for i in range(len(ds))]
        labels = [int(ds[i][1]) for i in range(len(ds))]
        return images, labels
    except Exception:
        pass

    try:
        from tensorflow.keras.datasets import mnist

        (x_train, y_train), _ = mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        return x_train, y_train
    except Exception as exc:
        raise RuntimeError(
            "MNIST loading failed. Install torchvision or tensorflow."
        ) from exc


def first_image_bits_from_mnist(threshold: float = 0.5) -> list[int]:
    images, _ = load_mnist()
    img = images[0].reshape(784)
    bits = [1 if float(v) >= threshold else 0 for v in img]
    return bits


def fpga_add(ser: serial.Serial, a: int, b: int) -> int:
    print(f"Sending ADD request: {a} + {b}")
    status, result = send_request(ser, OP_ADD_I32, [a, b], response_timeout=TIMEOUT_SEC)
    require_ok(status, "ADD request")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python ctrl/use_fpga_calc.py <int_a> <int_b> [start_lba] [num_sectors] [n_steps] [threshold]\n"
            "num_sectors=0 means auto-copy length from RAW1 header."
        )
        sys.exit(1)

    a = int(sys.argv[1])
    b = int(sys.argv[2])
    start_lba = int(sys.argv[3]) if len(sys.argv) >= 4 else 2048
    num_sectors = int(sys.argv[4]) if len(sys.argv) >= 5 else 0
    n_steps = int(sys.argv[5]) if len(sys.argv) >= 6 else 350
    threshold = float(sys.argv[6]) if len(sys.argv) >= 7 else 0.5

    print(f"Opening serial port {SERIAL_PORTNAME}")
    with serial.Serial(
        SERIAL_PORTNAME,
        BAUD,
        timeout=TIMEOUT_SEC,
        write_timeout=WRITE_TIMEOUT_SEC
    ) as ser:
        fpga_sd_to_ddr_copy(
            ser=ser,
            start_lba=start_lba,
            num_sectors=num_sectors,
            timeout_sec=600.0
        )

        fpga_run_sample_infer(ser=ser, sample_idx=0, n_steps=n_steps, timeout_sec=600.0)
        fpga_counts = fpga_read_spike_counts(ser)

        bits = first_image_bits_from_mnist(threshold=threshold)
        py_counts = run_fixed_point_python(bits_784=bits, n_steps=n_steps)
        compare_counts(fpga_counts, py_counts)

        result = fpga_add(ser, a, b)
        print(f"FPGA result: {a} + {b} = {result}")
