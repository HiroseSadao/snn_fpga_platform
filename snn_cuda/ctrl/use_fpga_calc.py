import struct
import sys
import serial

# Communication Parameters
SERIAL_PORTNAME = "COM7"  # CHANGE ME to match your system's serial port name!
BAUD = 115200             # Must match FPGA UART baud
TIMEOUT_SEC = 2.0

# Protocol constants
REQ_SYNC = 0xA5
RESP_SYNC = 0x5A
PROTO_VER = 0x01

OP_ADD_I32 = 0x01

STATUS_OK = 0x00
STATUS_BAD_PACKET = 0xE1
STATUS_UNSUPPORTED_OP = 0xE2


def calc_checksum(payload: bytes) -> int:
    checksum = 0
    for b in payload:
        checksum ^= b
    return checksum & 0xFF


def build_request(opcode: int, args: list[int]) -> bytes:
    payload = bytearray()
    payload.append(PROTO_VER)
    payload.append(opcode & 0xFF)
    payload.append(len(args) & 0xFF)
    for value in args:
        payload.extend(struct.pack("<i", int(value)))

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
    # Response format: [SYNC][STATUS][RESULT_I32_LE][CHECKSUM]
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


def fpga_add(a: int, b: int) -> int:
    req = build_request(OP_ADD_I32, [a, b])

    print(f"Opening serial port {SERIAL_PORTNAME}")
    with serial.Serial(SERIAL_PORTNAME, BAUD, timeout=TIMEOUT_SEC) as ser:
        print(f"Sending ADD request to FPGA: {a} + {b}")
        ser.write(req)

        # 1(sync) + 1(status) + 4(result) + 1(checksum)
        resp_raw = read_exact(ser, 7)
        status, result = parse_response(resp_raw)

        if status == STATUS_OK:
            return result
        if status == STATUS_BAD_PACKET:
            raise RuntimeError("FPGA rejected packet: BAD_PACKET")
        if status == STATUS_UNSUPPORTED_OP:
            raise RuntimeError("FPGA rejected packet: UNSUPPORTED_OP")
        raise RuntimeError(f"FPGA returned unknown status: 0x{status:02X}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ctrl/use_fpga_calc.py <int_a> <int_b>")
        sys.exit(1)

    a = int(sys.argv[1])
    b = int(sys.argv[2])
    result = fpga_add(a, b)
    print(f"FPGA result: {a} + {b} = {result}")
