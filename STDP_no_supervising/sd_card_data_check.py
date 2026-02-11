import argparse
import os
import sys
import struct


def hexdump(data: bytes, width: int = 16) -> str:
    lines = []
    for i in range(0, len(data), width):
        chunk = data[i : i + width]
        hex_part = " ".join(f"{b:02X}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
        lines.append(f"{i:08X}  {hex_part:<{width*3}}  {ascii_part}")
    return "\n".join(lines)


def read_physical_drive(drive_num: int, offset: int, length: int) -> bytes:
    path = rf"\\.\PhysicalDrive{drive_num}"
    with open(path, "rb", buffering=0) as f:
        if offset:
            f.seek(offset, os.SEEK_SET)
        return f.read(length)


def read_spk1_header(drive_num: int, lba_start: int, sector_size: int = 512):
    header_size = struct.calcsize("<4sIIII")
    byte_offset = lba_start * sector_size
    # Read a full sector to satisfy Windows raw disk alignment requirements.
    data = read_physical_drive(drive_num, byte_offset, sector_size)
    if len(data) < header_size:
        raise RuntimeError("Failed to read full header.")
    magic, version, num_images, n_time, n_neurons = struct.unpack(
        "<4sIIII", data[:header_size]
    )
    if magic != b"SPK1":
        raise RuntimeError(f"Unexpected magic {magic!r} at LBA {lba_start}.")
    return {
        "version": version,
        "num_images": num_images,
        "n_time": n_time,
        "n_neurons": n_neurons,
        "byte_offset": byte_offset,
    }


def read_aligned(
    drive_num: int, offset: int, length: int, sector_size: int = 512
) -> bytes:
    if length <= 0:
        return b""
    aligned_offset = (offset // sector_size) * sector_size
    end = offset + length
    aligned_end = ((end + sector_size - 1) // sector_size) * sector_size
    aligned_len = aligned_end - aligned_offset
    data = read_physical_drive(drive_num, aligned_offset, aligned_len)
    start = offset - aligned_offset
    return data[start : start + length]


def read_spk1_labels_tail(
    drive_num: int,
    lba_start: int,
    last_count: int,
    sector_size: int = 512,
):
    header_size = struct.calcsize("<4sIIII")
    hdr = read_spk1_header(drive_num, lba_start, sector_size)
    num_images = hdr["num_images"]
    if last_count <= 0:
        return hdr, []
    tail_count = min(last_count, num_images)
    labels_offset = hdr["byte_offset"] + header_size
    tail_offset = labels_offset + (num_images - tail_count)
    tail_bytes = read_aligned(
        drive_num, tail_offset, tail_count, sector_size=sector_size
    )
    if len(tail_bytes) != tail_count:
        raise RuntimeError("Failed to read full label tail.")
    return hdr, list(tail_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read raw data from a Windows physical drive (read-only)."
    )
    parser.add_argument(
        "--drive",
        type=int,
        default=2,
        help="Physical drive number (default: 2 -> \\\\.\\PhysicalDrive2)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Byte offset to start reading from (default: 0)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=512,
        help="Number of bytes to read (default: 512)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output file path to save raw bytes (optional).",
    )
    parser.add_argument(
        "--print-count",
        action="store_true",
        help="Print stored image count by reading SPK1 header at LBA.",
    )
    parser.add_argument(
        "--lba",
        type=int,
        default=2048,
        help="LBA start for SPK1 header (default: 2048).",
    )
    parser.add_argument(
        "--sector-size",
        type=int,
        default=512,
        help="Sector size in bytes (default: 512).",
    )
    parser.add_argument(
        "--print-last-labels",
        action="store_true",
        help="Print last N labels from SPK1 header/labels at LBA.",
    )
    parser.add_argument(
        "--last-count",
        type=int,
        default=1000,
        help="Number of last labels to print (default: 1000).",
    )
    args = parser.parse_args()

    if args.length <= 0:
        print("length must be positive", file=sys.stderr)
        return 2
    if args.offset < 0:
        print("offset must be >= 0", file=sys.stderr)
        return 2
    if args.drive < 0:
        print("drive must be >= 0", file=sys.stderr)
        return 2
    if args.lba < 0:
        print("lba must be >= 0", file=sys.stderr)
        return 2
    if args.sector_size <= 0:
        print("sector_size must be positive", file=sys.stderr)
        return 2
    if args.last_count < 0:
        print("last_count must be >= 0", file=sys.stderr)
        return 2

    try:
        if args.print_last_labels:
            hdr, labels = read_spk1_labels_tail(
                drive_num=args.drive,
                lba_start=args.lba,
                last_count=args.last_count,
                sector_size=args.sector_size,
            )
            print(
                "Images stored (from raw header at LBA {lba}): {num_images}".format(
                    lba=args.lba, num_images=hdr["num_images"]
                )
            )
            print(f"Last {len(labels)} labels:")
            print(labels)
            return 0
        if args.print_count:
            hdr = read_spk1_header(
                drive_num=args.drive,
                lba_start=args.lba,
                sector_size=args.sector_size,
            )
            print(
                "Images stored (from raw header at LBA {lba}): {num_images}".format(
                    lba=args.lba, num_images=hdr["num_images"]
                )
            )
            return 0
        data = read_physical_drive(args.drive, args.offset, args.length)
    except PermissionError:
        print(
            "Permission denied. Run this script from an elevated (Administrator) shell.",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError:
        print(
            f"PhysicalDrive{args.drive} not found. Check the drive number.",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(f"Failed to read from PhysicalDrive{args.drive}: {exc}", file=sys.stderr)
        return 1

    if args.out:
        with open(args.out, "wb") as out_f:
            out_f.write(data)
        print(f"Wrote {len(data)} bytes to {args.out}")
    else:
        print(hexdump(data))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
