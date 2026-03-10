import argparse
import gzip
import struct
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MNIST_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)
MNIST_CACHE_DIR = Path("bram_snn_cuda/mnist_data")
N_FEATURES = 28 * 28
RAW1_MAGIC = b"RAW1"
RAW2_MAGIC = b"RAW2"
RAW_VERSION = 1


def _download_mnist_file(filename: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / filename
    if out_path.exists():
        return out_path
    last_error: Exception | None = None
    for base_url in MNIST_MIRRORS:
        try:
            urllib.request.urlretrieve(base_url + filename, out_path)
            return out_path
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to download MNIST file: {filename}") from last_error


def _load_idx_images(gz_path: Path) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise RuntimeError(f"Invalid MNIST image file magic in {gz_path}: {magic}")
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _load_idx_labels(gz_path: Path) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise RuntimeError(f"Invalid MNIST label file magic in {gz_path}: {magic}")
        n = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n)


def load_mnist(cache_dir: str | Path = MNIST_CACHE_DIR) -> tuple[np.ndarray, np.ndarray]:
    cache_path = Path(cache_dir)
    images_path = _download_mnist_file("train-images-idx3-ubyte.gz", cache_path)
    labels_path = _download_mnist_file("train-labels-idx1-ubyte.gz", cache_path)
    return _load_idx_images(images_path), _load_idx_labels(labels_path)


def _prepare_dense_u8(images: np.ndarray) -> tuple[np.ndarray, int]:
    payload = np.clip(np.rint(images * 255.0), 0, 255).astype(np.uint8)
    return payload, N_FEATURES


def _prepare_bin1(images: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    images_bin = (images >= threshold).astype(np.uint8)
    payload = np.packbits(images_bin, axis=1, bitorder="little")
    return payload, payload.shape[1]


def _prepare_sparse_u8(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    payload_chunks: list[bytes] = []
    payload_size = 0
    for image in images:
        nz_idx = np.flatnonzero(image)
        header = struct.pack("<H", int(nz_idx.size))
        payload_chunks.append(header)
        payload_size += len(header)
        if nz_idx.size:
            nz_val = image[nz_idx]
            interleaved = bytearray()
            for idx, val in zip(nz_idx.tolist(), nz_val.tolist()):
                interleaved.extend(struct.pack("<HB", int(idx), int(val)))
            entries = bytes(interleaved)
            payload_chunks.append(entries)
            payload_size += len(entries)
        offsets.append(payload_size)

    payload = b"".join(payload_chunks)
    return np.asarray(offsets, dtype=np.uint32), np.frombuffer(payload, dtype=np.uint8)


def write_raw_image_file(
    out_path: str | Path,
    *,
    num_images: int = 10000,
    fmt: str = "u8",
    threshold: float = 0.5,
) -> tuple[Path, np.ndarray, int]:
    x_train, y_train = load_mnist()
    x_train = x_train.astype(np.float32)
    if x_train.max() > 1.0:
        x_train = x_train / 255.0

    num_images = min(num_images, x_train.shape[0])
    images = x_train[:num_images].reshape(num_images, N_FEATURES)
    labels = np.asarray(y_train[:num_images], dtype=np.uint8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "u8":
        payload, bytes_per_image = _prepare_dense_u8(images)
        header = struct.pack("<4sIIII", RAW1_MAGIC, RAW_VERSION, num_images, N_FEATURES, bytes_per_image)
        with out_path.open("wb") as f:
            f.write(header)
            f.write(labels.tobytes(order="C"))
            f.write(payload.tobytes(order="C"))
        return out_path, labels, bytes_per_image

    if fmt == "bin1":
        payload, bytes_per_image = _prepare_bin1(images, threshold)
        header = struct.pack("<4sIIII", RAW1_MAGIC, RAW_VERSION, num_images, N_FEATURES, bytes_per_image)
        with out_path.open("wb") as f:
            f.write(header)
            f.write(labels.tobytes(order="C"))
            f.write(payload.tobytes(order="C"))
        return out_path, labels, bytes_per_image

    if fmt == "sparse_u8":
        dense_u8, _ = _prepare_dense_u8(images)
        offsets, payload = _prepare_sparse_u8(dense_u8)
        header = struct.pack(
            "<4sIIIII",
            RAW2_MAGIC,
            RAW_VERSION,
            num_images,
            N_FEATURES,
            3,  # bytes per sparse entry: u16 idx + u8 value
            2,  # offset entry bytes / semantic tag for sparse_u8
        )
        with out_path.open("wb") as f:
            f.write(header)
            f.write(labels.tobytes(order="C"))
            f.write(offsets.tobytes(order="C"))
            f.write(payload.tobytes(order="C"))
        return out_path, labels, -1

    raise ValueError(f"Unsupported format: {fmt}")


def write_raw_to_physical_drive(
    bin_path: str | Path,
    physical_drive_number: int,
    *,
    lba_start: int = 2048,
    chunk_size: int = 64 * 1024,
    log_interval_bytes: int = 64 * 1024 * 1024,
) -> tuple[int, int]:
    bin_path = Path(bin_path)
    total_size = bin_path.stat().st_size

    device_path = rf"\\.\PhysicalDrive{physical_drive_number}"
    byte_offset = lba_start * 512

    total_written = 0
    chunk_index = 0
    next_log = log_interval_bytes if log_interval_bytes > 0 else None
    print(
        f"Writing {total_size} bytes to {device_path} at byte offset {byte_offset} "
        f"(chunk_size={chunk_size})"
    )
    with open(bin_path, "rb") as src, open(device_path, "r+b") as dst:
        dst.seek(byte_offset)
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            remaining = total_size - total_written
            if remaining <= len(chunk) and (len(chunk) % 512) != 0:
                pad_len = 512 - (len(chunk) % 512)
                chunk += b"\x00" * pad_len
            try:
                dst.write(chunk)
            except OSError as exc:
                current_offset = byte_offset + total_written
                print(
                    "Write failed.",
                    f"errno={getattr(exc, 'errno', None)}",
                    f"chunk_index={chunk_index}",
                    f"chunk_size={len(chunk)}",
                    f"total_written={total_written}",
                    f"byte_offset={current_offset}",
                    sep=" ",
                )
                raise
            total_written += len(chunk)
            chunk_index += 1
            if next_log is not None and total_written >= next_log:
                print(f"Wrote {total_written} / {total_size} bytes...")
                next_log += log_interval_bytes

        if total_written % 512 != 0:
            pad_len = 512 - (total_written % 512)
            dst.write(b"\x00" * pad_len)
            total_written += pad_len
            print(f"Padded {pad_len} bytes to 512-byte boundary.")

        dst.flush()

    return total_written, byte_offset


def _preview_image(fmt: str, images: np.ndarray, threshold: float) -> np.ndarray:
    if fmt == "u8":
        return np.clip(np.rint(images[0] * 255.0), 0, 255).astype(np.uint8)
    if fmt == "bin1":
        return (images[0].reshape(N_FEATURES) >= threshold).astype(np.uint8).reshape(28, 28) * 255
    if fmt == "sparse_u8":
        return np.clip(np.rint(images[0] * 255.0), 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported preview format: {fmt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST raw-image generator and SD writer")
    parser.add_argument("--num-images", type=int, default=10000)
    parser.add_argument("--format", type=str, choices=["u8", "bin1", "sparse_u8"], default="u8")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="raw_samples.bin")
    parser.add_argument("--write-raw", action="store_true")
    parser.add_argument("--physical-drive", type=int, default=2)
    parser.add_argument("--lba", type=int, default=2048)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    x_train, y_train = load_mnist()
    x_train = x_train.astype(np.float32)
    if x_train.max() > 1.0:
        x_train = x_train / 255.0

    if not args.no_plot:
        first_img = _preview_image(args.format, x_train, args.threshold)
        plt.imshow(first_img, cmap="gray", vmin=0, vmax=255)
        plt.title(f"label={y_train[0]}")
        plt.show()

    out_path, labels, bytes_per_image = write_raw_image_file(
        out_path=args.out,
        num_images=args.num_images,
        fmt=args.format,
        threshold=args.threshold,
    )
    if args.format == "sparse_u8":
        print(
            f"Wrote {out_path} "
            f"(labels={labels[:10].tolist()}, format={args.format}, variable_length_records=True)"
        )
    else:
        print(
            f"Wrote {out_path} "
            f"(labels={labels[:10].tolist()}, format={args.format}, "
            f"bytes_per_image={bytes_per_image}, threshold={args.threshold})"
        )

    if args.write_raw:
        total_bytes, offset = write_raw_to_physical_drive(
            bin_path=out_path,
            physical_drive_number=args.physical_drive,
            lba_start=args.lba,
        )
        print(
            f"Wrote {total_bytes} bytes to \\\\.\\PhysicalDrive{args.physical_drive} "
            f"at LBA {args.lba} (byte offset {offset})"
        )
