import numpy as np
import matplotlib.pyplot as plt
import struct
from pathlib import Path
import argparse


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


def write_raw_image_file(out_path, num_images=10000, threshold=0.5):
    x_train, y_train = load_mnist()
    x_train = x_train.astype(np.float32)
    if x_train.max() > 1.0:
        x_train = x_train / 255.0

    num_images = min(num_images, x_train.shape[0])
    images = x_train[:num_images].reshape(num_images, 784)
    labels = np.array(y_train[:num_images], dtype=np.uint8)

    # 784-bit image per sample: binarize and pack bits (8 pixels/byte) => 98 bytes/image.
    images_bin = (images >= threshold).astype(np.uint8)
    images_packed = np.packbits(images_bin, axis=1, bitorder="little")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Header (little-endian):
    # magic(4) = b"RAW1"
    # version(4) = 1
    # num_images(4)
    # n_bits(4) = 784
    # bytes_per_image(4) = 98
    header = struct.pack("<4sIIII", b"RAW1", 1, num_images, 784, 98)

    with out_path.open("wb") as f:
        f.write(header)
        f.write(labels.tobytes(order="C"))
        f.write(images_packed.tobytes(order="C"))

    return out_path, labels, images_packed.shape[1]


def write_raw_to_physical_drive(
    bin_path,
    physical_drive_number,
    lba_start=2048,
    chunk_size=64 * 1024,
    log_interval_bytes=64 * 1024 * 1024,
):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST raw-image (784-bit) generator and SD writer")
    parser.add_argument("--num-images", type=int, default=10000)
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

    first_bin = (x_train[0].reshape(784) >= args.threshold).astype(np.uint8).reshape(28, 28)

    if not args.no_plot:
        plt.imshow(first_bin, cmap="gray")
        plt.title(f"label={y_train[0]}")
        plt.show()

    out_path, labels, bytes_per_image = write_raw_image_file(
        out_path=args.out,
        num_images=args.num_images,
        threshold=args.threshold,
    )
    print(
        f"Wrote {out_path} "
        f"(labels={labels[:10].tolist()}, bytes_per_image={bytes_per_image}, threshold={args.threshold})"
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
