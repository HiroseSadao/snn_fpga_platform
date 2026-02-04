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


def online_load_and_encoding_dataset(dataset, i, dt, n_time, max_fr=32, norm=140):
    img = dataset[i].reshape(-1)
    fr_tmp = max_fr * norm / np.sum(img)
    fr = fr_tmp * np.repeat(np.expand_dims(img, axis=0), n_time, axis=0)
    input_spikes = np.where(np.random.rand(n_time, 784) < fr * dt, 1, 0)
    input_spikes = input_spikes.astype(np.uint8)
    return input_spikes


def write_spike_file(
    out_path,
    num_images=4,
    seed=0,
    dt=0.001,
    t_inj=0.35,
):
    x_train, y_train = load_mnist()
    x_train = x_train.astype(np.float32)

    n_time = round(t_inj / dt)
    rng = np.random.default_rng(seed)

    spikes_list = []
    labels = []
    for i in range(num_images):
        # use a deterministic random stream per sample
        np.random.seed(rng.integers(0, 2**31 - 1, dtype=np.int64))
        spikes = online_load_and_encoding_dataset(
            dataset=x_train, i=i, dt=dt, n_time=n_time
        )
        spikes_list.append(spikes)
        labels.append(int(y_train[i]))

    spikes_all = np.stack(spikes_list, axis=0)  # [N, T, 784]
    labels = np.array(labels, dtype=np.uint8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Header (little-endian):
    # magic(4) = b"SPK1"
    # version(4) = 1
    # num_images(4)
    # n_time(4)
    # n_neurons(4) = 784
    header = struct.pack("<4sIIII", b"SPK1", 1, num_images, n_time, 784)

    with out_path.open("wb") as f:
        f.write(header)
        f.write(labels.tobytes(order="C"))
        f.write(spikes_all.tobytes(order="C"))

    return out_path, labels, n_time


def write_raw_to_physical_drive(bin_path, physical_drive_number, lba_start=2048):
    bin_path = Path(bin_path)
    data = bin_path.read_bytes()

    if len(data) % 512 != 0:
        pad_len = 512 - (len(data) % 512)
        data += b"\x00" * pad_len

    device_path = rf"\\.\PhysicalDrive{physical_drive_number}"
    byte_offset = lba_start * 512

    with open(device_path, "r+b") as f:
        f.seek(byte_offset)
        f.write(data)

    return len(data), byte_offset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST spike generator and SD writer")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--t-inj", type=float, default=0.35)
    parser.add_argument("--out", type=str, default="spike_samples.bin")
    parser.add_argument("--write-raw", action="store_true")
    parser.add_argument("--physical-drive", type=int, default=2)
    parser.add_argument("--lba", type=int, default=2048)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    dt = args.dt
    t_inj = args.t_inj
    nt_inj = round(t_inj / dt)

    x_train, y_train = load_mnist()
    x_train = x_train.astype(np.float32)

    input_spikes = online_load_and_encoding_dataset(
        dataset=x_train, i=0, dt=dt, n_time=nt_inj
    )

    if not args.no_plot:
        plt.imshow(
            np.reshape(np.sum(input_spikes, axis=0), (28, 28)),
            cmap="gray",
        )
        plt.title(f"label={y_train[0]}")
        plt.show()

    # Write a small spike data file for SD card testing
    out_path, labels, n_time = write_spike_file(
        out_path=args.out,
        num_images=args.num_images,
        seed=args.seed,
        dt=dt,
        t_inj=t_inj,
    )
    print(f"Wrote {out_path} (labels={labels.tolist()}, n_time={n_time})")

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
