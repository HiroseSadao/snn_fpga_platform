import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    dt = 0.001
    t_inj = 0.35
    nt_inj = round(t_inj / dt)

    x_train, y_train = load_mnist()
    x_train = x_train.astype(np.float32)

    input_spikes = online_load_and_encoding_dataset(
        dataset=x_train, i=0, dt=dt, n_time=nt_inj
    )

    plt.imshow(
        np.reshape(np.sum(input_spikes, axis=0), (28, 28)),
        cmap="gray",
    )
    plt.title(f"label={y_train[0]}")
    plt.show()
