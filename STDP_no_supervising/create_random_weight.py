# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


FP_SHIFT = 16
FP_SCALE = 1 << FP_SHIFT
INIT_W_SCALE = 1e-3
INIT_VAL_FP = int(round(INIT_W_SCALE * FP_SCALE))  # 66


def gen_init_weights(n_neurons: int, n_in: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    w = np.random.rand(n_neurons, n_in) * INIT_W_SCALE
    w_fp = np.rint(w * FP_SCALE).astype(np.int32)
    return w_fp


def write_mem_files(out_dir: Path, w_fp: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    n_neurons, n_in = w_fp.shape
    neuron_groups = (n_neurons + 3) // 4
    depth = neuron_groups * n_in

    banks = [[INIT_VAL_FP for _ in range(depth)] for _ in range(4)]
    for row in range(neuron_groups):
        for in_idx in range(n_in):
            base = row * n_in + in_idx
            for bank in range(4):
                neuron = row * 4 + bank
                if neuron < n_neurons:
                    banks[bank][base] = int(w_fp[neuron, in_idx])

    for bank in range(4):
        path = out_dir / f"w_init{bank}.mem"
        with path.open("w", encoding="utf-8") as f:
            for v in banks[bank]:
                if v < 0:
                    v = (v + (1 << 32)) & 0xFFFFFFFF
                f.write(f"{v:08x}\n")

    sum_abs = np.sum(np.abs(w_fp), axis=1).astype(np.int64)
    path = out_dir / "sum_abs.mem"
    with path.open("w", encoding="utf-8") as f:
        for v in sum_abs:
            if v < 0:
                v = (v + (1 << 32)) & 0xFFFFFFFF
            f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random STDP weights as .mem files.")
    parser.add_argument("--n-in", type=int, default=784)
    parser.add_argument("--n-neurons", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    args = parser.parse_args()

    w_fp = gen_init_weights(args.n_neurons, args.n_in, args.seed)
    write_mem_files(args.out_dir, w_fp)


if __name__ == "__main__":
    main()
