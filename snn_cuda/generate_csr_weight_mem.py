from pathlib import Path

import numpy as np


SEED = 0
N_INP = 784
N_NEURONS = 50
Q16_SCALE = 1 << 16

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSR_ROW_PTR_PATH = DATA_DIR / "csr_row_ptr.mem"
CSR_COL_IDX_PATH = DATA_DIR / "csr_col_idx.mem"
CSR_WEIGHT_PATH = DATA_DIR / "csr_weight_q16.mem"


def _read_mem(path: Path) -> list[int]:
    return [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]


def build_dense_brian2_weights() -> np.ndarray:
    np.random.seed(SEED)
    weights = 0.3 * np.random.rand(N_INP, N_NEURONS)
    col_sums = weights.sum(axis=0)
    col_sums[col_sums == 0.0] = 1.0
    weights *= 78.0 / col_sums
    return weights


def build_sparse_weight_mem() -> list[int]:
    dense_weights = build_dense_brian2_weights()
    row_ptr = _read_mem(CSR_ROW_PTR_PATH)
    col_idx = _read_mem(CSR_COL_IDX_PATH)

    if len(row_ptr) != (N_NEURONS + 1):
        raise ValueError(f"Expected {N_NEURONS + 1} CSR row pointers, got {len(row_ptr)}")
    if row_ptr[-1] != len(col_idx):
        raise ValueError(f"CSR edge count mismatch: row_ptr[-1]={row_ptr[-1]}, len(col_idx)={len(col_idx)}")

    weights_q16: list[int] = []
    for post_idx in range(N_NEURONS):
        start = row_ptr[post_idx]
        end = row_ptr[post_idx + 1]
        for edge_idx in range(start, end):
            pre_idx = col_idx[edge_idx]
            weight = dense_weights[pre_idx, post_idx]
            weight_q16 = int(np.clip(np.rint(weight * Q16_SCALE), 0, Q16_SCALE - 1))
            weights_q16.append(weight_q16)
    return weights_q16


def main() -> None:
    weights_q16 = build_sparse_weight_mem()
    CSR_WEIGHT_PATH.write_text("".join(f"{value:04X}\n" for value in weights_q16))
    print(f"Wrote {len(weights_q16)} weights to {CSR_WEIGHT_PATH}")


if __name__ == "__main__":
    main()
