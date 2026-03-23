from pathlib import Path

import numpy as np


SEED = 0
N_INP = 784
N_NEURONS = 50
Q16_SCALE = 1 << 16
MS = 1e-3

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CSR_ROW_PTR_PATH = DATA_DIR / "csr_row_ptr.mem"
CSR_COL_IDX_PATH = DATA_DIR / "csr_col_idx.mem"
CSR_WEIGHT_PATH = DATA_DIR / "csr_weight_q16.mem"
DENSE_WEIGHT_PATH = DATA_DIR / "dense_weight_q16.mem"
DENSE_DELAY_PATH = DATA_DIR / "dense_delay_step.mem"
BRIAN_DELAY_BIN_CANDIDATES = (
    BASE_DIR / "brian2_minimal_results" / "cpp_train" / "results" / "_dynamic_array_inp_exc_delay_4118314961",
    BASE_DIR / "output" / "results" / "_dynamic_array_inp_exc_delay_4118314961",
)


def _read_mem(path: Path) -> list[int]:
    return [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]


def build_dense_brian2_weights() -> np.ndarray:
    np.random.seed(SEED)
    weights = 0.3 * np.random.rand(N_INP, N_NEURONS)
    col_sums = weights.sum(axis=0)
    col_sums[col_sums == 0.0] = 1.0
    weights *= 78.0 / col_sums
    return weights


def build_dense_brian2_delay_steps() -> np.ndarray:
    for path in BRIAN_DELAY_BIN_CANDIDATES:
        if path.exists():
            delays_sec = np.fromfile(path, dtype=np.float64)
            if delays_sec.size != (N_INP * N_NEURONS):
                raise ValueError(f"Unexpected Brian2 delay size in {path}: {delays_sec.size}")
            delay_steps = np.rint(delays_sec / MS).astype(np.int32)
            return delay_steps.reshape(N_INP, N_NEURONS)

    rng = np.random.RandomState(SEED)
    delay_steps = np.rint(rng.rand(N_INP, N_NEURONS) * 10.0).astype(np.int32)
    return delay_steps


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

    dense_weights = build_dense_brian2_weights()
    DENSE_WEIGHT_PATH.write_text(
        "".join(
            f"{int(np.clip(np.rint(dense_weights[pre_idx, post_idx] * Q16_SCALE), 0, Q16_SCALE - 1)):04X}\n"
            for post_idx in range(N_NEURONS)
            for pre_idx in range(N_INP)
        )
    )
    print(f"Wrote {N_INP * N_NEURONS} dense weights to {DENSE_WEIGHT_PATH}")

    dense_delay_steps = build_dense_brian2_delay_steps()
    DENSE_DELAY_PATH.write_text(
        "".join(
            f"{int(dense_delay_steps[pre_idx, post_idx]):X}\n"
            for post_idx in range(N_NEURONS)
            for pre_idx in range(N_INP)
        )
    )
    print(f"Wrote {N_INP * N_NEURONS} dense delays to {DENSE_DELAY_PATH}")


if __name__ == "__main__":
    main()
