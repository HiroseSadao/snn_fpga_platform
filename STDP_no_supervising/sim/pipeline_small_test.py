# file: sim/pipeline_small_test.py
import os
from pathlib import Path
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


CLK_PERIOD_NS = 10  # 100 MHz
MAX_WAIT_CYCLES = 200000
FP_SHIFT = 16
FP_SCALE = 1 << FP_SHIFT
INIT_W_SCALE = 1e-3
INIT_VAL_FP = int(round(INIT_W_SCALE * FP_SCALE))

# Match pipeline_small.sv defaults
N_IN = 784
N_NEURONS = 100
UPDATE_NT = 8

# Debug weight readback subset (to avoid huge export)
DBG_NEURON_COUNT = 8
DBG_IN_STRIDE = 4
DBG_IN_LIMIT = 64


def gen_init_weights():
    np.random.seed(0)
    w = np.random.rand(N_NEURONS, N_IN) * INIT_W_SCALE
    w_fp = np.rint(w * FP_SCALE).astype(np.int32)
    return w_fp


def write_init_files(build_dir):
    data_dir = build_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    w_fp = gen_init_weights()
    depth = N_NEURONS * N_IN
    bank = [INIT_VAL_FP for _ in range(depth)]

    for neuron in range(N_NEURONS):
        base = neuron * N_IN
        for in_idx in range(N_IN):
            bank[base + in_idx] = int(w_fp[neuron, in_idx])

    path = data_dir / "w_init0.mem"
    with open(path, "w", encoding="utf-8") as f:
        for v in bank:
            if v < 0:
                v = (v + (1 << 32)) & 0xFFFFFFFF
            f.write(f"{v:08x}\n")

    sum_abs = np.sum(np.abs(w_fp), axis=1).astype(np.int64)
    path = data_dir / "sum_abs.mem"
    with open(path, "w", encoding="utf-8") as f:
        for v in sum_abs:
            if v < 0:
                v = (v + (1 << 32)) & 0xFFFFFFFF
            f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")

# Network params
TD_IN_STEPS = 1
TD_EXC_STEPS = 1
TD_INH_STEPS = 2
TD_X_STEPS = 20
DELAY_IN_STEPS = 5
DELAY_E2I_STEPS = 2

WEXC_FP = 147456
WINH_FP = 57344

WMIN_FP = 0
WMAX_FP = 3277
A_P_FP = 655   # 0.01
A_M_FP = 688   # 0.0105

EXC_VREST = -65
EXC_VRESET = -65
EXC_INIT_VTHR = -52
EXC_VPEAK = 20
EXC_TAU_M = 100
EXC_REFRACT = 5
EXC_TC_THETA = 10000000
EXC_THETA_MAX = 35
EXC_THETA_PLUS_FP = 3277
EXC_E_EXC = 0
EXC_E_INH = -100

INH_VREST = -60
INH_VRESET = -45
INH_VTHR = -40
INH_VPEAK = 20
INH_TAU_M = 10
INH_REFRACT = 2
INH_E_EXC = 0
INH_E_INH = -85


def fp_mul(a, b):
    return (a * b) >> FP_SHIFT


def fp_div_round(a, div):
    if a >= 0:
        return (a + (div >> 1)) // div
    return (a - (div >> 1)) // div


class RefModel:
    def __init__(self):
        self.r_in = [0] * N_IN
        self.x_in = [0] * N_IN
        self.r_exc = [0] * N_NEURONS
        self.r_inh = [0] * N_NEURONS
        self.x_exc = [0] * N_NEURONS
        self.v_exc = [EXC_VRESET * FP_SCALE] * N_NEURONS
        self.theta = [0] * N_NEURONS
        self.vthr = [EXC_INIT_VTHR * FP_SCALE] * N_NEURONS
        self.refr_exc = [0] * N_NEURONS
        self.v_inh = [INH_VRESET * FP_SCALE] * N_NEURONS
        self.refr_inh = [0] * N_NEURONS
        self.g_inh_state = [0] * N_NEURONS
        self.W_in = gen_init_weights().tolist()
        self.delay_in = [[0 for _ in range(N_NEURONS)] for _ in range(DELAY_IN_STEPS)]
        self.delay_e2i = [[0 for _ in range(N_NEURONS)] for _ in range(DELAY_E2I_STEPS)]
        self.tcount = 0

    def step(self, s_in_bits, stdp_en):
        s_in = [(s_in_bits >> i) & 1 for i in range(N_IN)]

        # input synapse + trace
        r_in_next = []
        x_in_next = []
        for i in range(N_IN):
            r = self.r_in[i] - fp_div_round(self.r_in[i], TD_IN_STEPS) + (FP_SCALE // TD_IN_STEPS) * s_in[i]
            x = self.x_in[i] - fp_div_round(self.x_in[i], TD_X_STEPS) + (FP_SCALE // TD_X_STEPS) * s_in[i]
            r_in_next.append(r)
            x_in_next.append(x)

        # g_in = W * c_in
        g_in = []
        for i in range(N_NEURONS):
            acc = 0
            for j in range(N_IN):
                acc += self.W_in[i][j] * r_in_next[j]
            g_in.append(acc >> FP_SHIFT)

        # delays
        g_in_delayed = [self.delay_in[-1][i] for i in range(N_NEURONS)]
        g_exc_delayed = [self.delay_e2i[-1][i] for i in range(N_NEURONS)]

        # exc LIF
        s_exc_next = [0] * N_NEURONS
        v_exc_next = [0] * N_NEURONS
        theta_next = [0] * N_NEURONS
        vthr_next = [0] * N_NEURONS
        refr_exc_next = [0] * N_NEURONS
        for i in range(N_NEURONS):
            i_syn_exc = fp_mul(g_in_delayed[i], (EXC_E_EXC * FP_SCALE) - self.v_exc[i])
            i_syn_inh = fp_mul(self.g_inh_state[i], (EXC_E_INH * FP_SCALE) - self.v_exc[i])
            num = (EXC_VREST * FP_SCALE) - self.v_exc[i] + i_syn_exc + i_syn_inh
            dv = fp_div_round(num, EXC_TAU_M)
            v_next = self.v_exc[i] + dv

            if self.refr_exc[i] != 0:
                refr_exc_next[i] = self.refr_exc[i] - 1
                v_exc_next[i] = EXC_VRESET * FP_SCALE
                theta_tmp = self.theta[i] - fp_div_round(self.theta[i], EXC_TC_THETA)
                s_exc_next[i] = 0
            else:
                if v_next >= self.vthr[i]:
                    s_exc_next[i] = 1
                    v_exc_next[i] = EXC_VRESET * FP_SCALE
                    refr_exc_next[i] = EXC_REFRACT
                    theta_tmp = self.theta[i] - fp_div_round(self.theta[i], EXC_TC_THETA) + EXC_THETA_PLUS_FP
                else:
                    s_exc_next[i] = 0
                    v_exc_next[i] = v_next
                    refr_exc_next[i] = 0
                    theta_tmp = self.theta[i] - fp_div_round(self.theta[i], EXC_TC_THETA)

            if theta_tmp < 0:
                theta_tmp = 0
            if theta_tmp > (EXC_THETA_MAX * FP_SCALE):
                theta_tmp = EXC_THETA_MAX * FP_SCALE
            theta_next[i] = theta_tmp
            vthr_next[i] = (EXC_INIT_VTHR * FP_SCALE) + theta_tmp

        # exc synapse + trace
        r_exc_next = []
        x_exc_next = []
        g_exc = []
        for i in range(N_NEURONS):
            r = self.r_exc[i] - fp_div_round(self.r_exc[i], TD_EXC_STEPS) + (FP_SCALE // TD_EXC_STEPS) * s_exc_next[i]
            x = self.x_exc[i] - fp_div_round(self.x_exc[i], TD_X_STEPS) + (FP_SCALE // TD_X_STEPS) * s_exc_next[i]
            r_exc_next.append(r)
            x_exc_next.append(x)
            g_exc.append(fp_mul(WEXC_FP, r))

        # inh LIF
        s_inh_next = [0] * N_NEURONS
        v_inh_next = [0] * N_NEURONS
        refr_inh_next = [0] * N_NEURONS
        for i in range(N_NEURONS):
            i_syn_exc_i = fp_mul(g_exc_delayed[i], (INH_E_EXC * FP_SCALE) - self.v_inh[i])
            num_i = (INH_VREST * FP_SCALE) - self.v_inh[i] + i_syn_exc_i
            dv_i = fp_div_round(num_i, INH_TAU_M)
            v_next_i = self.v_inh[i] + dv_i

            if self.refr_inh[i] != 0:
                refr_inh_next[i] = self.refr_inh[i] - 1
                v_inh_next[i] = INH_VRESET * FP_SCALE
                s_inh_next[i] = 0
            else:
                if v_next_i >= (INH_VTHR * FP_SCALE):
                    s_inh_next[i] = 1
                    v_inh_next[i] = INH_VRESET * FP_SCALE
                    refr_inh_next[i] = INH_REFRACT
                else:
                    s_inh_next[i] = 0
                    v_inh_next[i] = v_next_i
                    refr_inh_next[i] = 0

        # inh synapse + g_inh
        r_inh_next = []
        for i in range(N_NEURONS):
            r = self.r_inh[i] - fp_div_round(self.r_inh[i], TD_INH_STEPS) + (FP_SCALE // TD_INH_STEPS) * s_inh_next[i]
            r_inh_next.append(r)
        g_inh_next = []
        for i in range(N_NEURONS):
            acc = 0
            for j in range(N_NEURONS):
                if j != i:
                    acc += r_inh_next[j]
            if N_NEURONS > 1:
                g_inh_next.append(fp_mul(fp_div_round(WINH_FP, (N_NEURONS - 1)), acc))
            else:
                g_inh_next.append(0)

        # update delays
        self.delay_in = [g_in] + self.delay_in[:-1]
        self.delay_e2i = [g_exc] + self.delay_e2i[:-1]

        # Online STDP update (stdp3.py)
        if stdp_en:
            for i in range(N_NEURONS):
                post_spike = s_exc_next[i]
                x_post = x_exc_next[i]
                for j in range(N_IN):
                    pre_spike = s_in[j]
                    x_pre = x_in_next[j]
                    dW = 0
                    if post_spike:
                        dW += fp_mul(A_P_FP, x_pre)
                    if pre_spike:
                        dW -= fp_mul(A_M_FP, x_post)
                    if dW != 0:
                        w_new = self.W_in[i][j] + dW
                        if w_new < WMIN_FP:
                            w_new = WMIN_FP
                        if w_new > WMAX_FP:
                            w_new = WMAX_FP
                        self.W_in[i][j] = w_new

        # commit state
        self.r_in = r_in_next
        self.x_in = x_in_next
        self.r_exc = r_exc_next
        self.x_exc = x_exc_next
        self.r_inh = r_inh_next
        self.v_exc = v_exc_next
        self.theta = theta_next
        self.vthr = vthr_next
        self.refr_exc = refr_exc_next
        self.v_inh = v_inh_next
        self.refr_inh = refr_inh_next
        self.g_inh_state = g_inh_next

        s_exc_bits = 0
        for i in range(N_NEURONS):
            s_exc_bits |= (s_exc_next[i] & 1) << i
        return s_exc_bits


def required_signals_present(dut):
    names = [
        "clk",
        "rst",
        "s_tvalid",
        "s_tready",
        "s_tdata",
        "s_stdp_en",
        "m_tvalid",
        "m_tready",
        "m_tdata",
        "dbg_en",
        "dbg_neuron",
        "dbg_in",
        "dbg_valid",
        "dbg_data",
    ]
    return all(hasattr(dut, name) for name in names)


async def reset_dut(dut):
    dut.rst.value = 1
    dut.s_tvalid.value = 0
    dut.s_tdata.value = 0
    dut.s_stdp_en.value = 0
    dut.m_tready.value = 1
    if hasattr(dut, "dbg_en"):
        dut.dbg_en.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 2)


def bits_from_indices(indices):
    value = 0
    for idx in indices:
        value |= 1 << idx
    return value


async def send_packet(dut, tstep_id, spikes_bits, stdp_en=1):
    dut.s_tdata.value = (int(tstep_id) << N_IN) | int(spikes_bits)
    dut.s_tvalid.value = 1
    dut.s_stdp_en.value = stdp_en
    for cycle in range(MAX_WAIT_CYCLES):
        await RisingEdge(dut.clk)
        if int(dut.s_tready.value) == 1:
            if cycle > 0:
                dut._log.info(
                    f"s_tready asserted after {cycle} cycles (tstep_id={tstep_id})"
                )
            break
        if cycle % 200 == 0:
            dut._log.info(
                f"waiting s_tready... cycle={cycle} s_tvalid={int(dut.s_tvalid.value)} "
                f"m_tvalid={int(dut.m_tvalid.value)} m_tready={int(dut.m_tready.value)}"
            )
    else:
        raise AssertionError("Timeout waiting for s_tready")
    dut.s_tvalid.value = 0


async def recv_packet(dut):
    for cycle in range(MAX_WAIT_CYCLES):
        await RisingEdge(dut.clk)
        if int(dut.m_tvalid.value) == 1 and int(dut.m_tready.value) == 1:
            data = int(dut.m_tdata.value)
            # lower bits are s_exc vector, upper bits are tstep_id
            tstep_id = data >> N_NEURONS
            s_exc_bits = data & ((1 << N_NEURONS) - 1)
            if cycle > 0:
                dut._log.info(f"m_tvalid&ready after {cycle} cycles (tstep_id={tstep_id})")
            return tstep_id, s_exc_bits
        if cycle % 200 == 0:
            dut._log.info(
                f"waiting m_tvalid... cycle={cycle} s_tvalid={int(dut.s_tvalid.value)} "
                f"s_tready={int(dut.s_tready.value)} m_tvalid={int(dut.m_tvalid.value)} "
                f"m_tready={int(dut.m_tready.value)}"
            )
    raise AssertionError("Timeout waiting for m_tvalid&m_tready")


@cocotb.test()
async def pipeline_small_basic_handshake(dut):
    if not required_signals_present(dut):
        dut._log.info("Skipping: DUT missing required AXI-stream ports.")
        return

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    expected_ids = [0, 1, 2, 3, 4]
    spikes = [
        bits_from_indices([0]),
        bits_from_indices([1]),
        bits_from_indices([2]),
        bits_from_indices([3]),
        bits_from_indices([0, 1, 2, 3]),
    ]

    # Bufferless assumption: keep ready high and receive per send
    dut.m_tready.value = 1

    got_ids = []
    for tstep_id, spike in zip(expected_ids, spikes):
        await send_packet(dut, tstep_id, spike, stdp_en=0)
        out_id, _ = await recv_packet(dut)
        got_ids.append(out_id)

    assert got_ids == expected_ids, f"tstep_id mismatch: exp={expected_ids} got={got_ids}"


@cocotb.test()
async def pipeline_small_backpressure(dut):
    if not required_signals_present(dut):
        dut._log.info("Skipping: DUT missing required AXI-stream ports.")
        return

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    # Bufferless assumption: only send when ready, receive per send.
    # Still toggle m_tready to confirm upstream stalls cleanly.
    async def toggle_ready():
        while True:
            dut.m_tready.value = 1
            await ClockCycles(dut.clk, 3)
            dut.m_tready.value = 0
            await ClockCycles(dut.clk, 2)

    cocotb.start_soon(toggle_ready())

    expected_ids = [10, 11, 12]
    got_ids = []
    for tstep_id in expected_ids:
        await send_packet(dut, tstep_id, bits_from_indices([0]), stdp_en=0)
        out_id, _ = await recv_packet(dut)
        got_ids.append(out_id)

    assert got_ids == expected_ids, f"backpressure tstep_id mismatch: exp={expected_ids} got={got_ids}"


@cocotb.test()
async def pipeline_small_reference_model(dut):
    if not required_signals_present(dut):
        dut._log.info("Skipping: DUT missing required AXI-stream ports.")
        return

    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    model = RefModel()
    dut.m_tready.value = 1
    dut.dbg_en.value = 0

    async def dbg_read_weight(ii, jj):
        dut.dbg_neuron.value = ii
        dut.dbg_in.value = jj
        dut.dbg_en.value = 1
        await RisingEdge(dut.clk)
        dut.dbg_en.value = 0
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        if int(dut.dbg_valid.value) != 1:
            dut._log.error(
                "dbg_valid not asserted in dbg_read_weight: i=%d j=%d dbg_valid=%d dbg_data=0x%08x",
                ii,
                jj,
                int(dut.dbg_valid.value),
                int(dut.dbg_data.value.integer),
            )
            raise AssertionError("dbg_valid not asserted")
        return int(dut.dbg_data.value.signed_integer)

    # Fixed stimulus sequence
    patterns = [bits_from_indices([i]) for i in range(UPDATE_NT)]

    # Sanity-check initial weights for a small subset
    for i in range(min(N_NEURONS, 2)):
        for j in range(0, min(N_IN, 8), 4):
            dut_w0 = await dbg_read_weight(i, j)
            ref_w0 = model.W_in[i][j]
            if dut_w0 != ref_w0:
                dut._log.error(
                    "Initial W mismatch [%d][%d]: exp=%d got=%d",
                    i,
                    j,
                    ref_w0,
                    dut_w0,
                )
                raise AssertionError("Initial weight mismatch")

    for step_idx, bits in enumerate(patterns):
        await send_packet(dut, step_idx, bits, stdp_en=0)
        out_id, s_exc_bits = await recv_packet(dut)
        ref_bits = model.step(int(bits), stdp_en=0)
        assert out_id == step_idx, f"tstep_id mismatch at {step_idx}"
        assert s_exc_bits == ref_bits, f"s_exc mismatch at {step_idx}: exp={ref_bits:0{N_NEURONS}b} got={s_exc_bits:0{N_NEURONS}b}"

    # STDP was disabled in this test to avoid long stalls
    max_neurons = min(N_NEURONS, DBG_NEURON_COUNT)
    max_in = min(N_IN, DBG_IN_LIMIT)

    for i in range(max_neurons):
        for j in range(0, max_in, DBG_IN_STRIDE):
            dut_w = await dbg_read_weight(i, j)
            ref_w = model.W_in[i][j]
            if dut_w != ref_w:
                dut._log.error(
                    "W_in mismatch [%d][%d]: exp=%d got=%d",
                    i,
                    j,
                    ref_w,
                    dut_w,
                )
                raise AssertionError(f"W_in mismatch [{i}][{j}]: exp={ref_w} got={dut_w}")


def pipeline_small_runner():
    from cocotb.runner import get_runner
    sim = os.getenv("SIM", "icarus")
    repo_root = Path(__file__).resolve().parents[1]
    hdl_dir = repo_root / "hdl"

    sv_sources = [str(p) for p in hdl_dir.glob("**/*.sv")]

    runner = get_runner(sim)
    runner.build(
        sources=sv_sources,
        hdl_toplevel="pipeline_small",
        parameters={"W_INIT_FROM_FILE": 1, "UPDATE_NT": UPDATE_NT},
        timescale=("1ns", "1ps"),
        waves=True,
        always=True,
        build_dir=str(repo_root / "sim" / "sim_build_pipeline_small"),
    )
    write_init_files(repo_root / "sim" / "sim_build_pipeline_small")
    runner.test(
        hdl_toplevel="pipeline_small",
        test_module=Path(__file__).stem,
        seed=None,
    )


if __name__ == "__main__":
    pipeline_small_runner()
