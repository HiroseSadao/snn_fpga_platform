`default_nettype none

module pipeline_small #(
        parameter int TSTEP_W = 16,
        parameter int N_IN = 784,
        parameter int N_NEURONS = 100,
        parameter int UPDATE_NT = 350,
        parameter int NEURON_W = (N_NEURONS <= 1) ? 1 : $clog2(N_NEURONS),
        parameter int IN_W = (N_IN <= 1) ? 1 : $clog2(N_IN),
        parameter bit W_INIT_FROM_FILE = 0
    )(
        input  wire                         clk,
        input  wire                         rst, // synchronous reset

        input  wire                         s_tvalid,
        output logic                        s_tready,
        input  wire [TSTEP_W+N_IN-1:0]       s_tdata, // {tstep_id, s_in[N_IN-1:0]}
        input  wire                         s_stdp_en,

        output logic                        m_tvalid,
        input  wire                         m_tready,
        output logic [TSTEP_W+N_NEURONS-1:0] m_tdata, // {tstep_id, s_exc[N_NEURONS-1:0]}

        input  wire                         dbg_en,
        input  wire [NEURON_W-1:0]          dbg_neuron,
        input  wire [IN_W-1:0]              dbg_in,
        output logic                        dbg_valid,
        output logic signed [31:0]          dbg_data
    );

    // Fixed-point S16.16 constants
    localparam int FP_SHIFT = 16;
    localparam int FP_SCALE = (1 << FP_SHIFT);

    // Network params (match LIF_WTA_STDP_MNIST.py)
    localparam int TD_IN_STEPS   = 1;   // 1e-3 / 1e-3
    localparam int TD_EXC_STEPS  = 1;   // 1e-3 / 1e-3
    localparam int TD_INH_STEPS  = 2;   // 2e-3 / 1e-3
    localparam int TD_X_STEPS    = 20;  // 2e-2 / 1e-3
    localparam int DELAY_IN_STEPS = 5; // 5e-3 / 1e-3
    localparam int DELAY_E2I_STEPS = 2; // 2e-3 / 1e-3

    localparam int WEXC_FP = 147456; // 2.25
    localparam int WINH_FP = 57344;  // 0.875

    localparam int WMIN_FP = 0;
    localparam int WMAX_FP = 3277;   // 0.05
    localparam int LR_P_FP = 655;    // 1e-2
    localparam int LR_M_FP = 7;      // 1e-4
    localparam int NORM_FP = 6554;   // 0.1
    localparam int DW_CLIP_FP = 66;  // 1e-3

    // Exc LIF params
    localparam int EXC_VREST = -65;
    localparam int EXC_VRESET = -65;
    localparam int EXC_INIT_VTHR = -52;
    localparam int EXC_VPEAK = 20;
    localparam int EXC_TAU_M = 100;
    localparam int EXC_REFRACT = 5;
    localparam int EXC_TC_THETA = 10000000;
    localparam int EXC_THETA_MAX = 35;
    localparam int EXC_THETA_PLUS_FP = 3277; // 0.05
    localparam int EXC_E_EXC = 0;
    localparam int EXC_E_INH = -100;

    // Inh LIF params
    localparam int INH_VREST = -60;
    localparam int INH_VRESET = -45;
    localparam int INH_VTHR = -40;
    localparam int INH_VPEAK = 20;
    localparam int INH_TAU_M = 10;
    localparam int INH_REFRACT = 2;
    localparam int INH_E_EXC = 0;
    localparam int INH_E_INH = -85;

    localparam int LANES = 4;
    localparam int NEURON_GROUPS = (N_NEURONS + LANES - 1) / LANES;

    localparam int INPUT_SPIKE_FP = FP_SCALE / TD_IN_STEPS;
    localparam int TRACE_SPIKE_FP = FP_SCALE / TD_X_STEPS;

    // State
    integer i, j, t;

    logic [TSTEP_W-1:0] tstep_id_reg;
    logic [N_IN-1:0] s_in_reg;
    logic s_stdp_reg;

    logic signed [31:0] r_exc  [0:N_NEURONS-1];
    logic signed [31:0] r_inh  [0:N_NEURONS-1];
    logic signed [31:0] x_in   [0:N_IN-1];
    logic signed [31:0] x_exc  [0:N_NEURONS-1];

    logic signed [31:0] v_exc  [0:N_NEURONS-1];
    logic signed [31:0] theta  [0:N_NEURONS-1];
    logic signed [31:0] vthr   [0:N_NEURONS-1];
    logic [15:0]        refr_exc [0:N_NEURONS-1];

    logic signed [31:0] v_inh  [0:N_NEURONS-1];
    logic [15:0]        refr_inh [0:N_NEURONS-1];

    // g_in is maintained as a state (event-driven update)
    logic signed [31:0] g_in_state [0:N_NEURONS-1];
    logic signed [31:0] g_in_accum [0:N_NEURONS-1];

    logic signed [31:0] g_exc  [0:N_NEURONS-1];
    logic signed [31:0] g_inh_state [0:N_NEURONS-1];

    logic [N_NEURONS-1:0] s_exc;
    logic [N_NEURONS-1:0] s_inh;

    logic signed [31:0] delay_in [0:DELAY_IN_STEPS-1][0:N_NEURONS-1];
    logic signed [31:0] delay_e2i [0:DELAY_E2I_STEPS-1][0:N_NEURONS-1];

    // STDP buffers
    logic [N_IN-1:0] s_in_hist [0:UPDATE_NT-1];
    logic [N_NEURONS-1:0] s_exc_hist [0:UPDATE_NT-1];
    logic signed [31:0] x_in_hist [0:UPDATE_NT-1][0:N_IN-1];
    logic signed [31:0] x_exc_hist[0:UPDATE_NT-1][0:N_NEURONS-1];
    logic [$clog2(UPDATE_NT):0] tcount;

    // Per-neuron abs weight sum (for normalization)
    logic signed [31:0] sum_abs [0:N_NEURONS-1];
    logic signed [31:0] sum_abs_base [0:N_NEURONS-1];

    initial begin
        if (W_INIT_FROM_FILE) begin
            $readmemh("data/sum_abs.mem", sum_abs);
        end
    end

    typedef enum logic [3:0] {
        S_IDLE,
        S_SCAN,
        S_GIN_WAIT,
        S_NEURON,
        S_OUT,
        S_STDP_INIT,
        S_STDP_T,
        S_STDP_READ,
        S_STDP_CALC
    } state_e;
    state_e state;

    function automatic signed [31:0] fp_mul(input signed [31:0] a, input signed [31:0] b);
        begin
            fp_mul = (a * b) >>> FP_SHIFT;
        end
    endfunction

    function automatic signed [31:0] fp_div_round(input signed [31:0] a, input int div);
        begin
            if (a >= 0)
                fp_div_round = (a + (div >> 1)) / div;
            else
                fp_div_round = (a - (div >> 1)) / div;
        end
    endfunction

    // Memory interface for W_in (4-bank)
    logic mem_r_en;
    logic [NEURON_W-1:0] mem_r_neuron [0:LANES-1];
    logic [IN_W-1:0] mem_r_in [0:LANES-1];
    logic signed [31:0] mem_r_data [0:LANES-1];

    logic mem_w_en [0:LANES-1];
    logic [NEURON_W-1:0] mem_w_neuron [0:LANES-1];
    logic [IN_W-1:0] mem_w_in [0:LANES-1];
    logic signed [31:0] mem_w_data [0:LANES-1];

    w_in_mem_4bank #(
        .N_IN(N_IN),
        .N_NEURONS(N_NEURONS),
        .INIT_VAL(32'sd66),
        .INIT_FROM_FILE(W_INIT_FROM_FILE)
    ) u_wmem (
        .clk(clk),
        .rst(rst),
        .r_en(mem_r_en),
        .r_neuron0(mem_r_neuron[0]),
        .r_neuron1(mem_r_neuron[1]),
        .r_neuron2(mem_r_neuron[2]),
        .r_neuron3(mem_r_neuron[3]),
        .r_in0(mem_r_in[0]),
        .r_in1(mem_r_in[1]),
        .r_in2(mem_r_in[2]),
        .r_in3(mem_r_in[3]),
        .r_data0(mem_r_data[0]),
        .r_data1(mem_r_data[1]),
        .r_data2(mem_r_data[2]),
        .r_data3(mem_r_data[3]),
        .w_en0(mem_w_en[0]),
        .w_en1(mem_w_en[1]),
        .w_en2(mem_w_en[2]),
        .w_en3(mem_w_en[3]),
        .w_neuron0(mem_w_neuron[0]),
        .w_neuron1(mem_w_neuron[1]),
        .w_neuron2(mem_w_neuron[2]),
        .w_neuron3(mem_w_neuron[3]),
        .w_in0(mem_w_in[0]),
        .w_in1(mem_w_in[1]),
        .w_in2(mem_w_in[2]),
        .w_in3(mem_w_in[3]),
        .w_data0(mem_w_data[0]),
        .w_data1(mem_w_data[1]),
        .w_data2(mem_w_data[2]),
        .w_data3(mem_w_data[3]),
        .dbg_en(dbg_en),
        .dbg_neuron(dbg_neuron),
        .dbg_in(dbg_in),
        .dbg_valid(dbg_valid),
        .dbg_data(dbg_data)
    );

    // Scan and STDP counters
    logic [IN_W-1:0] scan_in_idx;
    logic [$clog2(NEURON_GROUPS):0] scan_group_idx;
    logic scan_spike_active;

    logic stdp_pending;
    logic [IN_W-1:0] stdp_j;
    logic [$clog2(NEURON_GROUPS):0] stdp_g;
    logic [$clog2(UPDATE_NT):0] stdp_t;
    logic signed [31:0] stdp_sum1 [0:LANES-1];
    logic signed [31:0] stdp_sum2 [0:LANES-1];

    // Combinational math for neuron update
    logic signed [31:0] g_in_state_next [0:N_NEURONS-1];
    logic signed [31:0] r_exc_next [0:N_NEURONS-1];
    logic signed [31:0] x_exc_next [0:N_NEURONS-1];
    logic signed [31:0] r_inh_next [0:N_NEURONS-1];
    logic signed [31:0] v_exc_next [0:N_NEURONS-1];
    logic signed [31:0] theta_next [0:N_NEURONS-1];
    logic signed [31:0] vthr_next [0:N_NEURONS-1];
    logic [15:0]        refr_exc_next [0:N_NEURONS-1];

    logic signed [31:0] v_inh_next [0:N_NEURONS-1];
    logic [15:0]        refr_inh_next [0:N_NEURONS-1];

    logic [N_NEURONS-1:0] s_exc_next;
    logic [N_NEURONS-1:0] s_inh_next;
    logic signed [31:0] g_exc_next [0:N_NEURONS-1];
    logic signed [31:0] g_inh_next [0:N_NEURONS-1];

    always_comb begin
        for (i = 0; i < N_NEURONS; i = i + 1) begin
            g_in_state_next[i] = g_in_state[i] - fp_div_round(g_in_state[i], TD_IN_STEPS) + g_in_accum[i];
        end

        for (i = 0; i < N_NEURONS; i = i + 1) begin
            logic signed [31:0] i_syn_exc;
            logic signed [31:0] i_syn_inh;
            logic signed [31:0] num_exc;
            logic signed [31:0] dv_exc;
            logic signed [31:0] v_next_exc;
            logic signed [31:0] theta_tmp;
            logic signed [31:0] g_in_delayed_val;

            g_in_delayed_val = delay_in[DELAY_IN_STEPS-1][i];

            i_syn_exc = fp_mul(g_in_delayed_val, (EXC_E_EXC*FP_SCALE) - v_exc[i]);
            i_syn_inh = fp_mul(g_inh_state[i], (EXC_E_INH*FP_SCALE) - v_exc[i]);
            num_exc = (EXC_VREST*FP_SCALE) - v_exc[i] + i_syn_exc + i_syn_inh;
            dv_exc = fp_div_round(num_exc, EXC_TAU_M);
            v_next_exc = v_exc[i] + dv_exc;

            if (refr_exc[i] != 0) begin
                refr_exc_next[i] = refr_exc[i] - 1'b1;
                v_exc_next[i] = EXC_VRESET * FP_SCALE;
                theta_tmp = theta[i] - fp_div_round(theta[i], EXC_TC_THETA);
                s_exc_next[i] = 1'b0;
            end else begin
                if (v_next_exc >= vthr[i]) begin
                    s_exc_next[i] = 1'b1;
                    v_exc_next[i] = EXC_VRESET * FP_SCALE;
                    refr_exc_next[i] = EXC_REFRACT;
                    theta_tmp = theta[i] - fp_div_round(theta[i], EXC_TC_THETA) + EXC_THETA_PLUS_FP;
                end else begin
                    s_exc_next[i] = 1'b0;
                    v_exc_next[i] = v_next_exc;
                    refr_exc_next[i] = 0;
                    theta_tmp = theta[i] - fp_div_round(theta[i], EXC_TC_THETA);
                end
            end
            if (theta_tmp < 0) theta_tmp = 0;
            if (theta_tmp > (EXC_THETA_MAX*FP_SCALE)) theta_tmp = EXC_THETA_MAX*FP_SCALE;
            theta_next[i] = theta_tmp;
            vthr_next[i] = (EXC_INIT_VTHR * FP_SCALE) + theta_tmp;
        end

        for (i = 0; i < N_NEURONS; i = i + 1) begin
            r_exc_next[i] = r_exc[i] - fp_div_round(r_exc[i], TD_EXC_STEPS)
                          + (s_exc_next[i] ? (FP_SCALE / TD_EXC_STEPS) : 0);
            x_exc_next[i] = x_exc[i] - fp_div_round(x_exc[i], TD_X_STEPS)
                          + (s_exc_next[i] ? (FP_SCALE / TD_X_STEPS) : 0);
            g_exc_next[i] = fp_mul(WEXC_FP, r_exc_next[i]);
        end

        for (i = 0; i < N_NEURONS; i = i + 1) begin
            logic signed [31:0] i_syn_exc_i;
            logic signed [31:0] num_i;
            logic signed [31:0] dv_i;
            logic signed [31:0] v_next_i;
            logic signed [31:0] g_exc_delayed_val;

            g_exc_delayed_val = delay_e2i[DELAY_E2I_STEPS-1][i];
            i_syn_exc_i = fp_mul(g_exc_delayed_val, (INH_E_EXC*FP_SCALE) - v_inh[i]);
            num_i = (INH_VREST*FP_SCALE) - v_inh[i] + i_syn_exc_i;
            dv_i = fp_div_round(num_i, INH_TAU_M);
            v_next_i = v_inh[i] + dv_i;

            if (refr_inh[i] != 0) begin
                refr_inh_next[i] = refr_inh[i] - 1'b1;
                v_inh_next[i] = INH_VRESET * FP_SCALE;
                s_inh_next[i] = 1'b0;
            end else begin
                if (v_next_i >= (INH_VTHR*FP_SCALE)) begin
                    s_inh_next[i] = 1'b1;
                    v_inh_next[i] = INH_VRESET * FP_SCALE;
                    refr_inh_next[i] = INH_REFRACT;
                end else begin
                    s_inh_next[i] = 1'b0;
                    v_inh_next[i] = v_next_i;
                    refr_inh_next[i] = 0;
                end
            end
        end

        for (i = 0; i < N_NEURONS; i = i + 1) begin
            r_inh_next[i] = r_inh[i] - fp_div_round(r_inh[i], TD_INH_STEPS)
                          + (s_inh_next[i] ? (FP_SCALE / TD_INH_STEPS) : 0);
        end

        for (i = 0; i < N_NEURONS; i = i + 1) begin
            logic signed [63:0] acc_inh;
            acc_inh = 0;
            for (j = 0; j < N_NEURONS; j = j + 1) begin
                if (j != i) begin
                    acc_inh = acc_inh + r_inh_next[j];
                end
            end
            if (N_NEURONS > 1) begin
                g_inh_next[i] = fp_mul(fp_div_round(WINH_FP, (N_NEURONS-1)), acc_inh[31:0]);
            end else begin
                g_inh_next[i] = 0;
            end
        end
    end

    // Sequential state updates
    always_ff @(posedge clk) begin
        if (rst) begin
            s_tready <= 1'b1;
            m_tvalid <= 1'b0;
            m_tdata <= '0;
            tstep_id_reg <= '0;
            s_in_reg <= '0;
            s_stdp_reg <= 1'b0;
            tcount <= 0;
            state <= S_IDLE;
            scan_in_idx <= '0;
            scan_group_idx <= '0;
            scan_spike_active <= 1'b0;
            stdp_pending <= 1'b0;
            stdp_j <= '0;
            stdp_g <= '0;
            stdp_t <= '0;

            for (i = 0; i < N_IN; i = i + 1) begin
                x_in[i] <= '0;
            end
            for (i = 0; i < N_NEURONS; i = i + 1) begin
                r_exc[i] <= '0;
                r_inh[i] <= '0;
                x_exc[i] <= '0;
                v_exc[i] <= EXC_VRESET * FP_SCALE;
                theta[i] <= '0;
                vthr[i] <= EXC_INIT_VTHR * FP_SCALE;
                refr_exc[i] <= '0;
                v_inh[i] <= INH_VRESET * FP_SCALE;
                refr_inh[i] <= '0;
                g_inh_state[i] <= '0;
                g_in_state[i] <= '0;
                g_in_accum[i] <= '0;
                g_exc[i] <= '0;
                if (!W_INIT_FROM_FILE) begin
                    sum_abs[i] <= 32'sd66 * N_IN;
                end
                s_exc[i] <= 1'b0;
                s_inh[i] <= 1'b0;
            end
            for (t = 0; t < UPDATE_NT; t = t + 1) begin
                s_in_hist[t] <= '0;
                s_exc_hist[t] <= '0;
                for (i = 0; i < N_IN; i = i + 1) begin
                    x_in_hist[t][i] <= '0;
                end
                for (i = 0; i < N_NEURONS; i = i + 1) begin
                    x_exc_hist[t][i] <= '0;
                end
            end
            for (i = 0; i < DELAY_IN_STEPS; i = i + 1) begin
                for (j = 0; j < N_NEURONS; j = j + 1) begin
                    delay_in[i][j] <= '0;
                end
            end
            for (i = 0; i < DELAY_E2I_STEPS; i = i + 1) begin
                for (j = 0; j < N_NEURONS; j = j + 1) begin
                    delay_e2i[i][j] <= '0;
                end
            end
        end else begin
            // defaults
            mem_r_en <= 1'b0;
            for (i = 0; i < LANES; i = i + 1) begin
                mem_w_en[i] <= 1'b0;
            end

            case (state)
                S_IDLE: begin
                    s_tready <= 1'b1;
                    if (s_tvalid && s_tready) begin
                        tstep_id_reg <= s_tdata[TSTEP_W+N_IN-1 -: TSTEP_W];
                        s_in_reg <= s_tdata[N_IN-1:0];
                        s_stdp_reg <= s_stdp_en;

                        // clear accumulators
                        for (i = 0; i < N_NEURONS; i = i + 1) begin
                            g_in_accum[i] <= '0;
                        end

                        scan_in_idx <= '0;
                        scan_group_idx <= '0;
                        scan_spike_active <= 1'b0;
                        s_tready <= 1'b0;
                        state <= S_SCAN;
                    end
                end

                // Scan inputs; update x_in/x_in_hist; if spike, read weights in groups of 4 neurons
                S_SCAN: begin
                    if (scan_in_idx < N_IN) begin
                        // update trace for this input
                        begin : trace_update
                            logic signed [31:0] x_next;
                            x_next = x_in[scan_in_idx] - fp_div_round(x_in[scan_in_idx], TD_X_STEPS)
                                   + (s_in_reg[scan_in_idx] ? TRACE_SPIKE_FP : 0);
                            x_in[scan_in_idx] <= x_next;
                            if (s_stdp_reg) begin
                                x_in_hist[tcount][scan_in_idx] <= x_next;
                                s_in_hist[tcount][scan_in_idx] <= s_in_reg[scan_in_idx];
                            end
                        end

                        if (s_in_reg[scan_in_idx]) begin
                            // start reading weights for this input
                            scan_group_idx <= '0;
                            scan_spike_active <= 1'b1;
                            mem_r_en <= 1'b1;
                            for (i = 0; i < LANES; i = i + 1) begin
                                mem_r_neuron[i] <= scan_group_idx * LANES + i;
                                mem_r_in[i] <= scan_in_idx;
                            end
                            state <= S_GIN_WAIT;
                        end else begin
                            scan_in_idx <= scan_in_idx + 1'b1;
                        end
                    end else begin
                        state <= S_NEURON;
                    end
                end

                // Wait for W_in read, accumulate g_in for this spike
                S_GIN_WAIT: begin
                    if (scan_spike_active) begin
                        for (i = 0; i < LANES; i = i + 1) begin
                            int neuron_idx;
                            neuron_idx = scan_group_idx * LANES + i;
                            if (neuron_idx < N_NEURONS) begin
                                g_in_accum[neuron_idx] <= g_in_accum[neuron_idx]
                                    + fp_mul(mem_r_data[i], INPUT_SPIKE_FP);
                            end
                        end

                        if (scan_group_idx == NEURON_GROUPS-1) begin
                            scan_spike_active <= 1'b0;
                            scan_in_idx <= scan_in_idx + 1'b1;
                            state <= S_SCAN;
                        end else begin
                            scan_group_idx <= scan_group_idx + 1'b1;
                            mem_r_en <= 1'b1;
                            for (i = 0; i < LANES; i = i + 1) begin
                                mem_r_neuron[i] <= (scan_group_idx + 1'b1) * LANES + i;
                                mem_r_in[i] <= scan_in_idx;
                            end
                            state <= S_GIN_WAIT;
                        end
                    end else begin
                        state <= S_SCAN;
                    end
                end

                // Finish neuron update (LIF, synapses, delays) in one step
                S_NEURON: begin
                    for (i = 0; i < N_NEURONS; i = i + 1) begin
                        g_in_state[i] <= g_in_state_next[i];
                        r_exc[i] <= r_exc_next[i];
                        x_exc[i] <= x_exc_next[i];
                        r_inh[i] <= r_inh_next[i];
                        v_exc[i] <= v_exc_next[i];
                        theta[i] <= theta_next[i];
                        vthr[i] <= vthr_next[i];
                        refr_exc[i] <= refr_exc_next[i];
                        v_inh[i] <= v_inh_next[i];
                        refr_inh[i] <= refr_inh_next[i];
                        g_inh_state[i] <= g_inh_next[i];
                        g_exc[i] <= g_exc_next[i];
                        s_exc[i] <= s_exc_next[i];
                        s_inh[i] <= s_inh_next[i];
                    end

                    // update delays
                    for (i = DELAY_IN_STEPS-1; i > 0; i = i - 1) begin
                        for (j = 0; j < N_NEURONS; j = j + 1) begin
                            delay_in[i][j] <= delay_in[i-1][j];
                        end
                    end
                    for (j = 0; j < N_NEURONS; j = j + 1) begin
                        delay_in[0][j] <= g_in_state_next[j];
                    end
                    for (i = DELAY_E2I_STEPS-1; i > 0; i = i - 1) begin
                        for (j = 0; j < N_NEURONS; j = j + 1) begin
                            delay_e2i[i][j] <= delay_e2i[i-1][j];
                        end
                    end
                    for (j = 0; j < N_NEURONS; j = j + 1) begin
                        delay_e2i[0][j] <= g_exc_next[j];
                    end

                    // STDP buffers
                    if (s_stdp_reg) begin
                        s_exc_hist[tcount] <= s_exc_next;
                        for (i = 0; i < N_NEURONS; i = i + 1) begin
                            x_exc_hist[tcount][i] <= x_exc_next[i];
                        end
                        if (tcount == UPDATE_NT-1) begin
                            tcount <= 0;
                            stdp_pending <= 1'b1;
                        end else begin
                            tcount <= tcount + 1'b1;
                        end
                    end

                    m_tdata <= {tstep_id_reg, s_exc_next};
                    m_tvalid <= 1'b1;
                    state <= S_OUT;
                end

                S_OUT: begin
                    if (m_tvalid && m_tready) begin
                        m_tvalid <= 1'b0;
                        if (stdp_pending) begin
                            stdp_pending <= 1'b0;
                            state <= S_STDP_INIT;
                        end else begin
                            state <= S_IDLE;
                        end
                    end
                end

                // STDP update (4 weights per cycle)
                S_STDP_INIT: begin
                    stdp_j <= '0;
                    stdp_g <= '0;
                    stdp_t <= '0;
                    for (i = 0; i < N_NEURONS; i = i + 1) begin
                        sum_abs_base[i] <= sum_abs[i];
                    end
                    for (i = 0; i < LANES; i = i + 1) begin
                        stdp_sum1[i] <= '0;
                        stdp_sum2[i] <= '0;
                    end
                    state <= S_STDP_T;
                end

                S_STDP_T: begin
                    for (i = 0; i < LANES; i = i + 1) begin
                        int neuron_idx;
                        neuron_idx = stdp_g * LANES + i;
                        if (neuron_idx < N_NEURONS) begin
                            if (s_exc_hist[stdp_t][neuron_idx]) begin
                                stdp_sum1[i] <= stdp_sum1[i] + x_in_hist[stdp_t][stdp_j];
                            end
                            if (s_in_hist[stdp_t][stdp_j]) begin
                                stdp_sum2[i] <= stdp_sum2[i] + x_exc_hist[stdp_t][neuron_idx];
                            end
                        end
                    end

                    if (stdp_t == UPDATE_NT-1) begin
                        mem_r_en <= 1'b1;
                        for (i = 0; i < LANES; i = i + 1) begin
                            mem_r_neuron[i] <= stdp_g * LANES + i;
                            mem_r_in[i] <= stdp_j;
                        end
                        state <= S_STDP_READ;
                    end else begin
                        stdp_t <= stdp_t + 1'b1;
                    end
                end

                S_STDP_READ: begin
                    state <= S_STDP_CALC;
                end

                S_STDP_CALC: begin
                    for (i = 0; i < LANES; i = i + 1) begin
                        int neuron_idx;
                        logic signed [31:0] w_old;
                        logic signed [31:0] w_norm;
                        logic signed [31:0] dW;
                        logic signed [31:0] w_new;
                        logic signed [31:0] sum_abs_val;
                        logic signed [31:0] abs_old;
                        logic signed [31:0] abs_new;

                        neuron_idx = stdp_g * LANES + i;
                        if (neuron_idx < N_NEURONS) begin
                            w_old = mem_r_data[i];
                            sum_abs_val = sum_abs_base[neuron_idx];
                            if (sum_abs_val == 0) sum_abs_val = 1;

                            w_norm = fp_mul(w_old, fp_div_round(NORM_FP, sum_abs_val));

                            dW = fp_div_round(
                                fp_mul(fp_mul((WMAX_FP - w_norm), stdp_sum1[i]), LR_P_FP)
                              - fp_mul(fp_mul(w_norm, stdp_sum2[i]), LR_M_FP),
                                UPDATE_NT
                            );
                            if (dW > DW_CLIP_FP) dW = DW_CLIP_FP;
                            if (dW < -DW_CLIP_FP) dW = -DW_CLIP_FP;
                            w_new = w_norm + dW;
                            if (w_new < WMIN_FP) w_new = WMIN_FP;
                            if (w_new > WMAX_FP) w_new = WMAX_FP;

                            mem_w_en[i] <= 1'b1;
                            mem_w_neuron[i] <= neuron_idx[NEURON_W-1:0];
                            mem_w_in[i] <= stdp_j;
                            mem_w_data[i] <= w_new;

                            abs_old = w_old[31] ? -w_old : w_old;
                            abs_new = w_new[31] ? -w_new : w_new;
                            sum_abs[neuron_idx] <= sum_abs_val + abs_new - abs_old;
                        end
                    end

                    // advance to next weight group
                    if (stdp_g == NEURON_GROUPS-1) begin
                        stdp_g <= '0;
                        if (stdp_j == N_IN-1) begin
                            state <= S_IDLE;
                        end else begin
                            stdp_j <= stdp_j + 1'b1;
                            stdp_t <= '0;
                            for (i = 0; i < LANES; i = i + 1) begin
                                stdp_sum1[i] <= '0;
                                stdp_sum2[i] <= '0;
                            end
                            state <= S_STDP_T;
                        end
                    end else begin
                        stdp_g <= stdp_g + 1'b1;
                        stdp_t <= '0;
                        for (i = 0; i < LANES; i = i + 1) begin
                            stdp_sum1[i] <= '0;
                            stdp_sum2[i] <= '0;
                        end
                        state <= S_STDP_T;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
