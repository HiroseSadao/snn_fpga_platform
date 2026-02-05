`default_nettype none

module pipeline_small #(
        parameter int TSTEP_W = 16,
        parameter int N_IN = 4,
        parameter int N_NEURONS = 4,
        parameter int UPDATE_NT = 8
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
        output logic signed [N_NEURONS*N_IN*32-1:0] w_flat
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

    // State
    integer i, j, t;

    logic [TSTEP_W-1:0] tstep_id_reg;
    logic [N_IN-1:0] s_in_reg;
    logic s_stdp_reg;

    logic signed [31:0] r_in   [0:N_IN-1];
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

    logic signed [31:0] W_in   [0:N_NEURONS-1][0:N_IN-1];

    genvar gi, gj;
    generate
        for (gi = 0; gi < N_NEURONS; gi = gi + 1) begin : gen_wflat_i
            for (gj = 0; gj < N_IN; gj = gj + 1) begin : gen_wflat_j
                assign w_flat[(gi*N_IN+gj)*32 +: 32] = W_in[gi][gj];
            end
        end
    endgenerate

    logic signed [31:0] g_in   [0:N_NEURONS-1];
    logic signed [31:0] g_in_delayed [0:N_NEURONS-1];
    logic signed [31:0] g_exc  [0:N_NEURONS-1];
    logic signed [31:0] g_exc_delayed [0:N_NEURONS-1];
    logic signed [31:0] g_inh_next [0:N_NEURONS-1];
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

    typedef enum logic [0:0] {S_IDLE, S_OUT} state_e;
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

    // Combinational math for one timestep
    logic signed [31:0] r_in_next [0:N_IN-1];
    logic signed [31:0] x_in_next [0:N_IN-1];
    logic signed [31:0] r_exc_next [0:N_NEURONS-1];
    logic signed [31:0] x_exc_next [0:N_NEURONS-1];
    logic signed [31:0] r_inh_next [0:N_NEURONS-1];

    logic signed [31:0] v_exc_next [0:N_NEURONS-1];
    logic signed [31:0] theta_next [0:N_NEURONS-1];
    logic signed [31:0] vthr_next [0:N_NEURONS-1];
    logic [15:0] refr_exc_next [0:N_NEURONS-1];

    logic signed [31:0] v_inh_next [0:N_NEURONS-1];
    logic [15:0] refr_inh_next [0:N_NEURONS-1];

    logic [N_NEURONS-1:0] s_exc_next;
    logic [N_NEURONS-1:0] s_inh_next;
    logic stdp_do_update;

    logic signed [63:0] acc_in [0:N_NEURONS-1];
    logic signed [31:0] i_syn_exc_arr [0:N_NEURONS-1];
    logic signed [31:0] i_syn_inh_arr [0:N_NEURONS-1];
    logic signed [31:0] num_exc_arr [0:N_NEURONS-1];
    logic signed [31:0] dv_exc_arr [0:N_NEURONS-1];
    logic signed [31:0] v_next_exc_arr [0:N_NEURONS-1];
    logic signed [31:0] theta_tmp_arr [0:N_NEURONS-1];

    logic signed [31:0] i_syn_exc_i_arr [0:N_NEURONS-1];
    logic signed [31:0] num_i_arr [0:N_NEURONS-1];
    logic signed [31:0] dv_i_arr [0:N_NEURONS-1];
    logic signed [31:0] v_next_i_arr [0:N_NEURONS-1];

    logic signed [63:0] acc_inh_arr [0:N_NEURONS-1];

    logic signed [31:0] sum_abs_arr [0:N_NEURONS-1];
    logic signed [31:0] sum1_arr [0:N_NEURONS-1][0:N_IN-1];
    logic signed [31:0] sum2_arr [0:N_NEURONS-1][0:N_IN-1];
    logic signed [31:0] Wn_arr [0:N_NEURONS-1][0:N_IN-1];
    logic signed [31:0] W_new [0:N_NEURONS-1][0:N_IN-1];

    logic [N_IN-1:0] s_in_effective;
    logic s_stdp_effective;
    logic accept_in;

    always_comb begin
        accept_in = (state == S_IDLE) && s_tvalid && s_tready;
        s_in_effective = accept_in ? s_tdata[N_IN-1:0] : s_in_reg;
        s_stdp_effective = accept_in ? s_stdp_en : s_stdp_reg;
        stdp_do_update = 1'b0;
        // default pass-through
        for (i = 0; i < N_IN; i = i + 1) begin
            r_in_next[i] = r_in[i];
            x_in_next[i] = x_in[i];
        end
        for (i = 0; i < N_NEURONS; i = i + 1) begin
            r_exc_next[i] = r_exc[i];
            x_exc_next[i] = x_exc[i];
            r_inh_next[i] = r_inh[i];
            v_exc_next[i] = v_exc[i];
            theta_next[i] = theta[i];
            vthr_next[i] = vthr[i];
            refr_exc_next[i] = refr_exc[i];
            v_inh_next[i] = v_inh[i];
            refr_inh_next[i] = refr_inh[i];
            s_exc_next[i] = 1'b0;
            s_inh_next[i] = 1'b0;
            g_in[i] = 0;
            g_exc[i] = 0;
            g_inh_next[i] = 0;
        end

        // Update input synapse and trace
        for (i = 0; i < N_IN; i = i + 1) begin
            r_in_next[i] = r_in[i] - fp_div_round(r_in[i], TD_IN_STEPS)
                         + (s_in_effective[i] ? (FP_SCALE / TD_IN_STEPS) : 0);
            x_in_next[i] = x_in[i] - fp_div_round(x_in[i], TD_X_STEPS)
                         + (s_in_effective[i] ? (FP_SCALE / TD_X_STEPS) : 0);
        end

        // g_in = W * c_in
        for (i = 0; i < N_NEURONS; i = i + 1) begin
            acc_in[i] = 0;
            for (j = 0; j < N_IN; j = j + 1) begin
                acc_in[i] = acc_in[i] + $signed(W_in[i][j]) * $signed(r_in_next[j]);
            end
            g_in[i] = acc_in[i] >>> FP_SHIFT;
        end

        // Apply delays
        for (i = 0; i < N_NEURONS; i = i + 1) begin
            g_in_delayed[i] = delay_in[DELAY_IN_STEPS-1][i];
            g_exc_delayed[i] = delay_e2i[DELAY_E2I_STEPS-1][i];
        end

        // Exc LIF
        for (i = 0; i < N_NEURONS; i = i + 1) begin
            i_syn_exc_arr[i] = fp_mul(g_in_delayed[i], (EXC_E_EXC*FP_SCALE) - v_exc[i]);
            i_syn_inh_arr[i] = fp_mul(g_inh_state[i], (EXC_E_INH*FP_SCALE) - v_exc[i]);
            num_exc_arr[i] = (EXC_VREST*FP_SCALE) - v_exc[i] + i_syn_exc_arr[i] + i_syn_inh_arr[i];
            dv_exc_arr[i] = fp_div_round(num_exc_arr[i], EXC_TAU_M);
            v_next_exc_arr[i] = v_exc[i] + dv_exc_arr[i];

            if (refr_exc[i] != 0) begin
                refr_exc_next[i] = refr_exc[i] - 1'b1;
                v_exc_next[i] = EXC_VRESET * FP_SCALE;
                theta_tmp_arr[i] = theta[i] - fp_div_round(theta[i], EXC_TC_THETA);
                s_exc_next[i] = 1'b0;
            end else begin
                if (v_next_exc_arr[i] >= vthr[i]) begin
                    s_exc_next[i] = 1'b1;
                    v_exc_next[i] = EXC_VRESET * FP_SCALE;
                    refr_exc_next[i] = EXC_REFRACT;
                    theta_tmp_arr[i] = theta[i] - fp_div_round(theta[i], EXC_TC_THETA) + EXC_THETA_PLUS_FP;
                end else begin
                    s_exc_next[i] = 1'b0;
                    v_exc_next[i] = v_next_exc_arr[i];
                    refr_exc_next[i] = 0;
                    theta_tmp_arr[i] = theta[i] - fp_div_round(theta[i], EXC_TC_THETA);
                end
            end
            if (theta_tmp_arr[i] < 0) theta_tmp_arr[i] = 0;
            if (theta_tmp_arr[i] > (EXC_THETA_MAX*FP_SCALE)) theta_tmp_arr[i] = EXC_THETA_MAX*FP_SCALE;
            theta_next[i] = theta_tmp_arr[i];
            vthr_next[i] = (EXC_INIT_VTHR * FP_SCALE) + theta_tmp_arr[i];
        end

        // Exc synapse and trace
        for (i = 0; i < N_NEURONS; i = i + 1) begin
            r_exc_next[i] = r_exc[i] - fp_div_round(r_exc[i], TD_EXC_STEPS)
                          + (s_exc_next[i] ? (FP_SCALE / TD_EXC_STEPS) : 0);
            x_exc_next[i] = x_exc[i] - fp_div_round(x_exc[i], TD_X_STEPS)
                          + (s_exc_next[i] ? (FP_SCALE / TD_X_STEPS) : 0);
            g_exc[i] = fp_mul(WEXC_FP, r_exc_next[i]);
        end

        // Inh LIF
        for (i = 0; i < N_NEURONS; i = i + 1) begin
            i_syn_exc_i_arr[i] = fp_mul(g_exc_delayed[i], (INH_E_EXC*FP_SCALE) - v_inh[i]);
            num_i_arr[i] = (INH_VREST*FP_SCALE) - v_inh[i] + i_syn_exc_i_arr[i];
            dv_i_arr[i] = fp_div_round(num_i_arr[i], INH_TAU_M);
            v_next_i_arr[i] = v_inh[i] + dv_i_arr[i];

            if (refr_inh[i] != 0) begin
                refr_inh_next[i] = refr_inh[i] - 1'b1;
                v_inh_next[i] = INH_VRESET * FP_SCALE;
                s_inh_next[i] = 1'b0;
            end else begin
                if (v_next_i_arr[i] >= (INH_VTHR*FP_SCALE)) begin
                    s_inh_next[i] = 1'b1;
                    v_inh_next[i] = INH_VRESET * FP_SCALE;
                    refr_inh_next[i] = INH_REFRACT;
                end else begin
                    s_inh_next[i] = 1'b0;
                    v_inh_next[i] = v_next_i_arr[i];
                    refr_inh_next[i] = 0;
                end
            end
        end

        // Inh synapse and g_inh
        for (i = 0; i < N_NEURONS; i = i + 1) begin
            r_inh_next[i] = r_inh[i] - fp_div_round(r_inh[i], TD_INH_STEPS)
                          + (s_inh_next[i] ? (FP_SCALE / TD_INH_STEPS) : 0);
        end
        for (i = 0; i < N_NEURONS; i = i + 1) begin
            acc_inh_arr[i] = 0;
            for (j = 0; j < N_NEURONS; j = j + 1) begin
                if (j != i) begin
                    acc_inh_arr[i] = acc_inh_arr[i] + r_inh_next[j];
                end
            end
            if (N_NEURONS > 1) begin
                g_inh_next[i] = fp_mul(fp_div_round(WINH_FP, (N_NEURONS-1)), acc_inh_arr[i]);
            end else begin
                g_inh_next[i] = 0;
            end
        end

        stdp_do_update = s_stdp_effective && (tcount == UPDATE_NT-1);
        if (stdp_do_update) begin
            for (i = 0; i < N_NEURONS; i = i + 1) begin
                sum_abs_arr[i] = 0;
                for (j = 0; j < N_IN; j = j + 1) begin
                    sum_abs_arr[i] = sum_abs_arr[i] + (W_in[i][j][31] ? -W_in[i][j] : W_in[i][j]);
                end
                if (sum_abs_arr[i] == 0) sum_abs_arr[i] = 1;
                for (j = 0; j < N_IN; j = j + 1) begin
                    sum1_arr[i][j] = 0;
                    sum2_arr[i][j] = 0;
                    for (t = 0; t < UPDATE_NT; t = t + 1) begin
                        if (s_exc_hist[t][i]) sum1_arr[i][j] = sum1_arr[i][j] + x_in_hist[t][j];
                        if (s_in_hist[t][j]) sum2_arr[i][j] = sum2_arr[i][j] + x_exc_hist[t][i];
                    end
                    Wn_arr[i][j] = fp_mul(W_in[i][j], fp_div_round(NORM_FP, sum_abs_arr[i]));
                    W_new[i][j] = Wn_arr[i][j];
                    W_new[i][j] = W_new[i][j] + fp_div_round(
                        fp_mul(fp_mul((WMAX_FP - Wn_arr[i][j]), sum1_arr[i][j]), LR_P_FP)
                      - fp_mul(fp_mul(Wn_arr[i][j], sum2_arr[i][j]), LR_M_FP),
                        UPDATE_NT
                    );
                    if (W_new[i][j] > (Wn_arr[i][j] + DW_CLIP_FP)) W_new[i][j] = Wn_arr[i][j] + DW_CLIP_FP;
                    if (W_new[i][j] < (Wn_arr[i][j] - DW_CLIP_FP)) W_new[i][j] = Wn_arr[i][j] - DW_CLIP_FP;
                    if (W_new[i][j] < WMIN_FP) W_new[i][j] = WMIN_FP;
                    if (W_new[i][j] > WMAX_FP) W_new[i][j] = WMAX_FP;
                end
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
            for (i = 0; i < N_IN; i = i + 1) begin
                r_in[i] <= '0;
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
            for (i = 0; i < N_NEURONS; i = i + 1) begin
                for (j = 0; j < N_IN; j = j + 1) begin
                    W_in[i][j] <= 32'sd66; // 0.001 in S16.16
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
            case (state)
                S_IDLE: begin
                    s_tready <= 1'b1;
                    if (s_tvalid && s_tready) begin
                        tstep_id_reg <= s_tdata[TSTEP_W+N_IN-1 -: TSTEP_W];
                        s_in_reg <= s_tdata[N_IN-1:0];
                        s_stdp_reg <= s_stdp_en;
                        // apply updates
                        for (i = 0; i < N_IN; i = i + 1) begin
                            r_in[i] <= r_in_next[i];
                            x_in[i] <= x_in_next[i];
                        end
                        for (i = 0; i < N_NEURONS; i = i + 1) begin
                            r_exc[i] <= r_exc_next[i];
                            r_inh[i] <= r_inh_next[i];
                            x_exc[i] <= x_exc_next[i];
                            v_exc[i] <= v_exc_next[i];
                            theta[i] <= theta_next[i];
                            vthr[i] <= vthr_next[i];
                            refr_exc[i] <= refr_exc_next[i];
                            v_inh[i] <= v_inh_next[i];
                            refr_inh[i] <= refr_inh_next[i];
                            g_inh_state[i] <= g_inh_next[i];
                        end
                        // update delays
                        for (i = DELAY_IN_STEPS-1; i > 0; i = i - 1) begin
                            for (j = 0; j < N_NEURONS; j = j + 1) begin
                                delay_in[i][j] <= delay_in[i-1][j];
                            end
                        end
                        for (j = 0; j < N_NEURONS; j = j + 1) begin
                            delay_in[0][j] <= g_in[j];
                        end
                        for (i = DELAY_E2I_STEPS-1; i > 0; i = i - 1) begin
                            for (j = 0; j < N_NEURONS; j = j + 1) begin
                                delay_e2i[i][j] <= delay_e2i[i-1][j];
                            end
                        end
                        for (j = 0; j < N_NEURONS; j = j + 1) begin
                            delay_e2i[0][j] <= g_exc[j];
                        end

                        // STDP buffers
                        if (s_stdp_en) begin
                            s_in_hist[tcount] <= s_tdata[N_IN-1:0];
                            s_exc_hist[tcount] <= s_exc_next;
                            for (i = 0; i < N_IN; i = i + 1) begin
                                x_in_hist[tcount][i] <= x_in_next[i];
                            end
                            for (i = 0; i < N_NEURONS; i = i + 1) begin
                                x_exc_hist[tcount][i] <= x_exc_next[i];
                            end
                            if (tcount == UPDATE_NT-1) begin
                                for (i = 0; i < N_NEURONS; i = i + 1) begin
                                    for (j = 0; j < N_IN; j = j + 1) begin
                                        W_in[i][j] <= W_new[i][j];
                                    end
                                end
                                tcount <= 0;
                            end else begin
                                tcount <= tcount + 1'b1;
                            end
                        end

                        m_tdata <= {s_tdata[TSTEP_W+N_IN-1 -: TSTEP_W], s_exc_next};
                        m_tvalid <= 1'b1;
                        s_tready <= 1'b0;
                        state <= S_OUT;
                    end
                end

                S_OUT: begin
                    if (m_tvalid && m_tready) begin
                        m_tvalid <= 1'b0;
                        state <= S_IDLE;
                    end
                end
            endcase
        end
    end

endmodule

`default_nettype wire
