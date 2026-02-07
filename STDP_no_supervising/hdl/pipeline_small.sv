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
    // Online STDP (stdp3.py): A_p=0.01, A_m=1.05*A_p
    localparam int A_P_FP = 655;     // 0.01
    localparam int A_M_FP = 688;     // 0.0105

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

    logic signed [31:0] g_inh_state [0:N_NEURONS-1];

    localparam int DELAY_IN_DEPTH = DELAY_IN_STEPS * N_NEURONS;
    localparam int DELAY_E2I_DEPTH = DELAY_E2I_STEPS * N_NEURONS;
    localparam int DELAY_IN_ADDR_W = (DELAY_IN_DEPTH <= 1) ? 1 : $clog2(DELAY_IN_DEPTH);
    localparam int DELAY_E2I_ADDR_W = (DELAY_E2I_DEPTH <= 1) ? 1 : $clog2(DELAY_E2I_DEPTH);
    (* ram_style = "block" *) logic signed [31:0] delay_in_mem [0:DELAY_IN_DEPTH-1];
    (* ram_style = "block" *) logic signed [31:0] delay_e2i_mem [0:DELAY_E2I_DEPTH-1];
    logic [DELAY_IN_ADDR_W-1:0] delay_in_rd_addr;
    logic [DELAY_E2I_ADDR_W-1:0] delay_e2i_rd_addr;
    logic signed [31:0] delay_in_rd_data;
    logic signed [31:0] delay_e2i_rd_data;
    localparam int DELAY_IN_W = (DELAY_IN_STEPS <= 1) ? 1 : $clog2(DELAY_IN_STEPS);
    localparam int DELAY_E2I_W = (DELAY_E2I_STEPS <= 1) ? 1 : $clog2(DELAY_E2I_STEPS);
    logic [DELAY_IN_W-1:0] delay_in_wr_idx;
    logic [DELAY_E2I_W-1:0] delay_e2i_wr_idx;
    logic [DELAY_IN_W-1:0] delay_in_clr_step;
    logic [DELAY_E2I_W-1:0] delay_e2i_clr_step;
    logic [NEURON_W-1:0] delay_clr_neuron;

    function automatic [DELAY_IN_W-1:0] delay_in_rd_idx;
        begin
            if (DELAY_IN_STEPS <= 1) begin
                delay_in_rd_idx = '0;
            end else if (delay_in_wr_idx == 0) begin
                delay_in_rd_idx = DELAY_IN_STEPS-1;
            end else begin
                delay_in_rd_idx = delay_in_wr_idx - 1'b1;
            end
        end
    endfunction

    function automatic [DELAY_E2I_W-1:0] delay_e2i_rd_idx;
        begin
            if (DELAY_E2I_STEPS <= 1) begin
                delay_e2i_rd_idx = '0;
            end else if (delay_e2i_wr_idx == 0) begin
                delay_e2i_rd_idx = DELAY_E2I_STEPS-1;
            end else begin
                delay_e2i_rd_idx = delay_e2i_wr_idx - 1'b1;
            end
        end
    endfunction

    function automatic [DELAY_IN_W-1:0] delay_in_next_idx;
        input [DELAY_IN_W-1:0] idx;
        begin
            if (DELAY_IN_STEPS <= 1) begin
                delay_in_next_idx = '0;
            end else if (idx == DELAY_IN_STEPS-1) begin
                delay_in_next_idx = '0;
            end else begin
                delay_in_next_idx = idx + 1'b1;
            end
        end
    endfunction

    function automatic [DELAY_E2I_W-1:0] delay_e2i_next_idx;
        input [DELAY_E2I_W-1:0] idx;
        begin
            if (DELAY_E2I_STEPS <= 1) begin
                delay_e2i_next_idx = '0;
            end else if (idx == DELAY_E2I_STEPS-1) begin
                delay_e2i_next_idx = '0;
            end else begin
                delay_e2i_next_idx = idx + 1'b1;
            end
        end
    endfunction

    function automatic [DELAY_IN_ADDR_W-1:0] delay_in_addr(
        input [DELAY_IN_W-1:0] step,
        input [NEURON_W-1:0] neuron
    );
        int unsigned addr;
        begin
            addr = (step * N_NEURONS) + neuron;
            delay_in_addr = addr[DELAY_IN_ADDR_W-1:0];
        end
    endfunction

    function automatic [DELAY_E2I_ADDR_W-1:0] delay_e2i_addr(
        input [DELAY_E2I_W-1:0] step,
        input [NEURON_W-1:0] neuron
    );
        int unsigned addr;
        begin
            addr = (step * N_NEURONS) + neuron;
            delay_e2i_addr = addr[DELAY_E2I_ADDR_W-1:0];
        end
    endfunction

    // No history buffers for online STDP

    typedef enum logic [3:0] {
        S_IDLE,
        S_SCAN,
        S_GIN_WAIT,
        S_NEURON_PREP,
        S_NEURON_CALC1,
        S_NEURON_CALC2,
        S_NEURON_COMMIT,
        S_OUT,
        S_STDP_READ,
        S_STDP_CALC,
        S_CLR_DELAY_IN,
        S_CLR_DELAY_E2I
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

    always_ff @(posedge clk) begin
        if (rst) begin
            mul_result <= '0;
            div_result <= '0;
        end else begin
            mul_result <= (mul_a * mul_b) >>> FP_SHIFT;
            if (div_div != 0) begin
                if (div_a >= 0)
                    div_result <= (div_a + (div_div >> 1)) / div_div;
                else
                    div_result <= (div_a - (div_div >> 1)) / div_div;
            end else begin
                div_result <= '0;
            end
        end
    end

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

    // Per-neuron next-state storage (computed over multiple cycles)
    (* ram_style = "block" *) logic signed [31:0] g_in_state_next [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] r_exc_next [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] x_exc_next [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] r_inh_next [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] v_exc_next [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] theta_next [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] vthr_next [0:N_NEURONS-1];
    (* ram_style = "block" *) logic [15:0]        refr_exc_next [0:N_NEURONS-1];

    (* ram_style = "block" *) logic signed [31:0] v_inh_next [0:N_NEURONS-1];
    (* ram_style = "block" *) logic [15:0]        refr_inh_next [0:N_NEURONS-1];

    (* ram_style = "block" *) logic [N_NEURONS-1:0] s_exc_next;
    (* ram_style = "block" *) logic signed [31:0] g_exc_next [0:N_NEURONS-1];

    logic [$clog2(N_NEURONS):0] neuron_idx;
    logic signed [63:0] sum_r_inh_reg;

    typedef enum logic [3:0] {
        P_GIN_DIV,
        P_I_SYN_EXC_MUL,
        P_I_SYN_INH_MUL,
        P_DV_EXC_DIV,
        P_THETA_DIV,
        P_R_EXC_DIV,
        P_X_EXC_DIV,
        P_G_EXC_MUL,
        P_I_SYN_I_MUL,
        P_DV_I_DIV,
        P_R_INH_DIV
    } calc_phase_e;
    calc_phase_e calc_phase;

    logic signed [31:0] mul_a;
    logic signed [31:0] mul_b;
    logic signed [31:0] mul_result;
    logic signed [31:0] div_a;
    logic [31:0]        div_div;
    logic signed [31:0] div_result;

    logic signed [31:0] num_exc_val;
    logic signed [31:0] num_i_val;
    logic signed [31:0] i_syn_exc_val;
    logic signed [31:0] v_next_exc_val;
    logic signed [31:0] theta_decayed_val;
    logic signed [31:0] r_exc_next_val;
    logic signed [31:0] x_exc_next_val;
    logic signed [31:0] g_exc_next_val;
    logic signed [31:0] v_next_i_val;
    logic signed [31:0] theta_next_val;
    logic signed [31:0] vthr_next_val;
    logic signed [31:0] v_exc_next_val;
    logic signed [31:0] v_inh_next_val;
    logic [15:0]        refr_exc_next_val;
    logic [15:0]        refr_inh_next_val;
    logic              s_exc_next_val;

    // Delay line RAM read ports (1-cycle latency)
    always_ff @(posedge clk) begin
        if (rst) begin
            delay_in_rd_data <= '0;
            delay_e2i_rd_data <= '0;
        end else begin
            delay_in_rd_data <= delay_in_mem[delay_in_rd_addr];
            delay_e2i_rd_data <= delay_e2i_mem[delay_e2i_rd_addr];
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
            state <= S_CLR_DELAY_IN;
            scan_in_idx <= '0;
            scan_group_idx <= '0;
            scan_spike_active <= 1'b0;
            stdp_pending <= 1'b0;
            stdp_j <= '0;
            stdp_g <= '0;
            neuron_idx <= '0;
            sum_r_inh_reg <= '0;
            calc_phase <= P_GIN_DIV;
            mul_a <= '0;
            mul_b <= '0;
            div_a <= '0;
            div_div <= '0;
            num_exc_val <= '0;
            num_i_val <= '0;
            i_syn_exc_val <= '0;
            v_next_exc_val <= '0;
            theta_decayed_val <= '0;
            r_exc_next_val <= '0;
            x_exc_next_val <= '0;
            g_exc_next_val <= '0;
            v_next_i_val <= '0;
            theta_next_val <= '0;
            vthr_next_val <= '0;
            v_exc_next_val <= '0;
            v_inh_next_val <= '0;
            refr_exc_next_val <= '0;
            refr_inh_next_val <= '0;
            s_exc_next_val <= 1'b0;

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
            end
            delay_in_wr_idx <= '0;
            delay_e2i_wr_idx <= '0;
            delay_in_clr_step <= '0;
            delay_e2i_clr_step <= '0;
            delay_clr_neuron <= '0;
            delay_in_rd_addr <= '0;
            delay_e2i_rd_addr <= '0;
        end else begin
            // defaults
            mem_r_en <= 1'b0;
            for (i = 0; i < LANES; i = i + 1) begin
                mem_w_en[i] <= 1'b0;
            end

            case (state)
                S_CLR_DELAY_IN: begin
                    s_tready <= 1'b0;
                    delay_in_mem[delay_in_addr(delay_in_clr_step, delay_clr_neuron)] <= '0;
                    if (delay_clr_neuron == N_NEURONS-1) begin
                        delay_clr_neuron <= '0;
                        if (delay_in_clr_step == DELAY_IN_STEPS-1) begin
                            delay_in_clr_step <= '0;
                            state <= S_CLR_DELAY_E2I;
                        end else begin
                            delay_in_clr_step <= delay_in_clr_step + 1'b1;
                        end
                    end else begin
                        delay_clr_neuron <= delay_clr_neuron + 1'b1;
                    end
                end

                S_CLR_DELAY_E2I: begin
                    s_tready <= 1'b0;
                    delay_e2i_mem[delay_e2i_addr(delay_e2i_clr_step, delay_clr_neuron)] <= '0;
                    if (delay_clr_neuron == N_NEURONS-1) begin
                        delay_clr_neuron <= '0;
                        if (delay_e2i_clr_step == DELAY_E2I_STEPS-1) begin
                            delay_e2i_clr_step <= '0;
                            state <= S_IDLE;
                        end else begin
                            delay_e2i_clr_step <= delay_e2i_clr_step + 1'b1;
                        end
                    end else begin
                        delay_clr_neuron <= delay_clr_neuron + 1'b1;
                    end
                end

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

                // Scan inputs; update x_in; if spike, read weights in groups of 4 neurons
                S_SCAN: begin
                    if (scan_in_idx < N_IN) begin
                        // update trace for this input
                        begin : trace_update
                            logic signed [31:0] x_next;
                            x_next = x_in[scan_in_idx] - fp_div_round(x_in[scan_in_idx], TD_X_STEPS)
                                   + (s_in_reg[scan_in_idx] ? TRACE_SPIKE_FP : 0);
                            x_in[scan_in_idx] <= x_next;
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
                        state <= S_NEURON_PREP;
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
                S_NEURON_PREP: begin
                    neuron_idx <= '0;
                    sum_r_inh_reg <= '0;
                    calc_phase <= P_GIN_DIV;
                    if (s_stdp_reg) begin
                        stdp_pending <= 1'b1;
                        stdp_j <= '0;
                        stdp_g <= '0;
                    end
                    state <= S_NEURON_CALC1;
                end

                S_NEURON_CALC1: begin
                    int n;
                    n = neuron_idx;
                    case (calc_phase)
                        P_GIN_DIV: begin
                            div_a <= g_in_state[n];
                            div_div <= TD_IN_STEPS;
                        end
                        P_I_SYN_EXC_MUL: begin
                            mul_a <= delay_in_rd_data;
                            mul_b <= (EXC_E_EXC*FP_SCALE) - v_exc[n];
                        end
                        P_I_SYN_INH_MUL: begin
                            mul_a <= g_inh_state[n];
                            mul_b <= (EXC_E_INH*FP_SCALE) - v_exc[n];
                        end
                        P_DV_EXC_DIV: begin
                            div_a <= num_exc_val;
                            div_div <= EXC_TAU_M;
                        end
                        P_THETA_DIV: begin
                            div_a <= theta[n];
                            div_div <= EXC_TC_THETA;
                        end
                        P_R_EXC_DIV: begin
                            div_a <= r_exc[n];
                            div_div <= TD_EXC_STEPS;
                        end
                        P_X_EXC_DIV: begin
                            div_a <= x_exc[n];
                            div_div <= TD_X_STEPS;
                        end
                        P_G_EXC_MUL: begin
                            mul_a <= WEXC_FP;
                            mul_b <= r_exc_next_val;
                        end
                        P_I_SYN_I_MUL: begin
                            mul_a <= delay_e2i_rd_data;
                            mul_b <= (INH_E_EXC*FP_SCALE) - v_inh[n];
                        end
                        P_DV_I_DIV: begin
                            div_a <= num_i_val;
                            div_div <= INH_TAU_M;
                        end
                        P_R_INH_DIV: begin
                            div_a <= r_inh[n];
                            div_div <= TD_INH_STEPS;
                        end
                        default: begin
                            div_a <= '0;
                            div_div <= '0;
                            mul_a <= '0;
                            mul_b <= '0;
                        end
                    endcase
                    state <= S_NEURON_CALC2;
                end

                S_NEURON_CALC2: begin
                    int n;
                    logic signed [31:0] theta_tmp;
                    logic s_exc_local;
                    logic s_inh_local;
                    n = neuron_idx;

                    case (calc_phase)
                        P_GIN_DIV: begin
                            g_in_state_next[n] <= g_in_state[n] - div_result + g_in_accum[n];
                            delay_in_rd_addr <= delay_in_addr(delay_in_rd_idx(), n[NEURON_W-1:0]);
                            calc_phase <= P_I_SYN_EXC_MUL;
                        end
                        P_I_SYN_EXC_MUL: begin
                            i_syn_exc_val <= mul_result;
                            calc_phase <= P_I_SYN_INH_MUL;
                        end
                        P_I_SYN_INH_MUL: begin
                            num_exc_val <= (EXC_VREST*FP_SCALE) - v_exc[n] + i_syn_exc_val + mul_result;
                            calc_phase <= P_DV_EXC_DIV;
                        end
                        P_DV_EXC_DIV: begin
                            v_next_exc_val <= v_exc[n] + div_result;
                            calc_phase <= P_THETA_DIV;
                        end
                        P_THETA_DIV: begin
                            theta_decayed_val <= theta[n] - div_result;
                            calc_phase <= P_R_EXC_DIV;
                        end
                        P_R_EXC_DIV: begin
                            if (refr_exc[n] != 0) begin
                                refr_exc_next_val <= refr_exc[n] - 1'b1;
                                v_exc_next_val <= EXC_VRESET * FP_SCALE;
                                theta_tmp = theta_decayed_val;
                                s_exc_local = 1'b0;
                            end else begin
                                if (v_next_exc_val >= vthr[n]) begin
                                    s_exc_local = 1'b1;
                                    v_exc_next_val <= EXC_VRESET * FP_SCALE;
                                    refr_exc_next_val <= EXC_REFRACT;
                                    theta_tmp = theta_decayed_val + EXC_THETA_PLUS_FP;
                                end else begin
                                    s_exc_local = 1'b0;
                                    v_exc_next_val <= v_next_exc_val;
                                    refr_exc_next_val <= 0;
                                    theta_tmp = theta_decayed_val;
                                end
                            end
                            if (theta_tmp < 0) theta_tmp = 0;
                            if (theta_tmp > (EXC_THETA_MAX*FP_SCALE)) theta_tmp = EXC_THETA_MAX*FP_SCALE;
                            theta_next_val <= theta_tmp;
                            vthr_next_val <= (EXC_INIT_VTHR * FP_SCALE) + theta_tmp;
                            s_exc_next_val <= s_exc_local;
                            r_exc_next_val <= r_exc[n] - div_result
                                            + (s_exc_local ? (FP_SCALE / TD_EXC_STEPS) : 0);
                            calc_phase <= P_X_EXC_DIV;
                        end
                        P_X_EXC_DIV: begin
                            x_exc_next_val <= x_exc[n] - div_result
                                            + (s_exc_next_val ? (FP_SCALE / TD_X_STEPS) : 0);
                            calc_phase <= P_G_EXC_MUL;
                        end
                        P_G_EXC_MUL: begin
                            g_exc_next_val <= mul_result;
                            delay_e2i_rd_addr <= delay_e2i_addr(delay_e2i_rd_idx(), n[NEURON_W-1:0]);
                            calc_phase <= P_I_SYN_I_MUL;
                        end
                        P_I_SYN_I_MUL: begin
                            num_i_val <= (INH_VREST*FP_SCALE) - v_inh[n] + mul_result;
                            calc_phase <= P_DV_I_DIV;
                        end
                        P_DV_I_DIV: begin
                            v_next_i_val <= v_inh[n] + div_result;
                            calc_phase <= P_R_INH_DIV;
                        end
                        P_R_INH_DIV: begin
                            logic signed [31:0] r_inh_local;
                            if (refr_inh[n] != 0) begin
                                refr_inh_next_val <= refr_inh[n] - 1'b1;
                                v_inh_next_val <= INH_VRESET * FP_SCALE;
                                s_inh_local = 1'b0;
                            end else begin
                                if (v_next_i_val >= (INH_VTHR*FP_SCALE)) begin
                                    s_inh_local = 1'b1;
                                    v_inh_next_val <= INH_VRESET * FP_SCALE;
                                    refr_inh_next_val <= INH_REFRACT;
                                end else begin
                                    s_inh_local = 1'b0;
                                    v_inh_next_val <= v_next_i_val;
                                    refr_inh_next_val <= 0;
                                end
                            end
                            r_inh_local = r_inh[n] - div_result
                                        + (s_inh_local ? (FP_SCALE / TD_INH_STEPS) : 0);

                            s_exc_next[n] <= s_exc_next_val;
                            v_exc_next[n] <= v_exc_next_val;
                            v_inh_next[n] <= v_inh_next_val;
                            refr_exc_next[n] <= refr_exc_next_val;
                            refr_inh_next[n] <= refr_inh_next_val;
                            theta_next[n] <= theta_next_val;
                            vthr_next[n] <= vthr_next_val;
                            r_exc_next[n] <= r_exc_next_val;
                            x_exc_next[n] <= x_exc_next_val;
                            g_exc_next[n] <= g_exc_next_val;
                            r_inh_next[n] <= r_inh_local;
                            sum_r_inh_reg <= sum_r_inh_reg + r_inh_local;

                            if (neuron_idx == N_NEURONS-1) begin
                                neuron_idx <= '0;
                                calc_phase <= P_GIN_DIV;
                                state <= S_NEURON_COMMIT;
                            end else begin
                                neuron_idx <= neuron_idx + 1'b1;
                                calc_phase <= P_GIN_DIV;
                                state <= S_NEURON_CALC1;
                            end
                        end
                        default: begin
                            calc_phase <= P_GIN_DIV;
                            state <= S_NEURON_CALC1;
                        end
                    endcase
                end

                S_NEURON_COMMIT: begin
                    int n;
                    logic signed [63:0] acc_inh;
                    logic signed [31:0] g_inh_next_val;
                    n = neuron_idx;
                    acc_inh = sum_r_inh_reg - r_inh_next[n];
                    if (N_NEURONS > 1) begin
                        g_inh_next_val = fp_mul(fp_div_round(WINH_FP, (N_NEURONS-1)), acc_inh[31:0]);
                    end else begin
                        g_inh_next_val = 0;
                    end
                    g_in_state[n] <= g_in_state_next[n];
                    r_exc[n] <= r_exc_next[n];
                    x_exc[n] <= x_exc_next[n];
                    r_inh[n] <= r_inh_next[n];
                    v_exc[n] <= v_exc_next[n];
                    theta[n] <= theta_next[n];
                    vthr[n] <= vthr_next[n];
                    refr_exc[n] <= refr_exc_next[n];
                    v_inh[n] <= v_inh_next[n];
                    refr_inh[n] <= refr_inh_next[n];
                    g_inh_state[n] <= g_inh_next_val;
                    delay_in_mem[delay_in_addr(delay_in_wr_idx, n[NEURON_W-1:0])] <= g_in_state_next[n];
                    delay_e2i_mem[delay_e2i_addr(delay_e2i_wr_idx, n[NEURON_W-1:0])] <= g_exc_next[n];

                    if (neuron_idx == N_NEURONS-1) begin
                        delay_in_wr_idx <= delay_in_next_idx(delay_in_wr_idx);
                        delay_e2i_wr_idx <= delay_e2i_next_idx(delay_e2i_wr_idx);

                        m_tdata <= {tstep_id_reg, s_exc_next};
                        m_tvalid <= 1'b1;
                        state <= S_OUT;
                        neuron_idx <= '0;
                    end else begin
                        neuron_idx <= neuron_idx + 1'b1;
                    end
                end

        S_OUT: begin
            if (m_tvalid && m_tready) begin
                m_tvalid <= 1'b0;
                if (stdp_pending) begin
                    stdp_pending <= 1'b0;
                    // start online STDP update over all weights
                    mem_r_en <= 1'b1;
                    for (i = 0; i < LANES; i = i + 1) begin
                        mem_r_neuron[i] <= stdp_g * LANES + i;
                        mem_r_in[i] <= stdp_j;
                    end
                    state <= S_STDP_READ;
                end else begin
                    state <= S_IDLE;
                end
            end
        end

        // Online STDP update (4 weights per cycle)
        S_STDP_READ: begin
            state <= S_STDP_CALC;
        end

        S_STDP_CALC: begin
            for (i = 0; i < LANES; i = i + 1) begin
                int neuron_idx;
                logic signed [31:0] w_old;
                logic signed [31:0] dW;
                logic signed [31:0] w_new;
                logic signed [31:0] pre_term;
                logic signed [31:0] post_term;

                neuron_idx = stdp_g * LANES + i;
                if (neuron_idx < N_NEURONS) begin
                    w_old = mem_r_data[i];

                    pre_term = s_exc_next[neuron_idx] ? fp_mul(A_P_FP, x_in[stdp_j]) : 0;
                    post_term = s_in_reg[stdp_j] ? fp_mul(A_M_FP, x_exc_next[neuron_idx]) : 0;
                    dW = pre_term - post_term;

                    w_new = w_old + dW;
                    if (w_new < WMIN_FP) w_new = WMIN_FP;
                    if (w_new > WMAX_FP) w_new = WMAX_FP;

                    mem_w_en[i] <= 1'b1;
                    mem_w_neuron[i] <= neuron_idx[NEURON_W-1:0];
                    mem_w_in[i] <= stdp_j;
                    mem_w_data[i] <= w_new;
                end
            end

            if (stdp_g == NEURON_GROUPS-1) begin
                stdp_g <= '0;
                if (stdp_j == N_IN-1) begin
                    state <= S_IDLE;
                end else begin
                    stdp_j <= stdp_j + 1'b1;
                    mem_r_en <= 1'b1;
                    for (i = 0; i < LANES; i = i + 1) begin
                        mem_r_neuron[i] <= (stdp_g) * LANES + i;
                        mem_r_in[i] <= stdp_j + 1'b1;
                    end
                    state <= S_STDP_READ;
                end
            end else begin
                stdp_g <= stdp_g + 1'b1;
                mem_r_en <= 1'b1;
                for (i = 0; i < LANES; i = i + 1) begin
                    mem_r_neuron[i] <= (stdp_g + 1'b1) * LANES + i;
                    mem_r_in[i] <= stdp_j;
                end
                state <= S_STDP_READ;
            end
        end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
