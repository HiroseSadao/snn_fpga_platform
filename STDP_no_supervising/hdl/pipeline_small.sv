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
    localparam int W_W = 18; // S2.16 weights

    // Network params (match LIF_WTA_STDP_MNIST.py)
    localparam int TD_IN_STEPS   = 1;   // 1e-3 / 1e-3
    localparam int TD_EXC_STEPS  = 1;   // 1e-3 / 1e-3
    localparam int TD_INH_STEPS  = 2;   // 2e-3 / 1e-3
    localparam int TD_X_STEPS    = 20;  // 2e-2 / 1e-3
    localparam int TD_X2_STEPS   = 40;  // 4e-2 / 1e-3 (post2)
    localparam int DELAY_IN_STEPS = 11; // 0..10 ms random delay (approx)
    localparam int DELAY_E2I_STEPS = 2; // 2e-3 / 1e-3

    localparam int WEXC_FP = 147456; // 2.25
    localparam int WINH_FP = 57344;  // 0.875

    localparam int WMIN_FP = 0;
    localparam int WMAX_FP = 3277;   // 0.05
    // Online STDP (Brian2 Diehl_Cook_2015): A_p=0.01, A_m=0.0001
    localparam int A_P_FP = 655;     // 0.01
    localparam int A_M_FP = 7;       // 0.0001

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

    localparam int LANES = 1;
    localparam int NEURON_GROUPS = (N_NEURONS + LANES - 1) / LANES;

    localparam int INPUT_SPIKE_FP = FP_SCALE / TD_IN_STEPS;
    localparam int TRACE_SPIKE_FP = FP_SCALE; // event-driven traces set to 1.0 on spike
    localparam int EXC_SPIKE_FP = (TD_EXC_STEPS == 1) ? FP_SCALE
                                  : (TD_EXC_STEPS == 2) ? (FP_SCALE >>> 1)
                                  : (FP_SCALE / TD_EXC_STEPS);
    localparam int INH_SPIKE_FP = (TD_INH_STEPS == 1) ? FP_SCALE
                                  : (TD_INH_STEPS == 2) ? (FP_SCALE >>> 1)
                                  : (FP_SCALE / TD_INH_STEPS);
    localparam longint unsigned Q32_ONE = 64'h1_0000_0000;
    localparam int INV_TD_X_Q32       = (TD_X_STEPS <= 1) ? 32'hFFFF_FFFF
                                 : (Q32_ONE + (TD_X_STEPS/2)) / TD_X_STEPS;
    localparam int INV_TD_X2_Q32      = (TD_X2_STEPS <= 1) ? 32'hFFFF_FFFF
                                 : (Q32_ONE + (TD_X2_STEPS/2)) / TD_X2_STEPS;
    localparam int INV_EXC_TAU_M_Q32  = (EXC_TAU_M <= 1) ? 32'hFFFF_FFFF
                                 : (Q32_ONE + (EXC_TAU_M/2)) / EXC_TAU_M;
    localparam int INV_INH_TAU_M_Q32  = (INH_TAU_M <= 1) ? 32'hFFFF_FFFF
                                 : (Q32_ONE + (INH_TAU_M/2)) / INH_TAU_M;
    localparam int INV_TC_THETA_Q32   = (EXC_TC_THETA <= 1) ? 32'hFFFF_FFFF
                                 : (Q32_ONE + (EXC_TC_THETA/2)) / EXC_TC_THETA;
    localparam int WINH_DIV_FP = (N_NEURONS > 1)
                                 ? ((WINH_FP + ((N_NEURONS-1) >> 1)) / (N_NEURONS-1))
                                 : 0;

    // State
    integer i, j, t;

    logic [TSTEP_W-1:0] tstep_id_reg;
    logic [N_IN-1:0] s_in_reg;
    logic s_stdp_reg;

    // x_in is explicit 1R1W RAM (XPM in synthesis)
    localparam int X_IN_ADDR_W = (N_IN <= 1) ? 1 : $clog2(N_IN);
    logic x_in_rd_en;
    logic [X_IN_ADDR_W-1:0] x_in_rd_addr;
    logic signed [31:0] x_in_rd_data;
    logic x_in_wr_en;
    logic [X_IN_ADDR_W-1:0] x_in_wr_addr;
    logic signed [31:0] x_in_wr_data;
    logic [X_IN_ADDR_W-1:0] x_in_clr_idx;

    // g_in_accum is kept as LUTRAM/FF (distributed)
    (* ram_style = "distributed" *) logic signed [31:0] g_in_accum [0:N_NEURONS-1];

    // Packed per-neuron state RAM
    localparam int STATE_W = (32*10) + 32; // 10x32 + 2x16
    logic [STATE_W-1:0] state_wr_data;
    logic [STATE_W-1:0] state_rd_data;
    logic state_wr_en;
    logic state_rd_en;
    logic [NEURON_W-1:0] state_wr_addr;
    logic [NEURON_W-1:0] state_rd_addr;

    wire signed [31:0] st_g_in_state = state_rd_data[STATE_W-1   -: 32];
    wire signed [31:0] st_r_exc      = state_rd_data[STATE_W-33  -: 32];
    wire signed [31:0] st_x_exc      = state_rd_data[STATE_W-65  -: 32];
    wire signed [31:0] st_x_exc2     = state_rd_data[STATE_W-97  -: 32];
    wire signed [31:0] st_g_inh_state= state_rd_data[STATE_W-129 -: 32];
    wire signed [31:0] st_r_inh      = state_rd_data[STATE_W-161 -: 32];
    wire signed [31:0] st_v_exc      = state_rd_data[STATE_W-193 -: 32];
    wire signed [31:0] st_v_inh      = state_rd_data[STATE_W-225 -: 32];
    wire signed [31:0] st_theta      = state_rd_data[STATE_W-257 -: 32];
    wire signed [31:0] st_vthr       = state_rd_data[STATE_W-289 -: 32];
    wire [15:0]        st_refr_exc   = state_rd_data[STATE_W-321 -: 16];
    wire [15:0]        st_refr_inh   = state_rd_data[STATE_W-337 -: 16];

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
    logic delay_in_wr_en;
    logic delay_e2i_wr_en;
    logic [DELAY_IN_ADDR_W-1:0] delay_in_wr_addr;
    logic [DELAY_E2I_ADDR_W-1:0] delay_e2i_wr_addr;
    logic signed [31:0] delay_in_wr_data;
    logic signed [31:0] delay_e2i_wr_data;
    localparam int DELAY_IN_W = (DELAY_IN_STEPS <= 1) ? 1 : $clog2(DELAY_IN_STEPS);
    localparam int DELAY_E2I_W = (DELAY_E2I_STEPS <= 1) ? 1 : $clog2(DELAY_E2I_STEPS);
    logic [DELAY_IN_W-1:0] delay_in_wr_idx;
    logic [DELAY_E2I_W-1:0] delay_e2i_wr_idx;
    logic [DELAY_IN_W-1:0] delay_in_clr_step;
    logic [DELAY_E2I_W-1:0] delay_e2i_clr_step;
    logic [NEURON_W-1:0] delay_clr_neuron;

    // LFSR for random input delay selection (0..10 ms)
    logic [15:0] lfsr;
    logic [DELAY_IN_W-1:0] delay_in_rand;

    function automatic [15:0] lfsr_next(input [15:0] v);
        begin
            lfsr_next = {v[14:0], v[15]^v[13]^v[12]^v[10]};
        end
    endfunction

    logic signed [31:0] g_inh_next_val_reg;

    function automatic [DELAY_IN_W-1:0] delay_in_rd_idx_rand(
        input [DELAY_IN_W-1:0] wr_idx,
        input [DELAY_IN_W-1:0] delay
    );
        begin
            if (DELAY_IN_STEPS <= 1) begin
                delay_in_rd_idx_rand = '0;
            end else if (delay == 0) begin
                // Avoid reading the slot being written this step
                delay_in_rd_idx_rand = (wr_idx == 0) ? DELAY_IN_STEPS-1 : (wr_idx - 1'b1);
            end else if (wr_idx >= delay) begin
                delay_in_rd_idx_rand = wr_idx - delay;
            end else begin
                delay_in_rd_idx_rand = wr_idx + DELAY_IN_STEPS - delay;
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

    typedef enum logic [5:0] {
        S_WAIT_WMEM_INIT,
        S_INIT_STATE_MEM,
        S_IDLE,
        S_SCAN,
        S_SCAN_WAIT,
        S_GIN_WAIT,
        S_NEURON_PREP,
        S_NEURON_RD_WAIT,
        S_NEURON_CALC1,
        S_NEURON_CALC2,
        S_NEURON_COMMIT_RD,
        S_NEURON_COMMIT_WAIT,
        S_NEURON_COMMIT,
        S_NEURON_COMMIT_APPLY,
        S_OUT,
        S_STDP_SCAN_PRE,
        S_STDP_SCAN_POST,
        S_STDP_READ,
        S_STDP_WAIT,
        S_STDP_CALC,
        S_CLR_DELAY_IN,
        S_CLR_DELAY_E2I,
        S_CLR_XIN
    } state_e;
    state_e state;

    function automatic signed [31:0] fp_mul(input signed [31:0] a, input signed [31:0] b);
        begin
            fp_mul = (a * b) >>> FP_SHIFT;
        end
    endfunction

    function automatic signed [31:0] div_const_q32(
        input signed [31:0] a,
        input logic [31:0] inv_q32
    );
        longint signed prod;
        begin
            prod = $signed(a) * $signed({1'b0, inv_q32});
            div_const_q32 = prod >>> 32;
        end
    endfunction

    // Packed state RAM (XPM in synthesis, behavioral in sim)
`ifdef SYNTHESIS
    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(NEURON_W),
        .ADDR_WIDTH_B(NEURON_W),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(STATE_W),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(N_NEURONS * STATE_W),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(STATE_W),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(STATE_W),
        .WRITE_MODE_B("read_first")
    ) u_state_mem (
        .clka(clk),
        .ena(state_wr_en),
        .wea(state_wr_en),
        .addra(state_wr_addr),
        .dina(state_wr_data),
        .clkb(clk),
        .enb(state_rd_en),
        .addrb(state_rd_addr),
        .doutb(state_rd_data),
        .rstb(rst),
        .regceb(1'b1),
        .sleep(1'b0),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0)
    );
`else
    logic [STATE_W-1:0] state_mem_sim [0:N_NEURONS-1];
    always_ff @(posedge clk) begin
        if (rst) begin
            state_rd_data <= '0;
        end else begin
            if (state_wr_en) begin
                state_mem_sim[state_wr_addr] <= state_wr_data;
            end
            if (state_rd_en) begin
                state_rd_data <= state_mem_sim[state_rd_addr];
            end
        end
    end
`endif

    // Memory interface for W_in (4-bank)
    logic mem_r_en;
    logic [NEURON_W-1:0] mem_r_neuron [0:LANES-1];
    logic [IN_W-1:0] mem_r_in [0:LANES-1];
    logic signed [W_W-1:0] mem_r_data [0:LANES-1];

    logic mem_w_en [0:LANES-1];
    logic [NEURON_W-1:0] mem_w_neuron [0:LANES-1];
    logic [IN_W-1:0] mem_w_in [0:LANES-1];
    logic signed [W_W-1:0] mem_w_data [0:LANES-1];

    logic wmem_init_done;

    w_in_mem_4bank #(
        .N_IN(N_IN),
        .N_NEURONS(N_NEURONS),
        .W_W(W_W),
        .INIT_VAL(18'sd66),
        .INIT_FROM_FILE(W_INIT_FROM_FILE)
    ) u_wmem (
        .clk(clk),
        .rst(rst),
        .r_en(mem_r_en),
        .r_neuron0(mem_r_neuron[0]),
        .r_in0(mem_r_in[0]),
        .r_data0(mem_r_data[0]),
        .w_en0(mem_w_en[0]),
        .w_neuron0(mem_w_neuron[0]),
        .w_in0(mem_w_in[0]),
        .w_data0(mem_w_data[0]),
        .dbg_en(dbg_en),
        .dbg_neuron(dbg_neuron),
        .dbg_in(dbg_in),
        .dbg_valid(dbg_valid),
        .dbg_data(dbg_data),
        .init_done(wmem_init_done)
    );

    // Scan and STDP counters
    logic [IN_W-1:0] scan_in_idx;
    logic [$clog2(NEURON_GROUPS):0] scan_group_idx;
    logic scan_spike_active;

    logic stdp_pending;
    logic [IN_W-1:0] stdp_j;
    logic [$clog2(NEURON_GROUPS):0] stdp_g;
    localparam int PRE_CNT_W = (N_IN + 1 <= 1) ? 1 : $clog2(N_IN + 1);
    localparam int POST_CNT_W = (N_NEURONS + 1 <= 1) ? 1 : $clog2(N_NEURONS + 1);
    (* ram_style = "block" *) logic [IN_W-1:0] stdp_pre_idx [0:N_IN-1];
    (* ram_style = "block" *) logic [NEURON_W-1:0] stdp_post_idx [0:N_NEURONS-1];
    logic [PRE_CNT_W-1:0] stdp_pre_count;
    logic [POST_CNT_W-1:0] stdp_post_count;
    logic [IN_W-1:0] stdp_scan_in;
    logic [NEURON_W-1:0] stdp_scan_neuron;
    logic [PRE_CNT_W-1:0] stdp_pre_i;
    logic [POST_CNT_W-1:0] stdp_post_i;
    logic stdp_mode_post;

    logic [N_NEURONS-1:0] s_exc_next;

    logic [$clog2(N_NEURONS):0] neuron_idx;
    logic [NEURON_W-1:0] state_init_idx;
    logic signed [63:0] sum_r_inh_reg;

    typedef enum logic [3:0] {
        P_GIN_DIV,
        P_I_SYN_EXC_MUL,
        P_I_SYN_INH_MUL,
        P_DV_EXC_DIV,
        P_THETA_DIV,
        P_R_EXC_DIV,
        P_X_EXC_DIV,
        P_X_EXC2_DIV,
        P_G_EXC_MUL,
        P_I_SYN_I_MUL,
        P_DV_I_DIV,
        P_R_INH_DIV
    } calc_phase_e;
    calc_phase_e calc_phase;

    logic signed [31:0] mul_a;
    logic signed [31:0] mul_b;
    wire  signed [31:0] mul_result = (mul_a * mul_b) >>> FP_SHIFT;
    logic signed [31:0] div_result;

    logic signed [31:0] num_exc_val;
    logic signed [31:0] num_i_val;
    logic signed [31:0] i_syn_exc_val;
    logic signed [31:0] g_in_state_next_val;
    logic signed [31:0] v_next_exc_val;
    logic signed [31:0] theta_decayed_val;
    logic signed [31:0] r_exc_next_val;
    logic signed [31:0] x_exc_next_val;
    logic signed [31:0] x_exc2_next_val;
    logic signed [31:0] g_exc_next_val;
    logic signed [31:0] v_next_i_val;
    logic signed [31:0] theta_next_val;
    logic signed [31:0] vthr_next_val;
    logic signed [31:0] v_exc_next_val;
    logic signed [31:0] v_inh_next_val;
    logic [15:0]        refr_exc_next_val;
    logic [15:0]        refr_inh_next_val;
    logic              s_exc_next_val;

    // x_in RAM (XPM in synthesis, behavioral in sim)
`ifdef SYNTHESIS
    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(X_IN_ADDR_W),
        .ADDR_WIDTH_B(X_IN_ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(32),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(N_IN * 32),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(32),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(32),
        .WRITE_MODE_B("read_first")
    ) u_x_in (
        .clka(clk),
        .ena(x_in_wr_en),
        .wea(x_in_wr_en),
        .addra(x_in_wr_addr),
        .dina(x_in_wr_data),
        .clkb(clk),
        .enb(x_in_rd_en),
        .addrb(x_in_rd_addr),
        .doutb(x_in_rd_data),
        .rstb(rst),
        .regceb(1'b1),
        .sleep(1'b0),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0)
    );
`else
    logic signed [31:0] x_in_mem_sim [0:N_IN-1];
    always_ff @(posedge clk) begin
        if (x_in_wr_en) begin
            x_in_mem_sim[x_in_wr_addr] <= x_in_wr_data;
        end
        if (x_in_rd_en) begin
            x_in_rd_data <= x_in_mem_sim[x_in_rd_addr];
        end
    end
`endif

    // Delay line RAM (XPM in synthesis, behavioral in sim)
`ifdef SYNTHESIS
    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(DELAY_IN_ADDR_W),
        .ADDR_WIDTH_B(DELAY_IN_ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(32),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(DELAY_IN_DEPTH * 32),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(32),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(32),
        .WRITE_MODE_B("read_first")
    ) u_delay_in (
        .clka(clk),
        .ena(delay_in_wr_en),
        .wea(delay_in_wr_en),
        .addra(delay_in_wr_addr),
        .dina(delay_in_wr_data),
        .clkb(clk),
        .enb(1'b1),
        .addrb(delay_in_rd_addr),
        .doutb(delay_in_rd_data),
        .rstb(rst),
        .regceb(1'b1),
        .sleep(1'b0),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0)
    );

    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(DELAY_E2I_ADDR_W),
        .ADDR_WIDTH_B(DELAY_E2I_ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(32),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(DELAY_E2I_DEPTH * 32),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(32),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(32),
        .WRITE_MODE_B("read_first")
    ) u_delay_e2i (
        .clka(clk),
        .ena(delay_e2i_wr_en),
        .wea(delay_e2i_wr_en),
        .addra(delay_e2i_wr_addr),
        .dina(delay_e2i_wr_data),
        .clkb(clk),
        .enb(1'b1),
        .addrb(delay_e2i_rd_addr),
        .doutb(delay_e2i_rd_data),
        .rstb(rst),
        .regceb(1'b1),
        .sleep(1'b0),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0)
    );
`else
    // Behavioral delay memories
    always_ff @(posedge clk) begin
        if (rst) begin
            delay_in_rd_data <= '0;
            delay_e2i_rd_data <= '0;
        end else begin
            if (delay_in_wr_en) begin
                delay_in_mem[delay_in_wr_addr] <= delay_in_wr_data;
            end
            if (delay_e2i_wr_en) begin
                delay_e2i_mem[delay_e2i_wr_addr] <= delay_e2i_wr_data;
            end
            delay_in_rd_data <= delay_in_mem[delay_in_rd_addr];
            delay_e2i_rd_data <= delay_e2i_mem[delay_e2i_rd_addr];
        end
    end
`endif

    // (next_mem removed; state_mem is used for both current and next state)
    // Sequential state updates
    always_ff @(posedge clk) begin
        if (rst) begin
            s_tready <= 1'b1;
            m_tvalid <= 1'b0;
            m_tdata <= '0;
            tstep_id_reg <= '0;
            s_in_reg <= '0;
            s_stdp_reg <= 1'b0;
            state <= S_WAIT_WMEM_INIT;
            scan_in_idx <= '0;
            scan_group_idx <= '0;
            scan_spike_active <= 1'b0;
            stdp_pending <= 1'b0;
            stdp_j <= '0;
            stdp_g <= '0;
            stdp_pre_count <= '0;
            stdp_post_count <= '0;
            stdp_scan_in <= '0;
            stdp_scan_neuron <= '0;
            stdp_pre_i <= '0;
            stdp_post_i <= '0;
            stdp_mode_post <= 1'b0;
            neuron_idx <= '0;
            sum_r_inh_reg <= '0;
            calc_phase <= P_GIN_DIV;
            mul_a <= '0;
            mul_b <= '0;
            div_result <= '0;
            g_inh_next_val_reg <= '0;
            state_wr_en <= 1'b0;
            state_rd_en <= 1'b0;
            state_wr_addr <= '0;
            state_rd_addr <= '0;
            state_wr_data <= '0;
            state_init_idx <= '0;
            num_exc_val <= '0;
            num_i_val <= '0;
            i_syn_exc_val <= '0;
            g_in_state_next_val <= '0;
            v_next_exc_val <= '0;
            theta_decayed_val <= '0;
            r_exc_next_val <= '0;
            x_exc_next_val <= '0;
            x_exc2_next_val <= '0;
            g_exc_next_val <= '0;
            v_next_i_val <= '0;
            theta_next_val <= '0;
            vthr_next_val <= '0;
            v_exc_next_val <= '0;
            v_inh_next_val <= '0;
            refr_exc_next_val <= '0;
            refr_inh_next_val <= '0;
            s_exc_next_val <= 1'b0;
            x_in_rd_en <= 1'b0;
            x_in_rd_addr <= '0;
            x_in_wr_en <= 1'b0;
            x_in_wr_addr <= '0;
            x_in_wr_data <= '0;
            x_in_clr_idx <= '0;
            for (i = 0; i < N_NEURONS; i = i + 1) begin
                g_in_accum[i] <= '0;
                s_exc_next[i] <= 1'b0;
            end
            delay_in_wr_idx <= '0;
            delay_e2i_wr_idx <= '0;
            delay_in_clr_step <= '0;
            delay_e2i_clr_step <= '0;
            delay_clr_neuron <= '0;
            delay_in_rd_addr <= '0;
            delay_e2i_rd_addr <= '0;
            delay_in_wr_en <= 1'b0;
            delay_e2i_wr_en <= 1'b0;
            delay_in_wr_addr <= '0;
            delay_e2i_wr_addr <= '0;
            delay_in_wr_data <= '0;
            delay_e2i_wr_data <= '0;
            lfsr <= 16'hACE1;
            delay_in_rand <= '0;
        end else begin
            // defaults
            mem_r_en <= 1'b0;
            x_in_rd_en <= 1'b0;
            x_in_wr_en <= 1'b0;
            delay_in_wr_en <= 1'b0;
            delay_e2i_wr_en <= 1'b0;
            state_wr_en <= 1'b0;
            state_rd_en <= 1'b0;
            for (i = 0; i < LANES; i = i + 1) begin
                mem_w_en[i] <= 1'b0;
            end

            case (state)
                S_WAIT_WMEM_INIT: begin
                    s_tready <= 1'b0;
                    if (wmem_init_done) begin
                        state_init_idx <= '0;
                        state <= S_INIT_STATE_MEM;
                    end
                end

                S_INIT_STATE_MEM: begin
                    s_tready <= 1'b0;
                    state_wr_en <= 1'b1;
                    state_wr_addr <= state_init_idx;
                    state_wr_data <= {
                        32'sd0,                             // g_in_state
                        32'sd0,                             // r_exc
                        32'sd0,                             // x_exc
                        32'sd0,                             // x_exc2
                        32'sd0,                             // g_inh_state
                        32'sd0,                             // r_inh
                        EXC_VRESET * FP_SCALE,              // v_exc
                        INH_VRESET * FP_SCALE,              // v_inh
                        32'sd0,                             // theta
                        EXC_INIT_VTHR * FP_SCALE,           // vthr
                        16'd0,                              // refr_exc
                        16'd0                               // refr_inh
                    };
                    if (state_init_idx == N_NEURONS-1) begin
                        state_init_idx <= '0;
                        state <= S_CLR_DELAY_IN;
                    end else begin
                        state_init_idx <= state_init_idx + 1'b1;
                    end
                end

                S_CLR_DELAY_IN: begin
                    s_tready <= 1'b0;
                    delay_in_wr_en <= 1'b1;
                    delay_in_wr_addr <= delay_in_addr(delay_in_clr_step, delay_clr_neuron);
                    delay_in_wr_data <= '0;
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
                    delay_e2i_wr_en <= 1'b1;
                    delay_e2i_wr_addr <= delay_e2i_addr(delay_e2i_clr_step, delay_clr_neuron);
                    delay_e2i_wr_data <= '0;
                    if (delay_clr_neuron == N_NEURONS-1) begin
                        delay_clr_neuron <= '0;
                        if (delay_e2i_clr_step == DELAY_E2I_STEPS-1) begin
                            delay_e2i_clr_step <= '0;
                            state <= S_CLR_XIN;
                        end else begin
                            delay_e2i_clr_step <= delay_e2i_clr_step + 1'b1;
                        end
                    end else begin
                        delay_clr_neuron <= delay_clr_neuron + 1'b1;
                    end
                end

                S_CLR_XIN: begin
                    s_tready <= 1'b0;
                    x_in_wr_en <= 1'b1;
                    x_in_wr_addr <= x_in_clr_idx;
                    x_in_wr_data <= '0;
                    if (x_in_clr_idx == N_IN-1) begin
                        x_in_clr_idx <= '0;
                        state <= S_IDLE;
                    end else begin
                        x_in_clr_idx <= x_in_clr_idx + 1'b1;
                    end
                end

                S_IDLE: begin
                    s_tready <= 1'b1;
                    if (s_tvalid && s_tready) begin
                        tstep_id_reg <= s_tdata[TSTEP_W+N_IN-1 -: TSTEP_W];
                        s_in_reg <= s_tdata[N_IN-1:0];
                        s_stdp_reg <= s_stdp_en;
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

                // Scan inputs; update x_in; if spike, read weights
                S_SCAN: begin
                    if (scan_in_idx < N_IN) begin
                        x_in_rd_en <= 1'b1;
                        x_in_rd_addr <= scan_in_idx;
                        state <= S_SCAN_WAIT;
                    end else begin
                        state <= S_NEURON_PREP;
                    end
                end

                S_SCAN_WAIT: begin
                    logic signed [31:0] scan_val;
                    logic signed [31:0] div_signed;
                    logic signed [31:0] decayed;
                    scan_val = x_in_rd_data;
                    div_signed = div_const_q32(scan_val, INV_TD_X_Q32);
                    x_in_wr_en <= 1'b1;
                    x_in_wr_addr <= scan_in_idx;
                    if (s_in_reg[scan_in_idx]) begin
                        x_in_wr_data <= TRACE_SPIKE_FP;
                    end else begin
                        decayed = scan_val - div_signed;
                        if (decayed < 0) decayed = 0;
                        x_in_wr_data <= decayed;
                    end

                    if (s_in_reg[scan_in_idx]) begin
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
                        state <= S_SCAN;
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
                                    + fp_mul($signed(mem_r_data[i]), INPUT_SPIKE_FP);
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
                    state_rd_en <= 1'b1;
                    state_rd_addr <= '0;
                    state <= S_NEURON_RD_WAIT;
                end

                S_NEURON_RD_WAIT: begin
                    // wait one cycle for state RAM read
                    state <= S_NEURON_CALC1;
                end

                S_NEURON_CALC1: begin
                    int n;
                    n = neuron_idx;
                    case (calc_phase)
                        P_GIN_DIV: begin
                            if (TD_IN_STEPS <= 1) begin
                                div_result <= st_g_in_state;
                            end else if (TD_IN_STEPS == 2) begin
                                div_result <= st_g_in_state >>> 1;
                            end else begin
                                div_result <= div_const_q32(st_g_in_state,
                                                            (Q32_ONE + (TD_IN_STEPS/2)) / TD_IN_STEPS);
                            end
                            delay_in_rand <= lfsr % DELAY_IN_STEPS;
                            lfsr <= lfsr_next(lfsr);
                            state <= S_NEURON_CALC2;
                        end
                        P_I_SYN_EXC_MUL: begin
                            mul_a <= delay_in_rd_data;
                            mul_b <= (EXC_E_EXC*FP_SCALE) - st_v_exc;
                            state <= S_NEURON_CALC2;
                        end
                        P_I_SYN_INH_MUL: begin
                            mul_a <= st_g_inh_state;
                            mul_b <= (EXC_E_INH*FP_SCALE) - st_v_exc;
                            state <= S_NEURON_CALC2;
                        end
                        P_DV_EXC_DIV: begin
                            div_result <= div_const_q32(num_exc_val, INV_EXC_TAU_M_Q32);
                            state <= S_NEURON_CALC2;
                        end
                        P_THETA_DIV: begin
                            div_result <= div_const_q32(st_theta, INV_TC_THETA_Q32);
                            state <= S_NEURON_CALC2;
                        end
                        P_R_EXC_DIV: begin
                            if (TD_EXC_STEPS <= 1) begin
                                div_result <= st_r_exc;
                            end else if (TD_EXC_STEPS == 2) begin
                                div_result <= st_r_exc >>> 1;
                            end else begin
                                div_result <= div_const_q32(st_r_exc,
                                                            (Q32_ONE + (TD_EXC_STEPS/2)) / TD_EXC_STEPS);
                            end
                            state <= S_NEURON_CALC2;
                        end
                        P_X_EXC_DIV: begin
                            div_result <= div_const_q32(st_x_exc, INV_TD_X_Q32);
                            state <= S_NEURON_CALC2;
                        end
                        P_X_EXC2_DIV: begin
                            div_result <= div_const_q32(st_x_exc2, INV_TD_X2_Q32);
                            state <= S_NEURON_CALC2;
                        end
                        P_G_EXC_MUL: begin
                            mul_a <= WEXC_FP;
                            mul_b <= r_exc_next_val;
                            state <= S_NEURON_CALC2;
                        end
                        P_I_SYN_I_MUL: begin
                            mul_a <= delay_e2i_rd_data;
                            mul_b <= (INH_E_EXC*FP_SCALE) - st_v_inh;
                            state <= S_NEURON_CALC2;
                        end
                        P_DV_I_DIV: begin
                            div_result <= div_const_q32(num_i_val, INV_INH_TAU_M_Q32);
                            state <= S_NEURON_CALC2;
                        end
                        P_R_INH_DIV: begin
                            if (TD_INH_STEPS <= 1) begin
                                div_result <= st_r_inh;
                            end else if (TD_INH_STEPS == 2) begin
                                div_result <= st_r_inh >>> 1;
                            end else begin
                                div_result <= div_const_q32(st_r_inh,
                                                            (Q32_ONE + (TD_INH_STEPS/2)) / TD_INH_STEPS);
                            end
                            state <= S_NEURON_CALC2;
                        end
                        default: begin
                            mul_a <= '0;
                            mul_b <= '0;
                            state <= S_NEURON_CALC2;
                        end
                    endcase
                end

                S_NEURON_CALC2: begin
                    int n;
                    logic signed [31:0] theta_tmp;
                    logic s_exc_local;
                    logic s_inh_local;
                    n = neuron_idx;

                    case (calc_phase)
                        P_GIN_DIV: begin
                            g_in_state_next_val <= st_g_in_state - div_result + g_in_accum[n];
                            delay_in_rd_addr <= delay_in_addr(
                                delay_in_rd_idx_rand(delay_in_wr_idx, delay_in_rand),
                                n[NEURON_W-1:0]
                            );
                            calc_phase <= P_I_SYN_EXC_MUL;
                        end
                        P_I_SYN_EXC_MUL: begin
                            i_syn_exc_val <= mul_result;
                            calc_phase <= P_I_SYN_INH_MUL;
                        end
                        P_I_SYN_INH_MUL: begin
                            num_exc_val <= (EXC_VREST*FP_SCALE) - st_v_exc + i_syn_exc_val + mul_result;
                            calc_phase <= P_DV_EXC_DIV;
                        end
                        P_DV_EXC_DIV: begin
                            v_next_exc_val <= st_v_exc + div_result;
                            calc_phase <= P_THETA_DIV;
                        end
                        P_THETA_DIV: begin
                            theta_decayed_val <= st_theta - div_result;
                            calc_phase <= P_R_EXC_DIV;
                        end
                        P_R_EXC_DIV: begin
                            if (st_refr_exc != 0) begin
                                refr_exc_next_val <= st_refr_exc - 1'b1;
                                v_exc_next_val <= EXC_VRESET * FP_SCALE;
                                theta_tmp = theta_decayed_val;
                                s_exc_local = 1'b0;
                            end else begin
                                if (v_next_exc_val >= st_vthr) begin
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
                            r_exc_next_val <= st_r_exc - div_result
                                            + (s_exc_local ? EXC_SPIKE_FP : 0);
                            calc_phase <= P_X_EXC_DIV;
                        end
                        P_X_EXC_DIV: begin
                            if (s_exc_next_val) begin
                                x_exc_next_val <= TRACE_SPIKE_FP;
                            end else begin
                                logic signed [31:0] decayed_x;
                                decayed_x = st_x_exc - div_result;
                                if (decayed_x < 0) decayed_x = 0;
                                x_exc_next_val <= decayed_x;
                            end
                            calc_phase <= P_X_EXC2_DIV;
                        end
                        P_X_EXC2_DIV: begin
                            if (s_exc_next_val) begin
                                x_exc2_next_val <= TRACE_SPIKE_FP;
                            end else begin
                                logic signed [31:0] decayed_x2;
                                decayed_x2 = st_x_exc2 - div_result;
                                if (decayed_x2 < 0) decayed_x2 = 0;
                                x_exc2_next_val <= decayed_x2;
                            end
                            calc_phase <= P_G_EXC_MUL;
                        end
                        P_G_EXC_MUL: begin
                            g_exc_next_val <= mul_result;
                            delay_e2i_rd_addr <= delay_e2i_addr(delay_e2i_rd_idx(), n[NEURON_W-1:0]);
                            calc_phase <= P_I_SYN_I_MUL;
                        end
                        P_I_SYN_I_MUL: begin
                            num_i_val <= (INH_VREST*FP_SCALE) - st_v_inh + mul_result;
                            calc_phase <= P_DV_I_DIV;
                        end
                        P_DV_I_DIV: begin
                            v_next_i_val <= st_v_inh + div_result;
                            calc_phase <= P_R_INH_DIV;
                        end
                        P_R_INH_DIV: begin
                            logic signed [31:0] r_inh_local;
                            if (st_refr_inh != 0) begin
                                refr_inh_next_val <= st_refr_inh - 1'b1;
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
                            r_inh_local = st_r_inh - div_result
                                        + (s_inh_local ? INH_SPIKE_FP : 0);

                            s_exc_next[n] <= s_exc_next_val;
                            state_wr_en <= 1'b1;
                            state_wr_addr <= n[NEURON_W-1:0];
                            state_wr_data <= {
                                g_in_state_next_val,
                                r_exc_next_val,
                                x_exc_next_val,
                                x_exc2_next_val,
                                st_g_inh_state,
                                r_inh_local,
                                v_exc_next_val,
                                v_inh_next_val,
                                theta_next_val,
                                vthr_next_val,
                                refr_exc_next_val,
                                refr_inh_next_val
                            };
                            delay_in_wr_en <= 1'b1;
                            delay_in_wr_addr <= delay_in_addr(delay_in_wr_idx, n[NEURON_W-1:0]);
                            delay_in_wr_data <= g_in_state_next_val;
                            delay_e2i_wr_en <= 1'b1;
                            delay_e2i_wr_addr <= delay_e2i_addr(delay_e2i_wr_idx, n[NEURON_W-1:0]);
                            delay_e2i_wr_data <= g_exc_next_val;
                            sum_r_inh_reg <= sum_r_inh_reg + r_inh_local;

                            if (neuron_idx == N_NEURONS-1) begin
                                neuron_idx <= '0;
                                calc_phase <= P_GIN_DIV;
                                state <= S_NEURON_COMMIT_RD;
                            end else begin
                                neuron_idx <= neuron_idx + 1'b1;
                                calc_phase <= P_GIN_DIV;
                                state_rd_en <= 1'b1;
                                state_rd_addr <= neuron_idx[NEURON_W-1:0] + 1'b1;
                                state <= S_NEURON_RD_WAIT;
                            end
                        end
                        default: begin
                            calc_phase <= P_GIN_DIV;
                            state <= S_NEURON_CALC1;
                        end
                    endcase
                end

                S_NEURON_COMMIT_RD: begin
                    state_rd_en <= 1'b1;
                    state_rd_addr <= neuron_idx[NEURON_W-1:0];
                    state <= S_NEURON_COMMIT_WAIT;
                end

                S_NEURON_COMMIT_WAIT: begin
                    // wait one cycle for BRAM read latency
                    state <= S_NEURON_COMMIT;
                end

                S_NEURON_COMMIT: begin
                    int n;
                    logic signed [63:0] acc_inh;
                    n = neuron_idx;
                    acc_inh = sum_r_inh_reg - st_r_inh;
                    if (N_NEURONS > 1) begin
                        g_inh_next_val_reg <= fp_mul(WINH_DIV_FP, acc_inh[31:0]);
                    end else begin
                        g_inh_next_val_reg <= 0;
                    end
                    state <= S_NEURON_COMMIT_APPLY;
                end

                S_NEURON_COMMIT_APPLY: begin
                    int n;
                    n = neuron_idx;
                    state_wr_en <= 1'b1;
                    state_wr_addr <= n[NEURON_W-1:0];
                    state_wr_data <= {
                        st_g_in_state,
                        st_r_exc,
                        st_x_exc,
                        st_x_exc2,
                        g_inh_next_val_reg,
                        st_r_inh,
                        st_v_exc,
                        st_v_inh,
                        st_theta,
                        st_vthr,
                        st_refr_exc,
                        st_refr_inh
                    };

                    if (neuron_idx == N_NEURONS-1) begin
                        delay_in_wr_idx <= delay_in_next_idx(delay_in_wr_idx);
                        delay_e2i_wr_idx <= delay_e2i_next_idx(delay_e2i_wr_idx);

                        m_tdata <= {tstep_id_reg, s_exc_next};
                        m_tvalid <= 1'b1;
                        state <= S_OUT;
                        neuron_idx <= '0;
                    end else begin
                        neuron_idx <= neuron_idx + 1'b1;
                        state <= S_NEURON_COMMIT_RD;
                    end
                end

        S_OUT: begin
            if (m_tvalid && m_tready) begin
                m_tvalid <= 1'b0;
                if (stdp_pending) begin
                    stdp_pending <= 1'b0;
                    stdp_pre_count <= '0;
                    stdp_post_count <= '0;
                    stdp_scan_in <= '0;
                    stdp_scan_neuron <= '0;
                    state <= S_STDP_SCAN_PRE;
                end else begin
                    state <= S_IDLE;
                end
            end
        end

        S_STDP_SCAN_PRE: begin
            if (stdp_scan_in == N_IN-1) begin
                if (s_in_reg[stdp_scan_in]) begin
                    stdp_pre_idx[stdp_pre_count] <= stdp_scan_in;
                    stdp_pre_count <= stdp_pre_count + 1'b1;
                end
                stdp_scan_in <= '0;
                state <= S_STDP_SCAN_POST;
            end else begin
                if (s_in_reg[stdp_scan_in]) begin
                    stdp_pre_idx[stdp_pre_count] <= stdp_scan_in;
                    stdp_pre_count <= stdp_pre_count + 1'b1;
                end
                stdp_scan_in <= stdp_scan_in + 1'b1;
            end
        end

        S_STDP_SCAN_POST: begin
            logic post_hit;
            post_hit = s_exc_next[stdp_scan_neuron];
            if (post_hit) begin
                stdp_post_idx[stdp_post_count] <= stdp_scan_neuron[NEURON_W-1:0];
                stdp_post_count <= stdp_post_count + 1'b1;
            end
            if (stdp_scan_neuron == N_NEURONS-1) begin
                stdp_scan_neuron <= '0;
                if ((stdp_post_count + post_hit) != 0) begin
                    stdp_mode_post <= 1'b1;
                    stdp_post_i <= '0;
                    stdp_j <= '0;
                    state <= S_STDP_READ;
                end else if (stdp_pre_count != 0) begin
                    stdp_mode_post <= 1'b0;
                    stdp_pre_i <= '0;
                    stdp_g <= '0;
                    stdp_j <= stdp_pre_idx['0];
                    state <= S_STDP_READ;
                end else begin
                    state <= S_IDLE;
                end
            end else begin
                stdp_scan_neuron <= stdp_scan_neuron + 1'b1;
            end
        end

        // Online STDP update (Brian2-style: pre or post spike events)
        S_STDP_READ: begin
            logic [NEURON_W-1:0] stdp_neuron_sel;
            stdp_neuron_sel = stdp_mode_post ? stdp_post_idx[stdp_post_i]
                                             : stdp_g[NEURON_W-1:0];
            mem_r_en <= 1'b1;
            for (i = 0; i < LANES; i = i + 1) begin
                mem_r_neuron[i] <= stdp_neuron_sel;
                mem_r_in[i] <= stdp_j;
            end
            state_rd_en <= 1'b1;
            state_rd_addr <= stdp_neuron_sel;
            x_in_rd_en <= 1'b1;
            x_in_rd_addr <= stdp_j;
            state <= S_STDP_WAIT;
        end

        S_STDP_WAIT: begin
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

                neuron_idx = stdp_mode_post ? stdp_post_idx[stdp_post_i]
                                            : (stdp_g * LANES + i);
                if (neuron_idx < N_NEURONS) begin
                    w_old = $signed(mem_r_data[i]);

                    if (stdp_mode_post) begin
                        pre_term = fp_mul(A_P_FP, fp_mul(x_in_rd_data, st_x_exc2));
                        post_term = 0;
                    end else begin
                        pre_term = 0;
                        post_term = fp_mul(A_M_FP, st_x_exc);
                    end
                    dW = pre_term - post_term;

                    w_new = w_old + dW;
                    if (w_new < WMIN_FP) w_new = WMIN_FP;
                    if (w_new > WMAX_FP) w_new = WMAX_FP;

                    mem_w_en[i] <= 1'b1;
                    mem_w_neuron[i] <= neuron_idx[NEURON_W-1:0];
                    mem_w_in[i] <= stdp_j;
                    mem_w_data[i] <= w_new[W_W-1:0];
                end
            end

            if (stdp_mode_post) begin
                if (stdp_j == N_IN-1) begin
                    stdp_j <= '0;
                    if (stdp_post_i == stdp_post_count - 1'b1) begin
                        if (stdp_pre_count != 0) begin
                            stdp_mode_post <= 1'b0;
                            stdp_pre_i <= '0;
                            stdp_g <= '0;
                            stdp_j <= stdp_pre_idx['0];
                            state <= S_STDP_READ;
                        end else begin
                            state <= S_IDLE;
                        end
                    end else begin
                        stdp_post_i <= stdp_post_i + 1'b1;
                        state <= S_STDP_READ;
                    end
                end else begin
                    stdp_j <= stdp_j + 1'b1;
                    state <= S_STDP_READ;
                end
            end else begin
                if (stdp_g == N_NEURONS-1) begin
                    stdp_g <= '0;
                    if (stdp_pre_i == stdp_pre_count - 1'b1) begin
                        state <= S_IDLE;
                    end else begin
                        stdp_pre_i <= stdp_pre_i + 1'b1;
                        stdp_j <= stdp_pre_idx[stdp_pre_i + 1'b1];
                        state <= S_STDP_READ;
                    end
                end else begin
                    stdp_g <= stdp_g + 1'b1;
                    state <= S_STDP_READ;
                end
            end
        end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
