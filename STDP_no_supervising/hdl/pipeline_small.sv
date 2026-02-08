`default_nettype none

module pipeline_small #(
        parameter int TSTEP_W = 16,
        parameter int N_IN = 784,
        parameter int N_NEURONS = 50,
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

    localparam int LANES = 1;
    localparam int NEURON_GROUPS = (N_NEURONS + LANES - 1) / LANES;

    localparam int INPUT_SPIKE_FP = FP_SCALE / TD_IN_STEPS;
    localparam int TRACE_SPIKE_FP = FP_SCALE / TD_X_STEPS;

    // State
    integer i, j, t;

    logic [TSTEP_W-1:0] tstep_id_reg;
    logic [N_IN-1:0] s_in_reg;
    logic s_stdp_reg;

    (* ram_style = "block" *) logic signed [31:0] r_exc  [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] r_inh  [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] x_in   [0:N_IN-1];
    (* ram_style = "block" *) logic signed [31:0] x_exc  [0:N_NEURONS-1];

    (* ram_style = "block" *) logic signed [31:0] v_exc  [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] theta  [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] vthr   [0:N_NEURONS-1];
    (* ram_style = "block" *) logic [15:0]        refr_exc [0:N_NEURONS-1];

    (* ram_style = "block" *) logic signed [31:0] v_inh  [0:N_NEURONS-1];
    (* ram_style = "block" *) logic [15:0]        refr_inh [0:N_NEURONS-1];

    // g_in is maintained as a state (event-driven update)
    (* ram_style = "block" *) logic signed [31:0] g_in_state [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] g_in_accum [0:N_NEURONS-1];

    (* ram_style = "block" *) logic signed [31:0] g_inh_state [0:N_NEURONS-1];

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

    logic signed [31:0] acc_inh_reg;
    logic signed [31:0] g_inh_next_val_reg;

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

    typedef enum logic [4:0] {
        S_WAIT_WMEM_INIT,
        S_IDLE,
        S_SCAN,
        S_SCAN_DIV_WAIT,
        S_GIN_WAIT,
        S_NEURON_PREP,
        S_NEURON_CALC1,
        S_NEURON_DIV_WAIT,
        S_NEURON_CALC2,
        S_NEURON_COMMIT_RD,
        S_NEURON_COMMIT_WAIT,
        S_NEURON_COMMIT,
        S_NEURON_COMMIT_DIV_WAIT,
        S_NEURON_COMMIT_APPLY,
        S_OUT,
        S_STDP_SCAN_PRE,
        S_STDP_SCAN_POST,
        S_STDP_READ,
        S_STDP_WAIT,
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

    always_ff @(posedge clk) begin
        if (rst) begin
            mul_result <= '0;
        end else begin
            mul_result <= (mul_a * mul_b) >>> FP_SHIFT;
        end
    end

    divider2b #(.WIDTH(32)) u_divider(
        .clk_in        (clk),
        .rst_in        (rst),
        .dividend_in   (div_dividend),
        .divisor_in    (div_divisor),
        .data_valid_in (div_valid_in),
        .quotient_out  (div_quotient),
        .remainder_out (div_remainder),
        .data_valid_out(div_valid_out),
        .error_out     (div_error),
        .busy_out      (div_busy)
    );

    // Memory interface for W_in (4-bank)
    logic mem_r_en;
    logic [NEURON_W-1:0] mem_r_neuron [0:LANES-1];
    logic [IN_W-1:0] mem_r_in [0:LANES-1];
    logic signed [31:0] mem_r_data [0:LANES-1];

    logic mem_w_en [0:LANES-1];
    logic [NEURON_W-1:0] mem_w_neuron [0:LANES-1];
    logic [IN_W-1:0] mem_w_in [0:LANES-1];
    logic signed [31:0] mem_w_data [0:LANES-1];

    logic wmem_init_done;

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

    // Per-neuron next-state storage: heavy 4 in BRAM, light 4 back to arrays
    localparam int NEXT_W = (32*4);
    logic [NEXT_W-1:0] next_wr_data;
    logic [NEXT_W-1:0] next_rd_data;
    logic next_wr_en;
    logic next_rd_en;
    logic [NEURON_W-1:0] next_wr_addr;
    logic [NEURON_W-1:0] next_rd_addr;

    logic [N_NEURONS-1:0] s_exc_next;

    logic signed [31:0] r_inh_next  [0:N_NEURONS-1];
    logic signed [31:0] v_exc_next  [0:N_NEURONS-1];
    logic signed [31:0] v_inh_next  [0:N_NEURONS-1];
    logic signed [31:0] theta_next  [0:N_NEURONS-1];
    logic signed [31:0] vthr_next   [0:N_NEURONS-1];
    logic [15:0]        refr_exc_next [0:N_NEURONS-1];
    logic [15:0]        refr_inh_next [0:N_NEURONS-1];

    wire signed [31:0] g_in_state_next_mem = next_rd_data[NEXT_W-1 -: 32];
    wire signed [31:0] r_exc_next_mem      = next_rd_data[NEXT_W-33 -: 32];
    wire signed [31:0] x_exc_next_mem      = next_rd_data[NEXT_W-65 -: 32];
    wire signed [31:0] g_exc_next_mem      = next_rd_data[NEXT_W-97 -: 32];

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
    logic [31:0]        div_dividend;
    logic [31:0]        div_divisor;
    logic              div_valid_in;
    logic [31:0]        div_quotient;
    logic [31:0]        div_remainder;
    logic              div_valid_out;
    logic              div_error;
    logic              div_busy;
    logic              div_sign_reg;
    logic signed [31:0] div_result;

    logic signed [31:0] scan_x_in_val;
    logic              scan_spike_val;

    logic signed [31:0] num_exc_val;
    logic signed [31:0] num_i_val;
    logic signed [31:0] i_syn_exc_val;
    logic signed [31:0] g_in_state_next_val;
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

`ifdef SYNTHESIS
    // Next-state BRAM (1R1W)
    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(NEURON_W),
        .ADDR_WIDTH_B(NEURON_W),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(NEXT_W),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(N_NEURONS * NEXT_W),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(NEXT_W),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(NEXT_W),
        .WRITE_MODE_B("read_first")
    ) u_next_mem (
        .clka(clk),
        .ena(next_wr_en),
        .wea(next_wr_en),
        .addra(next_wr_addr),
        .dina(next_wr_data),
        .clkb(clk),
        .enb(next_rd_en),
        .addrb(next_rd_addr),
        .doutb(next_rd_data),
        .rstb(rst),
        .regceb(1'b1),
        .sleep(1'b0),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0)
    );
`else
    // Behavioral next-state memory for simulation
    logic [NEXT_W-1:0] next_mem_sim [0:N_NEURONS-1];
    always_ff @(posedge clk) begin
        if (rst) begin
            next_rd_data <= '0;
        end else begin
            if (next_wr_en) begin
                next_mem_sim[next_wr_addr] <= next_wr_data;
            end
            if (next_rd_en) begin
                next_rd_data <= next_mem_sim[next_rd_addr];
            end
        end
    end
`endif

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
            div_dividend <= '0;
            div_divisor <= '0;
            div_valid_in <= 1'b0;
            div_sign_reg <= 1'b0;
            div_result <= '0;
            scan_x_in_val <= '0;
            scan_spike_val <= 1'b0;
            acc_inh_reg <= '0;
            g_inh_next_val_reg <= '0;
            num_exc_val <= '0;
            num_i_val <= '0;
            i_syn_exc_val <= '0;
            g_in_state_next_val <= '0;
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
            next_wr_en <= 1'b0;
            next_rd_en <= 1'b0;
            next_wr_addr <= '0;
            next_rd_addr <= '0;
            next_wr_data <= '0;

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
                s_exc_next[i] <= 1'b0;
                r_inh_next[i] <= '0;
                v_exc_next[i] <= EXC_VRESET * FP_SCALE;
                v_inh_next[i] <= INH_VRESET * FP_SCALE;
                theta_next[i] <= '0;
                vthr_next[i] <= EXC_INIT_VTHR * FP_SCALE;
                refr_exc_next[i] <= '0;
                refr_inh_next[i] <= '0;
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
        end else begin
            // defaults
            mem_r_en <= 1'b0;
            next_wr_en <= 1'b0;
            next_rd_en <= 1'b0;
            div_valid_in <= 1'b0;
            delay_in_wr_en <= 1'b0;
            delay_e2i_wr_en <= 1'b0;
            for (i = 0; i < LANES; i = i + 1) begin
                mem_w_en[i] <= 1'b0;
            end

            case (state)
                S_WAIT_WMEM_INIT: begin
                    s_tready <= 1'b0;
                    if (wmem_init_done) begin
                        state <= S_CLR_DELAY_IN;
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
                        logic signed [31:0] scan_val;
                        logic signed [31:0] scan_abs;
                        scan_val = x_in[scan_in_idx];
                        scan_abs = scan_val[31] ? (~scan_val + 1'b1) : scan_val;
                        scan_x_in_val <= scan_val;
                        scan_spike_val <= s_in_reg[scan_in_idx];
                        div_sign_reg <= scan_val[31];
                        div_dividend <= scan_abs + (TD_X_STEPS >> 1);
                        div_divisor <= TD_X_STEPS;
                        div_valid_in <= 1'b1;
                        state <= S_SCAN_DIV_WAIT;
                    end else begin
                        state <= S_NEURON_PREP;
                    end
                end

                S_SCAN_DIV_WAIT: begin
                    if (div_valid_out) begin
                        logic signed [31:0] div_signed;
                        logic signed [31:0] x_next;
                        div_signed = div_sign_reg ? -$signed(div_quotient) : $signed(div_quotient);
                        x_next = scan_x_in_val - div_signed
                               + (scan_spike_val ? TRACE_SPIKE_FP : 0);
                        x_in[scan_in_idx] <= x_next;

                        if (scan_spike_val) begin
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
                            state <= S_SCAN;
                        end
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
                    logic do_div;
                    logic signed [31:0] div_src;
                    logic signed [31:0] div_abs;
                    logic [31:0] div_den;
                    n = neuron_idx;
                    do_div = 1'b0;
                    div_src = '0;
                    div_abs = '0;
                    div_den = '0;
                    case (calc_phase)
                        P_GIN_DIV: begin
                            div_src = g_in_state[n];
                            div_den = TD_IN_STEPS;
                            do_div = 1'b1;
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
                            div_src = num_exc_val;
                            div_den = EXC_TAU_M;
                            do_div = 1'b1;
                        end
                        P_THETA_DIV: begin
                            div_src = theta[n];
                            div_den = EXC_TC_THETA;
                            do_div = 1'b1;
                        end
                        P_R_EXC_DIV: begin
                            div_src = r_exc[n];
                            div_den = TD_EXC_STEPS;
                            do_div = 1'b1;
                        end
                        P_X_EXC_DIV: begin
                            div_src = x_exc[n];
                            div_den = TD_X_STEPS;
                            do_div = 1'b1;
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
                            div_src = num_i_val;
                            div_den = INH_TAU_M;
                            do_div = 1'b1;
                        end
                        P_R_INH_DIV: begin
                            div_src = r_inh[n];
                            div_den = TD_INH_STEPS;
                            do_div = 1'b1;
                        end
                        default: begin
                            mul_a <= '0;
                            mul_b <= '0;
                        end
                    endcase
                    if (do_div) begin
                        div_sign_reg <= div_src[31];
                        div_abs = div_src[31] ? (~div_src + 1'b1) : div_src;
                        div_dividend <= div_abs + (div_den >> 1);
                        div_divisor <= div_den;
                        div_valid_in <= 1'b1;
                        state <= S_NEURON_DIV_WAIT;
                    end else begin
                        state <= S_NEURON_CALC2;
                    end
                end

                S_NEURON_DIV_WAIT: begin
                    if (div_valid_out) begin
                        div_result <= div_sign_reg ? -$signed(div_quotient) : $signed(div_quotient);
                        state <= S_NEURON_CALC2;
                    end
                end

                S_NEURON_CALC2: begin
                    int n;
                    logic signed [31:0] theta_tmp;
                    logic s_exc_local;
                    logic s_inh_local;
                    n = neuron_idx;

                    case (calc_phase)
                        P_GIN_DIV: begin
                            g_in_state_next_val <= g_in_state[n] - div_result + g_in_accum[n];
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
                            next_wr_en <= 1'b1;
                            next_wr_addr <= n[NEURON_W-1:0];
                            next_wr_data <= {
                                g_in_state_next_val,
                                r_exc_next_val,
                                x_exc_next_val,
                                g_exc_next_val
                            };
                            r_inh_next[n] <= r_inh_local;
                            v_exc_next[n] <= v_exc_next_val;
                            v_inh_next[n] <= v_inh_next_val;
                            theta_next[n] <= theta_next_val;
                            vthr_next[n] <= vthr_next_val;
                            refr_exc_next[n] <= refr_exc_next_val;
                            refr_inh_next[n] <= refr_inh_next_val;
                            sum_r_inh_reg <= sum_r_inh_reg + r_inh_local;

                            if (neuron_idx == N_NEURONS-1) begin
                                neuron_idx <= '0;
                                calc_phase <= P_GIN_DIV;
                                state <= S_NEURON_COMMIT_RD;
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

                S_NEURON_COMMIT_RD: begin
                    next_rd_en <= 1'b1;
                    next_rd_addr <= neuron_idx[NEURON_W-1:0];
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
                    acc_inh = sum_r_inh_reg - r_inh_next[n];
                    if (N_NEURONS > 1) begin
                        acc_inh_reg <= acc_inh[31:0];
                        div_sign_reg <= 1'b0;
                        div_dividend <= WINH_FP + ((N_NEURONS-1) >> 1);
                        div_divisor <= (N_NEURONS-1);
                        div_valid_in <= 1'b1;
                        state <= S_NEURON_COMMIT_DIV_WAIT;
                    end else begin
                        g_inh_next_val_reg <= 0;
                        state <= S_NEURON_COMMIT_APPLY;
                    end
                end

                S_NEURON_COMMIT_DIV_WAIT: begin
                    if (div_valid_out) begin
                        g_inh_next_val_reg <= fp_mul(div_quotient, acc_inh_reg);
                        state <= S_NEURON_COMMIT_APPLY;
                    end
                end

                S_NEURON_COMMIT_APPLY: begin
                    int n;
                    n = neuron_idx;
                    g_in_state[n] <= g_in_state_next_mem;
                    r_exc[n] <= r_exc_next_mem;
                    x_exc[n] <= x_exc_next_mem;
                    r_inh[n] <= r_inh_next[n];
                    v_exc[n] <= v_exc_next[n];
                    theta[n] <= theta_next[n];
                    vthr[n] <= vthr_next[n];
                    refr_exc[n] <= refr_exc_next[n];
                    v_inh[n] <= v_inh_next[n];
                    refr_inh[n] <= refr_inh_next[n];
                    g_inh_state[n] <= g_inh_next_val_reg;
                    delay_in_wr_en <= 1'b1;
                    delay_in_wr_addr <= delay_in_addr(delay_in_wr_idx, n[NEURON_W-1:0]);
                    delay_in_wr_data <= g_in_state_next_mem;
                    delay_e2i_wr_en <= 1'b1;
                    delay_e2i_wr_addr <= delay_e2i_addr(delay_e2i_wr_idx, n[NEURON_W-1:0]);
                    delay_e2i_wr_data <= g_exc_next_mem;

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

        // Online STDP update (4 weights per cycle)
        S_STDP_READ: begin
            logic [NEURON_W-1:0] stdp_neuron_sel;
            stdp_neuron_sel = stdp_mode_post ? stdp_post_idx[stdp_post_i]
                                             : stdp_g[NEURON_W-1:0];
            mem_r_en <= 1'b1;
            for (i = 0; i < LANES; i = i + 1) begin
                mem_r_neuron[i] <= stdp_neuron_sel;
                mem_r_in[i] <= stdp_j;
            end
            if (!stdp_mode_post) begin
                next_rd_en <= 1'b1;
                next_rd_addr <= stdp_neuron_sel;
            end else begin
                next_rd_en <= 1'b0;
                next_rd_addr <= '0;
            end
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
                    w_old = mem_r_data[i];

                    if (stdp_mode_post) begin
                        pre_term = fp_mul(A_P_FP, x_in[stdp_j]);
                        post_term = 0;
                    end else begin
                        pre_term = 0;
                        post_term = fp_mul(A_M_FP, x_exc_next_mem);
                    end
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
