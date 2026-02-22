`default_nettype none // prevents system from inferring an undeclared logic (good practice)
 
module top_level(
    input  wire        clk_100mhz,
    input  wire [3:0]  btn,
    input  wire [15:0] sw,
    input  wire        uart_rxd,
    output logic       uart_txd,
    input  wire        SD_DQ0,
    output wire        SD_DQ1,
    output wire        SD_DQ2,
    output wire        SD_DQ3,
    output wire        SD_CMD,
    output wire        SD_CLK,
    input  wire        SD_CD_N,
    output logic [15:0] led,
    output logic [2:0] rgb0,
    output logic [2:0] rgb1,
    output logic [2:0] pmoda
);

    localparam int CLKS_PER_BIT = 217; // 25_000_000 / 115_200 ~= 217

    localparam logic [7:0] REQ_SYNC   = 8'hA5;
    localparam logic [7:0] RESP_SYNC  = 8'h5A;
    localparam logic [7:0] PROTO_VER  = 8'h01;
    localparam logic [7:0] OP_ADD_I32 = 8'h01;
    localparam logic [7:0] OP_DDR_WRITE32 = 8'h10;
    localparam logic [7:0] OP_SD_TO_DDR_COPY = 8'h11;
    localparam logic [7:0] OP_RUN_SAMPLE_INFER = 8'h20;
    localparam logic [7:0] OP_READ_SPIKE_COUNT = 8'h21;
    localparam logic [7:0] OP_READ_RAW_U8 = 8'h22;
    localparam logic [7:0] OP_READ_POISSON_THRESH = 8'h23;
    localparam logic [7:0] OP_READ_INFER_DEBUG = 8'h24;
    localparam logic [7:0] OP_WRITE_INFER_WEIGHT = 8'h25;
    localparam logic [7:0] OP_TRAIN_QUERY_CAPS = 8'h30;
    localparam logic [7:0] OP_TRACE_UPDATE = 8'h31;
    localparam logic [7:0] OP_STDP_UPDATE_TILE = 8'h32;
    localparam logic [31:0] DDR_ADDR_WORD_LIMIT = 32'd16777216; // 64MiB / 4
    localparam logic [7:0] MAX_SUPPORTED_NARGS = 8'd2;
    // Increase RX timeout margin to tolerate host-side inter-byte gaps on UART.
    localparam int RX_TIMEOUT_CLKS = CLKS_PER_BIT * 2000;
    localparam int N_IN = 784;
    localparam int N_NEURONS = 100;
    localparam int N_WEIGHTS = N_IN * N_NEURONS;
    localparam logic signed [31:0] FXP_ALPHA = 32'sd62259; // legacy/simple model coeff (unused in mine-style LIF)
    localparam logic signed [31:0] FXP_ALPHA_INH = 32'sd58982; // legacy/simple model coeff (unused in mine-style LIF)
    localparam logic signed [31:0] FXP_INPUT_W = 32'sd8192; // 0.125 in S16.16
    localparam logic signed [31:0] FXP_THRESH = 32'sd65536; // 1.0 in S16.16
    localparam logic signed [31:0] FXP_BIAS_LSB = 32'sd512; // 0.0078125 in S16.16
    localparam logic signed [31:0] FXP_ONE = 32'sd65536; // 1.0 in S16.16
    localparam logic signed [31:0] FXP_HALF = 32'sd32768; // 0.5 in S16.16
    localparam logic signed [31:0] FXP_WEXC = 32'sd147456; // 2.25 in S16.16
    localparam logic signed [31:0] FXP_INH_COEFF = 32'sd563; // (0.85/99) in S16.16
    localparam logic signed [31:0] FXP_INH_THRESH = -32'sd2621440; // -40.0 in S16.16
    localparam logic signed [31:0] FXP_SCALE_1000 = 32'sd65536000;   // 1000.0 in S16.16 (1/1ms)
    localparam logic signed [31:0] FXP_SCALE_500  = 32'sd32768000;   // 500.0 in S16.16 (1/2ms)
    localparam logic signed [31:0] FXP_GEXC_SPIKE = 32'sd147456000;  // 2.25 * 1000 in S16.16
    // Step-domain approximations for mine.py neuron dynamics (dt=1ms)
    localparam logic [15:0] EXC_TREF_STEPS = 16'd5;
    localparam logic [15:0] INH_TREF_STEPS = 16'd2;
    localparam logic signed [31:0] FXP_THETA_PLUS = 32'sd3277;   // approx 0.05 in S16.16
    localparam logic signed [31:0] FXP_THETA_DECAY = 32'sd65535; // ~1.0 (dt/tc_theta is tiny)
    localparam logic signed [31:0] FXP_THETA_MAX = 32'sd2293760; // 35.0 in S16.16
    localparam logic signed [31:0] FXP_THRESH_BASE = -32'sd3407872; // -52.0 in S16.16 (DiehlAndCook init_vthr)
    localparam logic signed [31:0] FXP_EXC_VREST = -32'sd4259840;   // -65.0 in S16.16
    localparam logic signed [31:0] FXP_EXC_VRESET = -32'sd4259840;  // -65.0 in S16.16
    localparam logic signed [31:0] FXP_EXC_EEXC = 32'sd0;           // 0.0 in S16.16
    localparam logic signed [31:0] FXP_EXC_EINH = -32'sd6553600;    // -100.0 in S16.16
    localparam logic signed [31:0] FXP_EXC_DT_OVER_TCM = 32'sd655;  // 0.01 in S16.16
    localparam logic signed [31:0] FXP_INH_VREST = -32'sd3932160;   // -60.0 in S16.16
    localparam logic signed [31:0] FXP_INH_VRESET = -32'sd2949120;  // -45.0 in S16.16
    localparam logic signed [31:0] FXP_INH_EEXC = 32'sd0;           // 0.0 in S16.16
    localparam logic signed [31:0] FXP_INH_EINH = -32'sd5570560;    // -85.0 in S16.16
    localparam logic signed [31:0] FXP_INH_DT_OVER_TCM = 32'sd6554; // 0.1 in S16.16
    // mine.py input_synapse has dt==td==1ms -> decay term becomes ~0 for c_in/g_in state update.
    localparam logic signed [31:0] FXP_INPUT_G_DECAY = 32'sd0;
    localparam logic [31:0] POISSON_NUM_CONST = 32'd9175; // floor(32*140*2048*1e-3)
    localparam logic [10:0] RNG_MAX = 11'd2047;
    localparam logic [31:0] LCG_A = 32'd1664525;
    localparam logic [31:0] LCG_C = 32'd1013904223;

    localparam logic [7:0] STATUS_OK             = 8'h00;
    localparam logic [7:0] STATUS_BAD_PACKET     = 8'hE1;
    localparam logic [7:0] STATUS_UNSUPPORTED_OP = 8'hE2;
    localparam logic [7:0] BADDBG_READ_SPIKE_ARG = 8'h11;
    localparam logic [7:0] BADDBG_SD_REQ_ARG     = 8'h20;
    localparam logic [7:0] BADDBG_SD_CD_N        = 8'h21;
    localparam logic [7:0] BADDBG_SD_WAIT_TO     = 8'h22;
    localparam logic [7:0] BADDBG_SD_BAD_HEADER  = 8'h23;
    localparam logic [7:0] BADDBG_SD_SECTOR_END  = 8'h24;
    localparam logic [7:0] BADDBG_READ_INFER_DBG = 8'h14;
    localparam logic [7:0] BADDBG_WRITE_WEIGHT   = 8'h15;
    // Training kernel capability bits (host-visible via OP_TRAIN_QUERY_CAPS)
    // [0]=query_caps impl, [1]=logical DDR map fixed, [2]=trace opcode reserved,
    // [3]=tile opcode reserved, [8]=trace kernel exec impl, [9]=tile kernel exec impl.
    localparam logic [31:0] TRAIN_CAPS_VALUE = 32'h0000000F;
    // Step1 logical DDR word map contract (future external DDR integration target).
    localparam logic [31:0] TRAIN_BASE_W_Q16_WORDS  = 32'd0;
    localparam logic [31:0] TRAIN_BASE_A_Q16_WORDS  = TRAIN_BASE_W_Q16_WORDS + N_WEIGHTS;
    localparam logic [31:0] TRAIN_BASE_BT_Q16_WORDS = TRAIN_BASE_A_Q16_WORDS + N_WEIGHTS;
    localparam logic [31:0] TRAIN_BASE_THETA_WORDS  = TRAIN_BASE_BT_Q16_WORDS + N_WEIGHTS;
    localparam logic [31:0] TRAIN_BASE_VSTATE_WORDS = TRAIN_BASE_THETA_WORDS + N_NEURONS;
    localparam logic [31:0] TRAIN_BASE_DELAY_WORDS  = TRAIN_BASE_VSTATE_WORDS + N_NEURONS;
    localparam logic [31:0] TRAIN_BASE_GIN_WORDS    = TRAIN_BASE_DELAY_WORDS + (N_NEURONS * 8);

    typedef enum logic [2:0] {
        RX_WAIT_SYNC,
        RX_GET_VER,
        RX_GET_OPCODE,
        RX_GET_NARGS,
        RX_GET_ARGS,
        RX_GET_CHECKSUM
    } rx_state_t;

    typedef enum logic [1:0] {
        TX_IDLE,
        TX_SEND,
        TX_WAIT_DONE
    } tx_state_t;
    typedef enum logic [1:0] {
        MEMRD_NONE,
        MEMRD_SPIKE_COUNT,
        MEMRD_RAW_U8,
        MEMRD_POISSON_THRESH
    } memrd_kind_t;
    typedef enum logic [2:0] {
        INFER_IDLE,
        INFER_INIT_CLEAR,
        INFER_PREP_DIV_START,
        INFER_PREP_DIV_WAIT,
        INFER_GEN_INPUT_SPIKES,
        INFER_ACCUM_NEURON,
        INFER_APPLY_WTA,
        INFER_WTA_PASS2
    } infer_state_t;

    rx_state_t rx_state;
    tx_state_t tx_state;

    logic       rx_dv;
    logic [7:0] rx_byte;
    logic       tx_dv;
    logic [7:0] tx_byte;
    logic       tx_active;
    logic       tx_done;

    logic [7:0] req_ver;
    logic [7:0] req_opcode;
    logic [7:0] req_nargs;
    logic [7:0] req_checksum;
    logic [7:0] req_checksum_accum;
    logic [2:0] arg_byte_idx;
    logic [2:0] args_seen;
    logic signed [31:0] arg0;
    logic signed [31:0] arg1;

    logic       response_ready;
    logic [7:0] resp_status;
    logic signed [31:0] resp_result;
    logic [7:0] resp_checksum;
    logic [2:0] tx_byte_idx;
    logic [31:0] ddr_write_count;
    logic [31:0] ddr_last_addr;
    logic [31:0] ddr_last_data;
    logic [15:0] rx_timeout_counter;
    logic [1:0]  clk_div;
    wire         clk_25mhz = clk_div[1];
    wire         core_clk = clk_25mhz;

    logic        sd_rd;
    logic        sd_wr;
    logic [31:0] sd_address;
    logic [7:0]  sd_dout;
    logic        sd_byte_available;
    logic        sd_ready;
    logic [4:0]  sd_status;

    logic        sd_copy_active;
    logic        sd_in_read;
    logic [31:0] sd_copy_lba;
    logic [31:0] sd_copy_sectors_left;
    logic [8:0]  sd_byte_count;
    logic [1:0]  sd_pack_idx;
    logic [31:0] sd_pack_word;
    logic [31:0] sd_copy_words_written;
    logic [23:0] sd_wait_counter;
    logic [7:0]  sd_header_bytes [0:19];
    logic        sd_header_done;
    logic [31:0] sd_file_total_bytes;
    logic [31:0] sd_file_bytes_seen;
    logic        sd_copy_done_pending;
    logic        sd_use_sector_limit;
    (* ram_style = "block" *) logic [7:0]  raw_image0_u8 [0:N_IN-1];
    logic        raw_image0_valid;
    logic [31:0] raw_num_images;
    logic [31:0] raw_bytes_per_image;
    logic [9:0]  raw_image0_capture_idx;
    logic [31:0] raw_image0_sum_u8;
    logic [9:0]  raw_image0_rd_addr;
    logic [7:0]  raw_image0_rd_data;

    logic        infer_active;
    infer_state_t infer_state;
    logic [31:0] infer_steps_target;
    logic [15:0] infer_step_idx;
    logic [6:0]  infer_neuron_idx;
    logic [9:0]  infer_input_idx;
    logic [9:0]  infer_prep_idx;
    logic signed [31:0] infer_accum;
    logic [1:0]  infer_accum_weight_phase;
    logic [16:0] infer_w_rd_addr;
    logic [15:0] infer_w_rd_data;
    (* ram_style = "block" *) logic signed [31:0] infer_g_in_state [0:N_NEURONS-1];
    (* ram_style = "block" *) logic [15:0] infer_w_q16 [0:N_WEIGHTS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_v_state [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_exc_theta [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_in_delay0 [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_in_delay1 [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_in_delay2 [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_in_delay3 [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_in_delay4 [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_v_inh_state [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_c_inh_state [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_inh_state [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_exc_delay0 [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_exc_delay1 [0:N_NEURONS-1];
    logic        infer_s_exc [0:N_NEURONS-1];
    (* ram_style = "block" *) logic [15:0] infer_spike_count [0:N_NEURONS-1];
    logic [6:0]  infer_spike_count_rd_addr;
    logic [15:0] infer_spike_count_rd_data;
    (* ram_style = "block" *) logic [15:0] infer_exc_last_spike_step [0:N_NEURONS-1];
    (* ram_style = "block" *) logic [15:0] infer_inh_last_spike_step [0:N_NEURONS-1];
    logic [6:0]  infer_apply_idx;
    logic signed [31:0] infer_sum_c_inh;
    logic [31:0] infer_total_spikes;
    logic [31:0] infer_rng_state;
    (* ram_style = "block" *) logic [10:0] infer_poisson_thresh [0:N_IN-1];
    logic [9:0]  infer_poisson_thresh_rd_addr;
    logic [10:0] infer_poisson_thresh_rd_data;
    logic        infer_input_spike [0:N_IN-1];
    logic [31:0] infer_dividend;
    logic [31:0] infer_divisor;
    logic        infer_div_valid;
    logic [31:0] infer_div_q;
    logic [31:0] infer_div_r;
    logic        infer_div_out_valid;
    logic        infer_div_err;
    logic        infer_div_busy;
    logic [31:0] infer_dbg_total_input_spikes;
    logic [31:0] infer_dbg_total_syn_hits;
    logic [31:0] infer_dbg_last_step_input_spikes;
    logic [31:0] infer_dbg_first_step_input_spikes;
    logic [31:0] infer_dbg_first_step_hits_n0;
    logic [31:0] infer_dbg_first_step_hits_n3;
    logic [31:0] infer_dbg_first_step_hits_n7;
    logic [31:0] infer_dbg_curr_step_input_spikes;
    logic        memrd_pending;
    logic        memrd_wait;
    memrd_kind_t memrd_kind;
    logic [15:0] memrd_idx;
    logic [2:0]  dbg_rgb1_state;
    integer rr;

    wire [7:0] r_in = {sw[15:11], 3'b000};
    wire [7:0] g_in = {sw[10:5],  2'b00};
    wire [7:0] b_in = {sw[4:0],   3'b000};

    assign SD_DQ1 = 1'b1;
    assign SD_DQ2 = 1'b1;

    assign rgb0[2] = tx_active;  // blue LED: UART TX active
    assign rgb0[1] = ddr_write_count[0]; // green LED: DDR write activity bit
    assign rgb0[0] = (resp_status == STATUS_OK); // red LED: OK result

    // rgb1 shows coarse progress for bring-up/debug:
    // 000=idle, 001=sd copy, 010=response pending, 011=infer init clear,
    // 100=infer prep threshold, 101=gen input spikes, 110=accum excit,
    // 111=inhib/update passes (APPLY_WTA or WTA_PASS2)
    always_comb begin
        dbg_rgb1_state = 3'b000;
        if (sd_copy_active) begin
            dbg_rgb1_state = 3'b001;
        end else if (response_ready) begin
            dbg_rgb1_state = 3'b010;
        end else if (infer_active) begin
            case (infer_state)
                INFER_INIT_CLEAR:      dbg_rgb1_state = 3'b011;
                INFER_PREP_DIV_START,
                INFER_PREP_DIV_WAIT:   dbg_rgb1_state = 3'b100;
                INFER_GEN_INPUT_SPIKES:dbg_rgb1_state = 3'b101;
                INFER_ACCUM_NEURON:    dbg_rgb1_state = 3'b110;
                INFER_APPLY_WTA,
                INFER_WTA_PASS2:       dbg_rgb1_state = 3'b111;
                default:               dbg_rgb1_state = 3'b000;
            endcase
        end
    end

    assign rgb1 = 3'b000;
    assign led[2:0] = dbg_rgb1_state;           // LD0..LD2: coarse state code
    assign led[3] = infer_active;               // LD3: inference active
    assign led[4] = sd_copy_active;             // LD4: SD copy active
    assign led[5] = response_ready;             // LD5: response pending
    assign led[6] = tx_active;                  // LD6: UART TX active
    assign led[7] = (resp_status == STATUS_OK); // LD7: last response OK
    assign led[15:8] = 8'h00;
    assign pmoda = {rgb0[0], rgb0[1], rgb0[2]};

    // Default inference weights are initialized on-FPGA at configuration time.
    // Pattern currently matches the legacy fixed wiring used by the Python fallback:
    // weight = 0.125 (Q0.16 = 8192) when ((input_idx + neuron_idx) & 3) == 0 else 0.
    initial begin : init_infer_weights
        integer wn;
        integer wi;
        integer widx;
        for (wn = 0; wn < N_NEURONS; wn = wn + 1) begin
            for (wi = 0; wi < N_IN; wi = wi + 1) begin
                widx = (wn * N_IN) + wi;
                if (((wi + wn) & 32'd3) == 0) begin
                    infer_w_q16[widx] = FXP_INPUT_W[15:0];
                end else begin
                    infer_w_q16[widx] = 16'd0;
                end
            end
        end
    end

    always_ff @(posedge clk_100mhz) begin
        if (btn[0]) begin
            clk_div <= 2'b00;
        end else begin
            clk_div <= clk_div + 2'b01;
        end
    end

    // Explicit synchronous read port for infer weight RAM to push Vivado toward BRAM
    // inference (instead of LUTRAM/distributed RAM).
    always_ff @(posedge core_clk) begin
        infer_w_rd_data <= infer_w_q16[infer_w_rd_addr];
        raw_image0_rd_data <= raw_image0_u8[raw_image0_rd_addr];
        infer_spike_count_rd_data <= infer_spike_count[infer_spike_count_rd_addr];
        infer_poisson_thresh_rd_data <= infer_poisson_thresh[infer_poisson_thresh_rd_addr];
    end

    uart_rx #(
        .CLKS_PER_BIT(CLKS_PER_BIT)
    ) u_uart_rx (
        .i_clk       (core_clk),
        .i_rst       (btn[0]),
        .i_rx_serial (uart_rxd),
        .o_rx_dv     (rx_dv),
        .o_rx_byte   (rx_byte)
    );

    uart_tx #(
        .CLKS_PER_BIT(CLKS_PER_BIT)
    ) u_uart_tx (
        .i_clk       (core_clk),
        .i_rst       (btn[0]),
        .i_tx_dv     (tx_dv),
        .i_tx_byte   (tx_byte),
        .o_tx_active (tx_active),
        .o_tx_serial (uart_txd),
        .o_tx_done   (tx_done)
    );

    sd_controller u_sd_controller (
        .cs                 (SD_DQ3),
        .mosi               (SD_CMD),
        .miso               (SD_DQ0),
        .sclk               (SD_CLK),
        .rd                 (sd_rd),
        .dout               (sd_dout),
        .byte_available     (sd_byte_available),
        .wr                 (sd_wr),
        .din                (8'h00),
        .ready_for_next_byte(),
        .reset              (btn[0]),
        .ready              (sd_ready),
        .address            (sd_address),
        .clk                (core_clk),
        .status             (sd_status)
    );

    rgb_controller u_rgb_controller (
        .clk   (core_clk),
        .rst   (btn[0]),
        .r_in  (r_in),
        .g_in  (g_in),
        .b_in  (b_in),
        .r_out (),
        .g_out (),
        .b_out ()
    );

    divider2b #(.WIDTH(32)) u_poisson_divider (
        .clk_in        (core_clk),
        .rst_in        (btn[0]),
        .dividend_in   (infer_dividend),
        .divisor_in    (infer_divisor),
        .data_valid_in (infer_div_valid),
        .quotient_out  (infer_div_q),
        .remainder_out (infer_div_r),
        .data_valid_out(infer_div_out_valid),
        .error_out     (infer_div_err),
        .busy_out      (infer_div_busy)
    );
    
    function automatic [7:0] calc_resp_checksum(
        input [7:0] status_in,
        input signed [31:0] result_in
    );
        begin
            calc_resp_checksum = status_in
                               ^ result_in[7:0]
                               ^ result_in[15:8]
                               ^ result_in[23:16]
                               ^ result_in[31:24];
        end
    endfunction

    function automatic signed [31:0] neuron_bias(input logic [6:0] neuron_idx);
        begin
            neuron_bias = ($signed({28'd0, neuron_idx[2:0]}) + 32'sd1) * FXP_BIAS_LSB;
        end
    endfunction

    // S16.16 multiply with symmetric rounding (reduces systematic truncation bias vs [47:16] slicing)
    function automatic signed [31:0] fxp_mul_s16_16(
        input signed [31:0] a,
        input signed [31:0] b
    );
        logic signed [63:0] p;
        begin
            p = $signed(a) * $signed(b);
            if (p >= 0) begin
                fxp_mul_s16_16 = $signed((p + 64'sd32768) >>> 16);
            end else begin
                fxp_mul_s16_16 = $signed((p - 64'sd32768) >>> 16);
            end
        end
    endfunction

    always_ff @(posedge core_clk) begin
        if (btn[0]) begin
            rx_state          <= RX_WAIT_SYNC;
            tx_state          <= TX_IDLE;
            req_ver           <= 8'h00;
            req_opcode        <= 8'h00;
            req_nargs         <= 8'h00;
            req_checksum      <= 8'h00;
            req_checksum_accum<= 8'h00;
            arg_byte_idx      <= 3'd0;
            args_seen         <= 3'd0;
            arg0              <= 32'sd0;
            arg1              <= 32'sd0;
            response_ready    <= 1'b0;
            resp_status       <= STATUS_BAD_PACKET;
            resp_result       <= 32'sd0;
            resp_checksum     <= 8'h00;
            tx_byte_idx       <= 3'd0;
            tx_dv             <= 1'b0;
            tx_byte           <= 8'h00;
            ddr_write_count   <= 32'd0;
            ddr_last_addr     <= 32'd0;
            ddr_last_data     <= 32'd0;
            rx_timeout_counter<= 16'd0;
            sd_rd             <= 1'b0;
            sd_wr             <= 1'b0;
            sd_address        <= 32'd0;
            sd_copy_active    <= 1'b0;
            sd_in_read        <= 1'b0;
            sd_copy_lba       <= 32'd0;
            sd_copy_sectors_left <= 32'd0;
            sd_byte_count     <= 9'd0;
            sd_pack_idx       <= 2'd0;
            sd_pack_word      <= 32'd0;
            sd_copy_words_written <= 32'd0;
            sd_wait_counter    <= 24'd0;
            sd_header_done     <= 1'b0;
            sd_file_total_bytes<= 32'd0;
            sd_file_bytes_seen <= 32'd0;
            sd_copy_done_pending <= 1'b0;
            sd_use_sector_limit <= 1'b0;
            raw_image0_valid    <= 1'b0;
            raw_num_images      <= 32'd0;
            raw_bytes_per_image <= 32'd0;
            raw_image0_capture_idx <= 10'd0;
            raw_image0_sum_u8   <= 32'd0;
            raw_image0_rd_addr  <= 10'd0;
            infer_active        <= 1'b0;
            infer_state         <= INFER_IDLE;
            infer_steps_target  <= 32'd0;
            infer_step_idx      <= 16'd0;
            infer_neuron_idx    <= 7'd0;
            infer_input_idx     <= 10'd0;
            infer_prep_idx      <= 10'd0;
            infer_accum         <= 32'sd0;
            infer_accum_weight_phase <= 2'd0;
            infer_w_rd_addr     <= 17'd0;
            infer_apply_idx     <= 7'd0;
            infer_sum_c_inh     <= 32'sd0;
            infer_total_spikes  <= 32'd0;
            infer_rng_state     <= 32'd0;
            infer_dividend      <= 32'd0;
            infer_divisor       <= 32'd1;
            infer_div_valid     <= 1'b0;
            infer_dbg_total_input_spikes <= 32'd0;
            infer_dbg_total_syn_hits <= 32'd0;
            infer_dbg_last_step_input_spikes <= 32'd0;
            infer_dbg_first_step_input_spikes <= 32'd0;
            infer_dbg_first_step_hits_n0 <= 32'd0;
            infer_dbg_first_step_hits_n3 <= 32'd0;
            infer_dbg_first_step_hits_n7 <= 32'd0;
            infer_dbg_curr_step_input_spikes <= 32'd0;
            infer_spike_count_rd_addr <= 7'd0;
            infer_poisson_thresh_rd_addr <= 10'd0;
            memrd_pending <= 1'b0;
            memrd_wait    <= 1'b0;
            memrd_kind    <= MEMRD_NONE;
            memrd_idx     <= 16'd0;
            // Large state arrays are cleared by a sequential init phase before inference
            // to reduce control sets and allow BRAM inference.
        end else begin
            tx_dv <= 1'b0;
            sd_rd <= 1'b0;
            sd_wr <= 1'b0;
            infer_div_valid <= 1'b0;
            if (response_ready || (rx_state == RX_WAIT_SYNC)) begin
                rx_timeout_counter <= 16'd0;
            end else if (rx_dv) begin
                rx_timeout_counter <= 16'd0;
            end else if (rx_timeout_counter >= RX_TIMEOUT_CLKS - 1) begin
                rx_state           <= RX_WAIT_SYNC;
                req_checksum_accum <= 8'h00;
                arg_byte_idx       <= 3'd0;
                args_seen          <= 3'd0;
                rx_timeout_counter <= 16'd0;
                resp_status        <= STATUS_BAD_PACKET;
                resp_result        <= 32'sd0;
                resp_checksum      <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                response_ready     <= 1'b1;
            end else begin
                rx_timeout_counter <= rx_timeout_counter + 16'd1;
            end

            if (memrd_pending && !response_ready) begin
                if (memrd_wait) begin
                    memrd_wait <= 1'b0;
                end else begin
                    memrd_pending <= 1'b0;
                    case (memrd_kind)
                        MEMRD_SPIKE_COUNT: begin
                            resp_status    <= STATUS_OK;
                            resp_result    <= {16'd0, infer_spike_count_rd_data};
                            resp_checksum  <= calc_resp_checksum(STATUS_OK, {16'd0, infer_spike_count_rd_data});
                            response_ready <= 1'b1;
                        end
                        MEMRD_RAW_U8: begin
                            resp_status    <= STATUS_OK;
                            resp_result    <= {24'd0, raw_image0_rd_data};
                            resp_checksum  <= calc_resp_checksum(STATUS_OK, {24'd0, raw_image0_rd_data});
                            response_ready <= 1'b1;
                        end
                        MEMRD_POISSON_THRESH: begin
                            resp_status    <= STATUS_OK;
                            resp_result    <= {21'd0, infer_poisson_thresh_rd_data};
                            resp_checksum  <= calc_resp_checksum(STATUS_OK, {21'd0, infer_poisson_thresh_rd_data});
                            response_ready <= 1'b1;
                        end
                        default: begin
                            resp_status    <= STATUS_BAD_PACKET;
                            resp_result    <= 32'sd0;
                            resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                            response_ready <= 1'b1;
                        end
                    endcase
                    memrd_kind <= MEMRD_NONE;
                end
            end

            if (rx_dv && !response_ready && !memrd_pending && !sd_copy_active && !infer_active) begin
                case (rx_state)
                    RX_WAIT_SYNC: begin
                        if (rx_byte == REQ_SYNC) begin
                            rx_state           <= RX_GET_VER;
                            req_checksum_accum <= 8'h00;
                            arg_byte_idx       <= 3'd0;
                            args_seen          <= 3'd0;
                            arg0               <= 32'sd0;
                            arg1               <= 32'sd0;
                        end
                    end

                    RX_GET_VER: begin
                        req_ver           <= rx_byte;
                        req_checksum_accum<= req_checksum_accum ^ rx_byte;
                        rx_state          <= RX_GET_OPCODE;
                    end

                    RX_GET_OPCODE: begin
                        req_opcode        <= rx_byte;
                        req_checksum_accum<= req_checksum_accum ^ rx_byte;
                        rx_state          <= RX_GET_NARGS;
                    end

                    RX_GET_NARGS: begin
                        req_nargs         <= rx_byte;
                        req_checksum_accum<= req_checksum_accum ^ rx_byte;
                        arg_byte_idx      <= 3'd0;
                        args_seen         <= 3'd0;
                        if (rx_byte > MAX_SUPPORTED_NARGS) begin
                            rx_state        <= RX_WAIT_SYNC;
                            resp_status     <= STATUS_BAD_PACKET;
                            resp_result     <= 32'sd0;
                            resp_checksum   <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                            response_ready  <= 1'b1;
                        end else if (
                            ((req_opcode == OP_ADD_I32) || (req_opcode == OP_DDR_WRITE32) ||
                             (req_opcode == OP_SD_TO_DDR_COPY) || (req_opcode == OP_RUN_SAMPLE_INFER) ||
                             (req_opcode == OP_READ_SPIKE_COUNT) || (req_opcode == OP_READ_RAW_U8) ||
                             (req_opcode == OP_READ_POISSON_THRESH) || (req_opcode == OP_READ_INFER_DEBUG) ||
                             (req_opcode == OP_WRITE_INFER_WEIGHT) || (req_opcode == OP_TRAIN_QUERY_CAPS) ||
                             (req_opcode == OP_TRACE_UPDATE) || (req_opcode == OP_STDP_UPDATE_TILE))
                            && (rx_byte != 8'd2)
                        ) begin
                            rx_state        <= RX_WAIT_SYNC;
                            resp_status     <= STATUS_BAD_PACKET;
                            resp_result     <= 32'sd0;
                            resp_checksum   <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                            response_ready  <= 1'b1;
                        end else if (rx_byte == 8'd0) begin
                            rx_state <= RX_GET_CHECKSUM;
                        end else begin
                            rx_state <= RX_GET_ARGS;
                        end
                    end

                    RX_GET_ARGS: begin
                        req_checksum_accum <= req_checksum_accum ^ rx_byte;

                        if (args_seen == 3'd0) begin
                            case (arg_byte_idx)
                                3'd0: arg0[7:0]   <= rx_byte;
                                3'd1: arg0[15:8]  <= rx_byte;
                                3'd2: arg0[23:16] <= rx_byte;
                                3'd3: arg0[31:24] <= rx_byte;
                                default: ;
                            endcase
                        end else if (args_seen == 3'd1) begin
                            case (arg_byte_idx)
                                3'd0: arg1[7:0]   <= rx_byte;
                                3'd1: arg1[15:8]  <= rx_byte;
                                3'd2: arg1[23:16] <= rx_byte;
                                3'd3: arg1[31:24] <= rx_byte;
                                default: ;
                            endcase
                        end

                        if (arg_byte_idx == 3'd3) begin
                            arg_byte_idx <= 3'd0;
                            args_seen    <= args_seen + 3'd1;
                            if ((args_seen + 3'd1) == req_nargs) begin
                                rx_state <= RX_GET_CHECKSUM;
                            end
                        end else begin
                            arg_byte_idx <= arg_byte_idx + 3'd1;
                        end
                    end

                    RX_GET_CHECKSUM: begin
                        req_checksum <= rx_byte;
                        rx_state     <= RX_WAIT_SYNC;

                        if ((req_checksum_accum != rx_byte) || (req_ver != PROTO_VER)) begin
                            resp_status    <= STATUS_BAD_PACKET;
                            resp_result    <= 32'sd0;
                            resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                            response_ready <= 1'b1;
                        end else begin
                            case (req_opcode)
                                OP_ADD_I32: begin
                                    if (req_nargs == 8'd2) begin
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= arg0 + arg1;
                                        resp_checksum  <= calc_resp_checksum(STATUS_OK, arg0 + arg1);
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_DDR_WRITE32: begin
                                    if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < DDR_ADDR_WORD_LIMIT)) begin
                                        ddr_write_count <= ddr_write_count + 32'd1;
                                        ddr_last_addr <= arg0;
                                        ddr_last_data <= arg1;
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= arg0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_OK, arg0);
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_SD_TO_DDR_COPY: begin
                                    if (
                                        (req_nargs == 8'd2) &&
                                        (arg0 >= 0) &&
                                        !sd_copy_active
                                    ) begin
                                        sd_copy_active        <= 1'b1;
                                        sd_in_read            <= 1'b0;
                                        sd_copy_lba           <= arg0;
                                        sd_copy_sectors_left  <= arg1;
                                        sd_byte_count         <= 9'd0;
                                        sd_pack_idx           <= 2'd0;
                                        sd_pack_word          <= 32'd0;
                                        sd_copy_words_written <= 32'd0;
                                        sd_wait_counter       <= 24'd0;
                                        sd_header_done        <= 1'b0;
                                        sd_file_total_bytes   <= 32'd0;
                                        sd_file_bytes_seen    <= 32'd0;
                                        sd_copy_done_pending  <= 1'b0;
                                        sd_use_sector_limit   <= (arg1 > 0);
                                        raw_image0_valid      <= 1'b0;
                                        raw_image0_capture_idx <= 10'd0;
                                        raw_image0_sum_u8     <= 32'd0;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        // [31:24]=reason, [23:16]=opcode, [15:0]=arg0[15:0]
                                        resp_result    <= {BADDBG_SD_REQ_ARG, req_opcode, arg0[15:0]};
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {BADDBG_SD_REQ_ARG, req_opcode, arg0[15:0]});
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_RUN_SAMPLE_INFER: begin
                                    if (
                                        (req_nargs == 8'd2) &&
                                        (arg1 > 0) &&
                                        raw_image0_valid &&
                                        (raw_bytes_per_image == 32'd784) &&
                                        (raw_image0_sum_u8 != 32'd0) &&
                                        !infer_active
                                    ) begin
                                        infer_active       <= 1'b1;
                                        infer_state        <= INFER_INIT_CLEAR;
                                        infer_steps_target <= arg1;
                                        infer_step_idx     <= 16'd0;
                                        infer_neuron_idx   <= 7'd0;
                                        infer_input_idx    <= 10'd0;
                                        infer_prep_idx     <= 10'd0;
                                        infer_accum        <= 32'sd0;
                                        infer_accum_weight_phase <= 2'd0;
                                        infer_apply_idx    <= 7'd0;
                                        infer_sum_c_inh    <= 32'sd0;
                                        infer_total_spikes <= 32'd0;
                                        infer_rng_state    <= arg0;
                                        infer_dbg_total_input_spikes <= 32'd0;
                                        infer_dbg_total_syn_hits <= 32'd0;
                                        infer_dbg_last_step_input_spikes <= 32'd0;
                                        infer_dbg_first_step_input_spikes <= 32'd0;
                                        infer_dbg_first_step_hits_n0 <= 32'd0;
                                        infer_dbg_first_step_hits_n3 <= 32'd0;
                                        infer_dbg_first_step_hits_n7 <= 32'd0;
                                        infer_dbg_curr_step_input_spikes <= 32'd0;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_READ_SPIKE_COUNT: begin
                                    if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < N_NEURONS)) begin
                                        infer_spike_count_rd_addr <= arg0[6:0];
                                        memrd_idx     <= arg0[15:0];
                                        memrd_kind    <= MEMRD_SPIKE_COUNT;
                                        memrd_wait    <= 1'b1;
                                        memrd_pending <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        // [31:24]=reason, [23:16]=opcode, [15:0]=arg0[15:0]
                                        resp_result    <= {BADDBG_READ_SPIKE_ARG, req_opcode, arg0[15:0]};
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {BADDBG_READ_SPIKE_ARG, req_opcode, arg0[15:0]});
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_READ_RAW_U8: begin
                                    if ((req_nargs == 8'd2) && raw_image0_valid && (arg0 >= 0) && (arg0 < N_IN)) begin
                                        raw_image0_rd_addr <= arg0[9:0];
                                        memrd_idx     <= arg0[15:0];
                                        memrd_kind    <= MEMRD_RAW_U8;
                                        memrd_wait    <= 1'b1;
                                        memrd_pending <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        // [31:24]=reason, [23:16]=opcode, [15:0]=arg0[15:0]
                                        resp_result    <= {8'h12, req_opcode, arg0[15:0]};
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {8'h12, req_opcode, arg0[15:0]});
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_READ_POISSON_THRESH: begin
                                    if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < N_IN)) begin
                                        infer_poisson_thresh_rd_addr <= arg0[9:0];
                                        memrd_idx     <= arg0[15:0];
                                        memrd_kind    <= MEMRD_POISSON_THRESH;
                                        memrd_wait    <= 1'b1;
                                        memrd_pending <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        // [31:24]=reason, [23:16]=opcode, [15:0]=arg0[15:0]
                                        resp_result    <= {8'h13, req_opcode, arg0[15:0]};
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {8'h13, req_opcode, arg0[15:0]});
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_READ_INFER_DEBUG: begin
                                    if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < 32'sd12)) begin
                                        logic signed [31:0] dbg_value;
                                        resp_status <= STATUS_OK;
                                        case (arg0[4:0])
                                            5'd0: dbg_value = infer_dbg_total_input_spikes;
                                            5'd1: dbg_value = infer_dbg_total_syn_hits;
                                            5'd2: dbg_value = infer_dbg_last_step_input_spikes;
                                            5'd3: dbg_value = infer_dbg_first_step_input_spikes;
                                            5'd4: dbg_value = infer_dbg_first_step_hits_n0;
                                            5'd5: dbg_value = infer_dbg_first_step_hits_n3;
                                            5'd6: dbg_value = infer_dbg_first_step_hits_n7;
                                            5'd7: dbg_value = infer_total_spikes;
                                            5'd8: dbg_value = infer_steps_target;
                                            5'd9: dbg_value = {16'd0, infer_step_idx};
                                            5'd10: dbg_value = {29'd0, infer_state};
                                            default: dbg_value = raw_image0_sum_u8;
                                        endcase
                                        resp_result    <= dbg_value;
                                        resp_checksum  <= calc_resp_checksum(STATUS_OK, dbg_value);
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        // [31:24]=reason, [23:16]=opcode, [15:0]=arg0[15:0]
                                        resp_result    <= {BADDBG_READ_INFER_DBG, req_opcode, arg0[15:0]};
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {BADDBG_READ_INFER_DBG, req_opcode, arg0[15:0]});
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_WRITE_INFER_WEIGHT: begin
                                    if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < N_WEIGHTS) && !infer_active) begin
                                        infer_w_q16[arg0[16:0]] <= arg1[15:0];
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= arg0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_OK, arg0);
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= {BADDBG_WRITE_WEIGHT, req_opcode, arg0[15:0]};
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {BADDBG_WRITE_WEIGHT, req_opcode, arg0[15:0]});
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRAIN_QUERY_CAPS: begin
                                    if (req_nargs == 8'd2) begin
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= TRAIN_CAPS_VALUE;
                                        resp_checksum  <= calc_resp_checksum(STATUS_OK, TRAIN_CAPS_VALUE);
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRACE_UPDATE: begin
                                    // Step2 kernel opcode reserved (execution not wired yet).
                                    resp_status    <= STATUS_UNSUPPORTED_OP;
                                    resp_result    <= TRAIN_CAPS_VALUE;
                                    resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, TRAIN_CAPS_VALUE);
                                    response_ready <= 1'b1;
                                end
                                OP_STDP_UPDATE_TILE: begin
                                    // Step3 kernel opcode reserved (execution not wired yet).
                                    resp_status    <= STATUS_UNSUPPORTED_OP;
                                    resp_result    <= TRAIN_CAPS_VALUE;
                                    resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, TRAIN_CAPS_VALUE);
                                    response_ready <= 1'b1;
                                end

                                default: begin
                                    resp_status    <= STATUS_UNSUPPORTED_OP;
                                    resp_result    <= 32'sd0;
                                    resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                    response_ready <= 1'b1;
                                end
                            endcase
                        end
                    end

                    default: begin
                        rx_state <= RX_WAIT_SYNC;
                    end
                endcase
            end

            if (sd_copy_active && !response_ready) begin
                if (!sd_in_read && (!sd_use_sector_limit || (sd_copy_sectors_left != 0)) && !sd_copy_done_pending) begin
                    if (SD_CD_N != 1'b0) begin
                        sd_copy_active <= 1'b0;
                        resp_status    <= STATUS_BAD_PACKET;
                        // [31:24]=reason, [23:16]=opcode, [15]=SD_CD_N, [4:0]=sd_status
                        resp_result    <= {BADDBG_SD_CD_N, OP_SD_TO_DDR_COPY, SD_CD_N, 10'd0, sd_status};
                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {BADDBG_SD_CD_N, OP_SD_TO_DDR_COPY, SD_CD_N, 10'd0, sd_status});
                        response_ready <= 1'b1;
                    end else if (sd_ready) begin
                        sd_address    <= sd_copy_lba;
                        sd_rd         <= 1'b1;
                        sd_in_read    <= 1'b1;
                        sd_byte_count <= 9'd0;
                        sd_pack_idx   <= 2'd0;
                        sd_pack_word  <= 32'd0;
                        sd_wait_counter <= 24'd0;
                    end else begin
                        if (sd_wait_counter == 24'hFFFFFF) begin
                            sd_copy_active <= 1'b0;
                            resp_status    <= STATUS_BAD_PACKET;
                            // [31:24]=reason, [23:16]=opcode, [15:0]=wait_counter[15:0]
                            resp_result    <= {BADDBG_SD_WAIT_TO, OP_SD_TO_DDR_COPY, sd_wait_counter[15:0]};
                            resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {BADDBG_SD_WAIT_TO, OP_SD_TO_DDR_COPY, sd_wait_counter[15:0]});
                            response_ready <= 1'b1;
                        end else begin
                            sd_wait_counter <= sd_wait_counter + 24'd1;
                        end
                    end
                end

                if (sd_in_read && sd_byte_available) begin
                    if (
                        !sd_header_done ||
                        (sd_file_bytes_seen < sd_file_total_bytes)
                    ) begin
                        if (!sd_header_done && (sd_file_bytes_seen < 32'd20)) begin
                            sd_header_bytes[sd_file_bytes_seen[4:0]] <= sd_dout;
                        end

                        if (!sd_header_done && (sd_file_bytes_seen == 32'd19)) begin
                            if (
                                (sd_header_bytes[0] == 8'h52) && // 'R'
                                (sd_header_bytes[1] == 8'h41) && // 'A'
                                (sd_header_bytes[2] == 8'h57) && // 'W'
                                (sd_header_bytes[3] == 8'h31) && // '1'
                                ({sd_header_bytes[7], sd_header_bytes[6], sd_header_bytes[5], sd_header_bytes[4]} == 32'd1) &&
                                ({sd_header_bytes[15], sd_header_bytes[14], sd_header_bytes[13], sd_header_bytes[12]} == 32'd784)
                            ) begin
                                sd_header_done <= 1'b1;
                                raw_num_images <= {sd_header_bytes[11], sd_header_bytes[10], sd_header_bytes[9], sd_header_bytes[8]};
                                raw_bytes_per_image <= {sd_dout, sd_header_bytes[18], sd_header_bytes[17], sd_header_bytes[16]};
                                raw_image0_valid <= 1'b0;
                                sd_file_total_bytes <=
                                    32'd20 +
                                    {sd_header_bytes[11], sd_header_bytes[10], sd_header_bytes[9], sd_header_bytes[8]} +
                                    (
                                        {sd_header_bytes[11], sd_header_bytes[10], sd_header_bytes[9], sd_header_bytes[8]} *
                                        {sd_dout, sd_header_bytes[18], sd_header_bytes[17], sd_header_bytes[16]}
                                    );
                            end else begin
                                sd_copy_active <= 1'b0;
                                sd_in_read     <= 1'b0;
                                resp_status    <= STATUS_BAD_PACKET;
                                // [31:24]=reason, [23:16]=opcode, [15:0]=file_bytes_seen[15:0]
                                resp_result    <= {BADDBG_SD_BAD_HEADER, OP_SD_TO_DDR_COPY, sd_file_bytes_seen[15:0]};
                                resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {BADDBG_SD_BAD_HEADER, OP_SD_TO_DDR_COPY, sd_file_bytes_seen[15:0]});
                                response_ready <= 1'b1;
                            end
                        end

                        if (
                            sd_header_done &&
                            (sd_file_bytes_seen >= (32'd20 + raw_num_images)) &&
                            (raw_bytes_per_image == 32'd784) &&
                            (raw_image0_capture_idx < 10'd784)
                        ) begin
                            raw_image0_u8[raw_image0_capture_idx] <= sd_dout;
                            raw_image0_sum_u8 <= raw_image0_sum_u8 + {24'd0, sd_dout};
                            if (raw_image0_capture_idx == 10'd783) begin
                                raw_image0_valid <= 1'b1;
                            end
                            raw_image0_capture_idx <= raw_image0_capture_idx + 10'd1;
                        end

                        case (sd_pack_idx)
                            2'd0: sd_pack_word[7:0]   <= sd_dout;
                            2'd1: sd_pack_word[15:8]  <= sd_dout;
                            2'd2: sd_pack_word[23:16] <= sd_dout;
                            default: begin
                                sd_pack_word[31:24] <= sd_dout;
                                ddr_last_data       <= {sd_dout, sd_pack_word[23:0]};
                                ddr_last_addr       <= ddr_write_count;
                                ddr_write_count     <= ddr_write_count + 32'd1;
                                sd_copy_words_written <= sd_copy_words_written + 32'd1;
                            end
                        endcase

                        sd_pack_idx <= sd_pack_idx + 2'd1;
                        sd_file_bytes_seen <= sd_file_bytes_seen + 32'd1;
                    end

                    if (
                        sd_header_done &&
                        (sd_file_bytes_seen >= sd_file_total_bytes) &&
                        !sd_copy_done_pending
                    ) begin
                        sd_copy_done_pending <= 1'b1;
                    end

                    if (sd_byte_count == 9'd511) begin
                        sd_in_read   <= 1'b0;
                        sd_copy_lba  <= sd_copy_lba + 32'd1;
                        if (sd_use_sector_limit && (sd_copy_sectors_left != 0)) begin
                            sd_copy_sectors_left <= sd_copy_sectors_left - 32'd1;
                        end
                        if (sd_copy_done_pending) begin
                            if (sd_pack_idx != 2'd0) begin
                                ddr_last_data <= sd_pack_word;
                                ddr_last_addr <= ddr_write_count;
                                ddr_write_count <= ddr_write_count + 32'd1;
                                sd_copy_words_written <= sd_copy_words_written + 32'd1;
                            end
                            sd_copy_active <= 1'b0;
                            resp_status    <= STATUS_OK;
                            resp_result    <= sd_copy_words_written + ((sd_pack_idx != 2'd0) ? 32'd1 : 32'd0);
                            resp_checksum  <= calc_resp_checksum(
                                STATUS_OK,
                                sd_copy_words_written + ((sd_pack_idx != 2'd0) ? 32'd1 : 32'd0)
                            );
                            response_ready <= 1'b1;
                        end else if (sd_use_sector_limit && (sd_copy_sectors_left == 32'd1)) begin
                            sd_copy_active <= 1'b0;
                            resp_status    <= STATUS_BAD_PACKET;
                            // [31:24]=reason, [23:16]=opcode, [15:0]=sectors_left[15:0]
                            resp_result    <= {BADDBG_SD_SECTOR_END, OP_SD_TO_DDR_COPY, sd_copy_sectors_left[15:0]};
                            resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {BADDBG_SD_SECTOR_END, OP_SD_TO_DDR_COPY, sd_copy_sectors_left[15:0]});
                            response_ready <= 1'b1;
                        end
                        sd_byte_count <= 9'd0;
                        sd_pack_idx   <= 2'd0;
                    end else begin
                        sd_byte_count <= sd_byte_count + 9'd1;
                    end
                end
            end

            if (infer_active && !response_ready) begin
                case (infer_state)
                    INFER_INIT_CLEAR: begin
                        infer_v_state[infer_apply_idx] <= FXP_EXC_VRESET;
                        infer_g_in_state[infer_apply_idx] <= 32'sd0;
                        infer_exc_theta[infer_apply_idx] <= 32'sd0;
                        infer_g_in_delay0[infer_apply_idx] <= 32'sd0;
                        infer_g_in_delay1[infer_apply_idx] <= 32'sd0;
                        infer_g_in_delay2[infer_apply_idx] <= 32'sd0;
                        infer_g_in_delay3[infer_apply_idx] <= 32'sd0;
                        infer_g_in_delay4[infer_apply_idx] <= 32'sd0;
                        infer_v_inh_state[infer_apply_idx] <= FXP_INH_VRESET;
                        infer_c_inh_state[infer_apply_idx] <= 32'sd0;
                        infer_g_inh_state[infer_apply_idx] <= 32'sd0;
                        infer_g_exc_delay0[infer_apply_idx] <= 32'sd0;
                        infer_g_exc_delay1[infer_apply_idx] <= 32'sd0;
                        infer_s_exc[infer_apply_idx] <= 1'b0;
                        infer_spike_count[infer_apply_idx] <= 16'd0;
                        infer_exc_last_spike_step[infer_apply_idx] <= 16'd0;
                        infer_inh_last_spike_step[infer_apply_idx] <= 16'd0;
                        if (infer_apply_idx == (N_NEURONS - 1)) begin
                            infer_apply_idx <= 7'd0;
                            infer_prep_idx <= 10'd0;
                            infer_state <= INFER_PREP_DIV_START;
                        end else begin
                            infer_apply_idx <= infer_apply_idx + 7'd1;
                        end
                    end

                    INFER_PREP_DIV_START: begin
                        if (infer_prep_idx < N_IN) begin
                            if (!infer_div_busy) begin
                                infer_dividend <= POISSON_NUM_CONST * {24'd0, raw_image0_u8[infer_prep_idx]};
                                infer_divisor  <= raw_image0_sum_u8;
                                infer_div_valid <= 1'b1;
                                infer_state <= INFER_PREP_DIV_WAIT;
                            end
                        end else begin
	                            infer_state <= INFER_GEN_INPUT_SPIKES;
	                            infer_input_idx <= 10'd0;
	                            infer_neuron_idx <= 7'd0;
	                            infer_accum <= 32'sd0;
                                infer_accum_weight_phase <= 2'd0;
                        end
                    end

                    INFER_PREP_DIV_WAIT: begin
                        if (infer_div_out_valid) begin
                            if (infer_div_q[31:11] != 0) begin
                                infer_poisson_thresh[infer_prep_idx] <= RNG_MAX;
                            end else if (infer_div_q[10:0] > RNG_MAX) begin
                                infer_poisson_thresh[infer_prep_idx] <= RNG_MAX;
                            end else begin
                                infer_poisson_thresh[infer_prep_idx] <= infer_div_q[10:0];
                            end
                            infer_prep_idx <= infer_prep_idx + 10'd1;
                            infer_state <= INFER_PREP_DIV_START;
                        end else if (infer_div_err) begin
                            infer_active <= 1'b0;
                            infer_state <= INFER_IDLE;
                            resp_status <= STATUS_BAD_PACKET;
                            resp_result <= 32'sd0;
                            resp_checksum <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                            response_ready <= 1'b1;
                        end
                    end

                    INFER_GEN_INPUT_SPIKES: begin
                        logic [31:0] rng_next;
                        logic spike_in_now;
                        rng_next = ($unsigned(infer_rng_state) * LCG_A) + LCG_C;
                        spike_in_now = (rng_next[31:21] < infer_poisson_thresh[infer_input_idx]);
                        infer_rng_state <= rng_next;
                        infer_input_spike[infer_input_idx] <= spike_in_now;
                        if (spike_in_now) begin
                            infer_dbg_total_input_spikes <= infer_dbg_total_input_spikes + 32'd1;
                            infer_dbg_curr_step_input_spikes <= infer_dbg_curr_step_input_spikes + 32'd1;
                        end
                        if (infer_input_idx == (N_IN - 1)) begin
                            infer_dbg_last_step_input_spikes <=
                                infer_dbg_curr_step_input_spikes + (spike_in_now ? 32'd1 : 32'd0);
                            if (infer_step_idx == 16'd0) begin
                                infer_dbg_first_step_input_spikes <=
                                    infer_dbg_curr_step_input_spikes + (spike_in_now ? 32'd1 : 32'd0);
                            end
                            infer_dbg_curr_step_input_spikes <= 32'd0;
                            infer_input_idx <= 10'd0;
                            infer_state <= INFER_ACCUM_NEURON;
                        end else begin
                            infer_input_idx <= infer_input_idx + 10'd1;
                        end
                    end

                    INFER_ACCUM_NEURON: begin
                        if (infer_input_idx < N_IN) begin
                            logic [16:0] w_idx;
                            if (infer_accum_weight_phase == 2'd0) begin
                                w_idx = (infer_neuron_idx * N_IN) + infer_input_idx;
                                infer_w_rd_addr <= w_idx;
                                infer_accum_weight_phase <= 2'd1;
                            end else if (infer_accum_weight_phase == 2'd1) begin
                                // BRAM synchronous read latency fill cycle: data becomes valid next cycle.
                                infer_accum_weight_phase <= 2'd2;
                            end else begin
                                if (infer_input_spike[infer_input_idx] && (infer_w_rd_data != 16'd0)) begin
                                    infer_accum <= infer_accum + $signed({16'd0, infer_w_rd_data});
                                    infer_dbg_total_syn_hits <= infer_dbg_total_syn_hits + 32'd1;
                                    if (infer_step_idx == 16'd0) begin
                                        if (infer_neuron_idx == 7'd0) begin
                                            infer_dbg_first_step_hits_n0 <= infer_dbg_first_step_hits_n0 + 32'd1;
                                        end
                                        if (infer_neuron_idx == 7'd3) begin
                                            infer_dbg_first_step_hits_n3 <= infer_dbg_first_step_hits_n3 + 32'd1;
                                        end
                                        if (infer_neuron_idx == 7'd7) begin
                                            infer_dbg_first_step_hits_n7 <= infer_dbg_first_step_hits_n7 + 32'd1;
                                        end
                                    end
                                end
                                infer_input_idx <= infer_input_idx + 10'd1;
                                infer_accum_weight_phase <= 2'd0;
                            end
	                        end else begin
	                            logic signed [31:0] v_next;
                                logic signed [31:0] v_prop;
                                logic signed [31:0] dv_exc_step;
                                logic signed [31:0] theta_next;
                                logic signed [31:0] exc_thresh_now;
                                logic signed [31:0] g_in_curr;
                                logic signed [31:0] g_in_state_next;
                                logic signed [31:0] delayed_g_in;
                                logic signed [31:0] exc_drive_dt;
                                logic signed [31:0] inh_drive_dt;
                                logic signed [31:0] leak_dt;
                                logic signed [31:0] i_syn_exc_step;
                                logic signed [31:0] i_syn_inh_step;
			                            logic spike_now;
                                logic exc_refractory_ok;
                                g_in_state_next = $signed(($signed(infer_g_in_state[infer_neuron_idx]) * $signed(FXP_INPUT_G_DECAY)) >>> 16)
                                               + fxp_mul_s16_16(infer_accum, FXP_SCALE_1000);
                                infer_g_in_state[infer_neuron_idx] <= g_in_state_next;
	                                g_in_curr = g_in_state_next;
	                                delayed_g_in = infer_g_in_delay4[infer_neuron_idx];
	                                infer_g_in_delay4[infer_neuron_idx] <= infer_g_in_delay3[infer_neuron_idx];
	                                infer_g_in_delay3[infer_neuron_idx] <= infer_g_in_delay2[infer_neuron_idx];
                                infer_g_in_delay2[infer_neuron_idx] <= infer_g_in_delay1[infer_neuron_idx];
                                infer_g_in_delay1[infer_neuron_idx] <= infer_g_in_delay0[infer_neuron_idx];
                                infer_g_in_delay0[infer_neuron_idx] <= g_in_curr;
                                exc_refractory_ok = ((infer_step_idx - infer_exc_last_spike_step[infer_neuron_idx]) > EXC_TREF_STEPS);
                                exc_drive_dt = fxp_mul_s16_16((FXP_EXC_EEXC - infer_v_state[infer_neuron_idx]), FXP_EXC_DT_OVER_TCM);
                                inh_drive_dt = fxp_mul_s16_16((FXP_EXC_EINH - infer_v_state[infer_neuron_idx]), FXP_EXC_DT_OVER_TCM);
                                leak_dt = fxp_mul_s16_16((FXP_EXC_VREST - infer_v_state[infer_neuron_idx]), FXP_EXC_DT_OVER_TCM);
                                i_syn_exc_step = fxp_mul_s16_16(delayed_g_in, exc_drive_dt);
                                i_syn_inh_step = fxp_mul_s16_16(infer_g_inh_state[infer_neuron_idx], inh_drive_dt);
                                dv_exc_step = leak_dt + i_syn_exc_step + i_syn_inh_step;
	                                v_prop = $signed(infer_v_state[infer_neuron_idx]) + $signed(dv_exc_step);
                                v_next = exc_refractory_ok ? v_prop : infer_v_state[infer_neuron_idx];
                                exc_thresh_now = FXP_THRESH_BASE + infer_exc_theta[infer_neuron_idx];
	                            spike_now = (v_next >= exc_thresh_now);
                                theta_next = fxp_mul_s16_16(infer_exc_theta[infer_neuron_idx], FXP_THETA_DECAY);
                                if (spike_now) begin
                                    theta_next = theta_next + FXP_THETA_PLUS;
                                end
                                if (theta_next < 32'sd0) begin
                                    theta_next = 32'sd0;
                                end
                                if (theta_next > FXP_THETA_MAX) begin
                                    theta_next = FXP_THETA_MAX;
                                end
                                infer_exc_theta[infer_neuron_idx] <= theta_next;

	                            if (spike_now) begin
                                    // mine.py sets the membrane to vreset after spike (no residual carry).
	                                infer_v_state[infer_neuron_idx] <= FXP_EXC_VRESET;
	                                infer_spike_count[infer_neuron_idx] <= infer_spike_count[infer_neuron_idx] + 16'd1;
	                                infer_total_spikes <= infer_total_spikes + 32'd1;
                                    infer_exc_last_spike_step[infer_neuron_idx] <= infer_step_idx;
	                                infer_s_exc[infer_neuron_idx] <= 1'b1;
	                            end else begin
	                                infer_v_state[infer_neuron_idx] <= v_next;
                                infer_s_exc[infer_neuron_idx] <= 1'b0;
                            end

                            infer_input_idx <= 10'd0;
                            infer_accum_weight_phase <= 2'd0;
	                            if (infer_neuron_idx == (N_NEURONS - 1)) begin
	                                infer_neuron_idx <= 7'd0;
	                                infer_apply_idx <= 7'd0;
	                                infer_sum_c_inh <= 32'sd0;
	                                infer_state <= INFER_APPLY_WTA;
	                                infer_accum <= 32'sd0;
	                            end else begin
	                                infer_neuron_idx <= infer_neuron_idx + 7'd1;
	                                infer_accum <= 32'sd0;
	                            end
	                        end
	                    end

	                    INFER_APPLY_WTA: begin
	                        logic signed [31:0] g_exc_new;
	                        logic signed [31:0] delayed_g_exc;
	                        logic signed [31:0] v_inh_next;
                            logic signed [31:0] v_inh_prop;
                            logic signed [31:0] dv_inh_step;
                            logic signed [31:0] exc_drive_dt_inh;
                            logic signed [31:0] leak_dt_inh;
                            logic signed [31:0] i_syn_exc_step_inh;
	                        logic s_inh_now;
	                        logic signed [31:0] c_inh_next;
                            logic inh_refractory_ok;
	                        g_exc_new = infer_s_exc[infer_apply_idx] ? FXP_GEXC_SPIKE : 32'sd0;
	                        delayed_g_exc = infer_g_exc_delay1[infer_apply_idx];
	                        infer_g_exc_delay1[infer_apply_idx] <= infer_g_exc_delay0[infer_apply_idx];
	                        infer_g_exc_delay0[infer_apply_idx] <= g_exc_new;
                            inh_refractory_ok = ((infer_step_idx - infer_inh_last_spike_step[infer_apply_idx]) > INH_TREF_STEPS);
                            exc_drive_dt_inh = fxp_mul_s16_16((FXP_INH_EEXC - infer_v_inh_state[infer_apply_idx]), FXP_INH_DT_OVER_TCM);
                            leak_dt_inh = fxp_mul_s16_16((FXP_INH_VREST - infer_v_inh_state[infer_apply_idx]), FXP_INH_DT_OVER_TCM);
                            i_syn_exc_step_inh = fxp_mul_s16_16(delayed_g_exc, exc_drive_dt_inh);
                            dv_inh_step = leak_dt_inh + i_syn_exc_step_inh;
                            v_inh_prop = $signed(infer_v_inh_state[infer_apply_idx]) + $signed(dv_inh_step);
                            v_inh_next = inh_refractory_ok ? v_inh_prop : infer_v_inh_state[infer_apply_idx];
	                        s_inh_now = (v_inh_next >= FXP_INH_THRESH);
	                        if (s_inh_now) begin
                                // mine.py inhibitory neuron also resets to vreset after spike.
	                            infer_v_inh_state[infer_apply_idx] <= FXP_INH_VRESET;
                                infer_inh_last_spike_step[infer_apply_idx] <= infer_step_idx;
	                        end else begin
	                            infer_v_inh_state[infer_apply_idx] <= v_inh_next;
	                        end
	                        c_inh_next = $signed(infer_c_inh_state[infer_apply_idx]) >>> 1;
	                        if (s_inh_now) begin
	                            c_inh_next = c_inh_next + FXP_SCALE_500;
	                        end
	                        infer_c_inh_state[infer_apply_idx] <= c_inh_next;
	                        infer_sum_c_inh <= infer_sum_c_inh + c_inh_next;

	                        if (infer_apply_idx == (N_NEURONS - 1)) begin
	                            infer_apply_idx <= 7'd0;
	                            infer_state <= INFER_WTA_PASS2;
	                        end else begin
	                            infer_apply_idx <= infer_apply_idx + 7'd1;
	                        end
	                    end

	                    INFER_WTA_PASS2: begin
	                        logic signed [31:0] diff_c_inh;
	                        diff_c_inh = infer_sum_c_inh - infer_c_inh_state[infer_apply_idx];
	                        if (diff_c_inh < 0) begin
	                            diff_c_inh = 32'sd0;
	                        end
	                        infer_g_inh_state[infer_apply_idx] <= fxp_mul_s16_16(diff_c_inh, FXP_INH_COEFF);

	                        if (infer_apply_idx == (N_NEURONS - 1)) begin
	                            if ((infer_step_idx + 16'd1) >= infer_steps_target[15:0]) begin
	                                infer_active <= 1'b0;
	                                infer_state <= INFER_IDLE;
	                                resp_status <= STATUS_OK;
	                                resp_result <= infer_total_spikes;
	                                resp_checksum <= calc_resp_checksum(STATUS_OK, infer_total_spikes);
	                                response_ready <= 1'b1;
	                            end else begin
	                                infer_step_idx <= infer_step_idx + 16'd1;
	                                infer_state <= INFER_GEN_INPUT_SPIKES;
	                            end
	                            infer_apply_idx <= 7'd0;
	                            infer_neuron_idx <= 7'd0;
	                            infer_input_idx <= 10'd0;
	                            infer_accum <= 32'sd0;
                                infer_accum_weight_phase <= 2'd0;
	                        end else begin
	                            infer_apply_idx <= infer_apply_idx + 7'd1;
	                        end
	                    end

                    default: begin
                        infer_state <= INFER_IDLE;
                    end
                endcase
            end

            case (tx_state)
                TX_IDLE: begin
                    tx_byte_idx <= 3'd0;
                    if (response_ready) begin
                        tx_state <= TX_SEND;
                    end
                end

                TX_SEND: begin
                    tx_dv <= 1'b1;
                    case (tx_byte_idx)
                        3'd0: tx_byte <= RESP_SYNC;
                        3'd1: tx_byte <= resp_status;
                        3'd2: tx_byte <= resp_result[7:0];
                        3'd3: tx_byte <= resp_result[15:8];
                        3'd4: tx_byte <= resp_result[23:16];
                        3'd5: tx_byte <= resp_result[31:24];
                        3'd6: tx_byte <= resp_checksum;
                        default: tx_byte <= 8'h00;
                    endcase
                    tx_state <= TX_WAIT_DONE;
                end

                TX_WAIT_DONE: begin
                    if (tx_done) begin
                        if (tx_byte_idx == 3'd6) begin
                            response_ready <= 1'b0;
                            tx_state       <= TX_IDLE;
                        end else begin
                            tx_byte_idx <= tx_byte_idx + 3'd1;
                            tx_state    <= TX_SEND;
                        end
                    end
                end

                default: begin
                    tx_state <= TX_IDLE;
                end
            endcase
        end
    end

endmodule // top_level
/* I usually add a comment to associate my endmodule line with the module name
 * this helps when if you have multiple module definitions in a file
 */

module uart_rx #(
    parameter integer CLKS_PER_BIT = 868
)(
    input  wire       i_clk,
    input  wire       i_rst,
    input  wire       i_rx_serial,
    output logic      o_rx_dv,
    output logic [7:0] o_rx_byte
);
    typedef enum logic [2:0] {
        S_IDLE,
        S_START,
        S_DATA,
        S_STOP,
        S_DONE
    } rx_sm_t;

    rx_sm_t state;
    logic [15:0] clk_count;
    logic [2:0]  bit_index;
    logic [7:0]  rx_shift;

    always_ff @(posedge i_clk) begin
        if (i_rst) begin
            state      <= S_IDLE;
            clk_count  <= 16'd0;
            bit_index  <= 3'd0;
            rx_shift   <= 8'h00;
            o_rx_dv    <= 1'b0;
            o_rx_byte  <= 8'h00;
        end else begin
            o_rx_dv <= 1'b0;
            case (state)
                S_IDLE: begin
                    clk_count <= 16'd0;
                    bit_index <= 3'd0;
                    if (i_rx_serial == 1'b0) begin
                        state <= S_START;
                    end
                end

                S_START: begin
                    if (clk_count == (CLKS_PER_BIT - 1) / 2) begin
                        if (i_rx_serial == 1'b0) begin
                            clk_count <= 16'd0;
                            state     <= S_DATA;
                        end else begin
                            state <= S_IDLE;
                        end
                    end else begin
                        clk_count <= clk_count + 16'd1;
                    end
                end

                S_DATA: begin
                    if (clk_count < CLKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 16'd1;
                    end else begin
                        clk_count           <= 16'd0;
                        rx_shift[bit_index] <= i_rx_serial;
                        if (bit_index < 3'd7) begin
                            bit_index <= bit_index + 3'd1;
                        end else begin
                            bit_index <= 3'd0;
                            state     <= S_STOP;
                        end
                    end
                end

                S_STOP: begin
                    if (clk_count < CLKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 16'd1;
                    end else begin
                        o_rx_byte <= rx_shift;
                        o_rx_dv   <= 1'b1;
                        clk_count <= 16'd0;
                        state     <= S_DONE;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule

module uart_tx #(
    parameter integer CLKS_PER_BIT = 868
)(
    input  wire       i_clk,
    input  wire       i_rst,
    input  wire       i_tx_dv,
    input  wire [7:0] i_tx_byte,
    output logic      o_tx_active,
    output logic      o_tx_serial,
    output logic      o_tx_done
);
    typedef enum logic [2:0] {
        S_IDLE,
        S_START,
        S_DATA,
        S_STOP,
        S_DONE
    } tx_sm_t;

    tx_sm_t state;
    logic [15:0] clk_count;
    logic [2:0]  bit_index;
    logic [7:0]  tx_data;

    always_ff @(posedge i_clk) begin
        if (i_rst) begin
            state       <= S_IDLE;
            clk_count   <= 16'd0;
            bit_index   <= 3'd0;
            tx_data     <= 8'h00;
            o_tx_active <= 1'b0;
            o_tx_serial <= 1'b1;
            o_tx_done   <= 1'b0;
        end else begin
            o_tx_done <= 1'b0;
            case (state)
                S_IDLE: begin
                    o_tx_active <= 1'b0;
                    o_tx_serial <= 1'b1;
                    clk_count   <= 16'd0;
                    bit_index   <= 3'd0;
                    if (i_tx_dv) begin
                        tx_data     <= i_tx_byte;
                        o_tx_active <= 1'b1;
                        state       <= S_START;
                    end
                end

                S_START: begin
                    o_tx_serial <= 1'b0;
                    if (clk_count < CLKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 16'd1;
                    end else begin
                        clk_count <= 16'd0;
                        state     <= S_DATA;
                    end
                end

                S_DATA: begin
                    o_tx_serial <= tx_data[bit_index];
                    if (clk_count < CLKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 16'd1;
                    end else begin
                        clk_count <= 16'd0;
                        if (bit_index < 3'd7) begin
                            bit_index <= bit_index + 3'd1;
                        end else begin
                            bit_index <= 3'd0;
                            state     <= S_STOP;
                        end
                    end
                end

                S_STOP: begin
                    o_tx_serial <= 1'b1;
                    if (clk_count < CLKS_PER_BIT - 1) begin
                        clk_count <= clk_count + 16'd1;
                    end else begin
                        clk_count <= 16'd0;
                        o_tx_done <= 1'b1;
                        state     <= S_DONE;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
 
// reset the default net type to wire, sometimes other code expects this.
`default_nettype wire
