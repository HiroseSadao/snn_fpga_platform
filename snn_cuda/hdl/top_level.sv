`default_nettype none // prevents system from inferring an undeclared logic (good practice)
 
module top_level(
    input  wire        clk_100mhz,
    input  wire [3:0]  btn,
    input  wire [15:0] sw,
    input  wire        uart_rxd,
    output logic       uart_txd,
    inout  wire [15:0] ddr3_dq,
    inout  wire [1:0]  ddr3_dqs_n,
    inout  wire [1:0]  ddr3_dqs_p,
    output wire [13:0] ddr3_addr,
    output wire [2:0]  ddr3_ba,
    output wire        ddr3_ras_n,
    output wire        ddr3_cas_n,
    output wire        ddr3_we_n,
    output wire        ddr3_reset_n,
    output wire        ddr3_clk_p,
    output wire        ddr3_clk_n,
    output wire        ddr3_clke,
    output wire [1:0]  ddr3_dm,
    output wire        ddr3_odt,
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
    localparam logic [7:0] OP_DDR_READ32 = 8'h12;
    localparam logic [7:0] OP_SD_SECTORS_TO_DDR = 8'h13;
    localparam logic [7:0] OP_LOAD_IMAGE_FROM_DDR = 8'h14;
    localparam logic [7:0] OP_DDR_ZERO32 = 8'h15;
    localparam logic [7:0] OP_RUN_SAMPLE_INFER = 8'h20;
    localparam logic [7:0] OP_READ_SPIKE_COUNT = 8'h21;
    localparam logic [7:0] OP_READ_RAW_U8 = 8'h22;
    localparam logic [7:0] OP_READ_POISSON_THRESH = 8'h23;
    localparam logic [7:0] OP_READ_INFER_DEBUG = 8'h24;
    localparam logic [7:0] OP_WRITE_INFER_WEIGHT = 8'h25;
    localparam logic [7:0] OP_SET_POISSON_MAX_FR = 8'h26;
    localparam logic [7:0] OP_READ_TRAIN_INJ_SPIKE_COUNT = 8'h27;
    localparam logic [7:0] OP_TRAIN_QUERY_CAPS = 8'h30;
    localparam logic [7:0] OP_TRACE_UPDATE = 8'h31;
    localparam logic [7:0] OP_STDP_UPDATE_TILE = 8'h32;
    localparam logic [7:0] OP_TRAIN_GEN_WORK = 8'h33;
    localparam logic [7:0] OP_READ_TRAIN_DEBUG = 8'h34;
    localparam logic [7:0] OP_STDP_UPDATE_ALL = 8'h35;
    localparam logic [7:0] OP_TRAIN_RUN_CHUNK = 8'h36;
    localparam logic [7:0] OP_TRAIN_RUN_SAMPLE_PHASE3 = 8'h37;
    localparam logic [7:0] OP_TRAIN_RUN_SAMPLE_PHASE4 = 8'h38;
    localparam logic [7:0] OP_TRAIN_LABEL_STATS_RESET = 8'h39;
    localparam logic [7:0] OP_TRAIN_LABEL_STATS_ACCUM = 8'h3A;
    localparam logic [7:0] OP_READ_TRAIN_LABEL_STAT_SUM = 8'h3B;
    localparam logic [7:0] OP_READ_TRAIN_LABEL_STAT_COUNT = 8'h3C;
    localparam logic [31:0] DDR_ADDR_WORD_LIMIT = 32'd16777216; // 64MiB / 4
    localparam logic [7:0] MAX_SUPPORTED_NARGS = 8'd2;
    // Increase RX timeout margin to tolerate host-side inter-byte gaps on UART.
    localparam int RX_TIMEOUT_CLKS = CLKS_PER_BIT * 2000;
    localparam int N_IN = 784;
    localparam int N_NEURONS = 100;
    localparam int N_WEIGHTS = N_IN * N_NEURONS;
    localparam int TRAIN_MINE_NT_BLANK = 150;
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
    // Build switch: keep training kernels enabled for mine.py-aligned learning builds.
    // Set to 1'b0 only for inference-only fast-build iteration.
    localparam logic TRAIN_ENABLE = 1'b1;
    // Release-oriented switch: disable development/self-check UART opcodes that are not needed
    // for the final mine.py-like train/test flow (phase3/phase4 + aggregate stats).
    localparam logic DEV_UART_OPS_ENABLE = 1'b0;

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
    // [0]=query_caps impl, [1]=logical DDR map fixed, [2]=trace opcode present,
    // [3]=tile opcode present, [8]=trace kernel exec impl, [9]=tile kernel exec impl,
    // [10]=train work generation helper impl, [11]=stdp all-rows batch impl,
    // [12]=train chunk runner (phase0 skeleton) impl, [13]=phase3 sample flow impl,
    // [14]=phase4 retry sample flow impl.
    localparam logic [31:0] TRAIN_CAPS_VALUE = 32'h00007F0F;
    // Step1 logical DDR word map contract (future external DDR integration target).
    localparam logic [31:0] TRAIN_BASE_W_Q16_WORDS  = 32'd0;
    localparam logic [31:0] TRAIN_BASE_A_Q16_WORDS  = TRAIN_BASE_W_Q16_WORDS + N_WEIGHTS;
    localparam logic [31:0] TRAIN_BASE_BT_Q16_WORDS = TRAIN_BASE_A_Q16_WORDS + N_WEIGHTS;
    localparam logic [31:0] TRAIN_BASE_THETA_WORDS  = TRAIN_BASE_BT_Q16_WORDS + N_WEIGHTS;
    localparam logic [31:0] TRAIN_BASE_VSTATE_WORDS = TRAIN_BASE_THETA_WORDS + N_NEURONS;
    localparam logic [31:0] TRAIN_BASE_DELAY_WORDS  = TRAIN_BASE_VSTATE_WORDS + N_NEURONS;
    localparam logic [31:0] TRAIN_BASE_GIN_WORDS    = TRAIN_BASE_DELAY_WORDS + (N_NEURONS * 8);
    // Fixed training kernel workspaces (host preloads before OP_TRACE_UPDATE)
    localparam logic [31:0] TRAIN_BASE_XIN_WORK_WORDS    = TRAIN_BASE_GIN_WORDS + N_NEURONS;
    localparam logic [31:0] TRAIN_BASE_XEXC_WORK_WORDS   = TRAIN_BASE_XIN_WORK_WORDS + N_IN;
    localparam logic [31:0] TRAIN_BASE_PRELIST_WORK_WORDS= TRAIN_BASE_XEXC_WORK_WORDS + N_NEURONS;
    // STDP tile kernel (phase1) fixed-point params, q16.16.
    localparam logic signed [31:0] TRAIN_WMAX_Q16    = 32'sd3277; // 0.05
    localparam logic signed [31:0] TRAIN_WMIN_Q16    = 32'sd0;
    localparam logic signed [31:0] TRAIN_NORM_Q16    = 32'sd6554; // 0.1
    localparam logic signed [31:0] TRAIN_LR_P_Q16    = 32'sd655;  // 1e-2
    localparam logic signed [31:0] TRAIN_LR_M_Q16    = 32'sd7;    // 1e-4
    localparam logic signed [31:0] TRAIN_CLIP_DW_Q16 = 32'sd66;   // 1e-3
    localparam logic [31:0]        TRAIN_UPDATE_NT   = 32'd100;
    localparam logic [31:0]        TRAIN_RETRY_MIN_INJ_SPIKES = 32'd5;
    localparam logic [15:0]        TRAIN_RETRY_MAX_FR_START   = 16'd32;
    localparam logic [15:0]        TRAIN_RETRY_MAX_FR_STEP    = 16'd16;
    localparam logic [15:0]        TRAIN_RETRY_MAX_FR_LIMIT   = 16'd256;

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
    typedef enum logic [2:0] {
        MEMRD_NONE,
        MEMRD_SPIKE_COUNT,
        MEMRD_RAW_U8,
        MEMRD_POISSON_THRESH,
        MEMRD_TRAIN_INJ_SPIKE_COUNT,
        MEMRD_TRAIN_LABEL_STAT_SUM,
        MEMRD_TRAIN_LABEL_STAT_COUNT
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
    typedef enum logic [1:0] {
        DDRBR_IDLE,
        DDRBR_ISSUE,
        DDRBR_WAIT_ACK
    } ddr_bridge_state_t;
    typedef enum logic [4:0] {
        TRK_IDLE,
        TRK_A_READ_X_REQ,
        TRK_A_READ_X_WAIT,
        TRK_A_READ_A_REQ,
        TRK_A_READ_A_WAIT,
        TRK_A_WRITE_A_REQ,
        TRK_A_WRITE_A_WAIT,
        TRK_B_READ_PRE_REQ,
        TRK_B_READ_PRE_WAIT,
        TRK_B_READ_X_REQ,
        TRK_B_READ_X_WAIT,
        TRK_B_READ_BT_REQ,
        TRK_B_READ_BT_WAIT,
        TRK_B_WRITE_BT_REQ,
        TRK_B_WRITE_BT_WAIT,
        TRK_DONE
    } train_trace_state_t;
    typedef enum logic [4:0] {
        TSK_IDLE,
        TSK_SUM_READ_W_REQ,
        TSK_SUM_READ_W_WAIT,
        TSK_READ_W_REQ,
        TSK_READ_W_WAIT,
        TSK_READ_A_REQ,
        TSK_READ_A_WAIT,
        TSK_READ_BT_REQ,
        TSK_READ_BT_WAIT,
        TSK_DIV_NORM_START,
        TSK_DIV_NORM_WAIT,
        TSK_DIV_DW_START,
        TSK_DIV_DW_WAIT,
        TSK_WRITE_W_REQ,
        TSK_WRITE_W_WAIT,
        TSK_DONE
    } train_stdp_state_t;
    typedef enum logic [1:0] {
        TGK_IDLE,
        TGK_WRITE_REQ,
        TGK_WRITE_WAIT,
        TGK_DONE
    } train_gen_state_t;
    typedef enum logic [2:0] {
        TLS_IDLE,
        TLS_RESET_SUM,
        TLS_RESET_COUNT,
        TLS_ACCUM_READ,
        TLS_ACCUM_WAIT,
        TLS_ACCUM_WRITE,
        TLS_DONE
    } train_label_stats_state_t;
    typedef enum logic [4:0] {
        TCK_IDLE,
        TCK_INFER_START,
        TCK_INFER_WAIT,
        TCK_PICK_WINNER_INIT,
        TCK_PICK_WINNER_SCAN,
        TCK_BLANK_INFER_START,
        TCK_BLANK_INFER_WAIT,
        TCK_GEN_XIN_START,
        TCK_GEN_XIN_WAIT,
        TCK_GEN_XEXC_START,
        TCK_GEN_XEXC_WAIT,
        TCK_BUILD_XIN_INIT,
        TCK_BUILD_XIN_SCAN,
        TCK_BUILD_XEXC_INIT,
        TCK_BUILD_XEXC_SCAN,
        TCK_PRELIST_BUILD_INIT,
        TCK_PRELIST_BUILD_SCAN,
        TCK_PRELIST_WRITE_REQ,
        TCK_PRELIST_WRITE_WAIT,
        TCK_TRACE_START,
        TCK_TRACE_WAIT,
        TCK_STDP_START,
        TCK_STDP_WAIT,
        TCK_DONE
    } train_chunk_state_t;

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
    wire         clk_controller;
    wire         clk_ddr3;
    wire         clk_ddr3_90;
    wire         clk_ref_200;
    wire         clk_100mhz_buf;
    wire         ddr_clk_wiz_locked;

    // DDR3 Wishbone interface (controller clock domain)
    logic        ddr_wb_stb;
    logic        ddr_wb_we;
    logic [23:0] ddr_wb_addr;
    logic [127:0] ddr_wb_wdata;
    logic [15:0] ddr_wb_sel;
    wire         ddr_wb_stall;
    wire         ddr_wb_ack;
    wire [127:0] ddr_wb_rdata;
    wire         ddr_calib_complete;

    // Core<->DDR bridge for UART DDR read/write smoke test
    logic        ddr_req_pending_core;
    logic        ddr_req_we_core;
    logic [31:0] ddr_req_addr_word_core;
    logic [31:0] ddr_req_wdata_core;
    logic        ddr_req_wide_core;
    logic [127:0] ddr_req_wdata128_core;
    logic [15:0] ddr_req_sel16_core;
    logic [2:0]  ddr_req_word_count_core;
    logic [31:0] ddr_resp_rdata_async;
    logic [7:0]  ddr_resp_status_async;
    logic        ddr_req_toggle_core;
    logic        ddr_rsp_toggle_core_sync1, ddr_rsp_toggle_core_sync2;
    logic        ddr_rsp_toggle_core_seen;

    logic        ddr_req_toggle_ddr_sync1, ddr_req_toggle_ddr_sync2;
    logic        ddr_req_toggle_ddr_seen;
    logic        ddr_rsp_toggle_ddr;
    ddr_bridge_state_t ddr_bridge_state;
    logic        ddr_req_we_ddr;
    logic [31:0] ddr_req_addr_word_ddr;
    logic [31:0] ddr_req_wdata_ddr;
    logic        ddr_req_wide_ddr;
    logic [127:0] ddr_req_wdata128_ddr;
    logic [15:0] ddr_req_sel16_ddr;
    logic [2:0]  ddr_req_word_count_ddr;
    logic [31:0] ddr_rsp_rdata_ddr;
    logic [7:0]  ddr_rsp_status_ddr;
    logic [1:0]  ddr_lane_sel_ddr;
    logic        ddr_req_from_sd_core;
    logic        ddr_req_from_sd_ddr;
    logic        ddr_req_from_imgload_core;
    logic        ddr_req_from_train_core;

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
    logic [31:0] sd_sector_ddr_base_word_bank [0:1];
    logic [7:0]  sd_sector_words_queued_bank [0:1];
    logic [1:0]  sd_sector_buf_ready;
    logic        sd_fill_bank;
    logic        sd_flush_bank;
    logic        sd_ddr_flush_active;
    logic [7:0]  sd_ddr_flush_idx;
    (* ram_style = "block" *) logic [31:0] sd_sector_word_buf [0:1][0:127];
    logic [23:0] sd_wait_counter;
    logic [7:0]  sd_header_bytes [0:19];
    logic        sd_header_done;
    logic [31:0] sd_file_total_bytes;
    logic [31:0] sd_file_bytes_seen;
    logic        sd_copy_done_pending;
    logic        sd_use_sector_limit;
    logic        sd_copy_raw1_mode;
    logic [31:0] sd_copy_dest_base_word;
    (* ram_style = "block" *) logic [7:0]  raw_image0_u8 [0:N_IN-1];
    logic        raw_image0_valid;
    logic [31:0] raw_num_images;
    logic [31:0] raw_bytes_per_image;
    logic [9:0]  raw_image0_capture_idx;
    logic [31:0] raw_image0_sum_u8;
    logic [9:0]  raw_image0_rd_addr;
    logic [7:0]  raw_image0_rd_data;
    logic        imgload_active;
    logic [31:0] imgload_addr_word;
    logic [1:0]  imgload_lane;
    logic [9:0]  imgload_byte_idx;
    logic [9:0]  imgload_total_bytes;
    logic [31:0] imgload_sum_u8_accum;
    logic        imgload_word_valid;
    logic [31:0] imgload_word_data;
    logic [1:0]  imgload_word_lane;
    logic        train_trace_active;
    train_trace_state_t train_trace_state;
    logic [6:0]  train_winner_idx;
    logic [9:0]  train_pre_count;
    logic [9:0]  train_a_idx;
    logic [9:0]  train_pre_idx;
    logic [6:0]  train_b_col_idx;
    logic [9:0]  train_curr_pre;
    logic [31:0] train_tmp_x_val;
    logic [31:0] train_tmp_mem_val;
    logic [31:0] train_trace_a_row_base;
    logic [31:0] train_trace_bt_pre_base;
    logic        train_stdp_active;
    train_stdp_state_t train_stdp_state;
    logic [6:0]  train_stdp_row0;
    logic [6:0]  train_stdp_row_end;
    logic [6:0]  train_stdp_row_idx;
    logic [9:0]  train_stdp_col_idx;
    logic signed [31:0] train_stdp_w_val;
    logic signed [31:0] train_stdp_a_val;
    logic signed [31:0] train_stdp_bt_val;
    logic signed [31:0] train_stdp_w_new;
    logic signed [31:0] train_stdp_w_norm_q16;
    logic signed [31:0] train_stdp_dW_q16;
    logic [31:0] train_stdp_row_sum_abs;
    logic [31:0] train_stdp_w_row_base;
    logic [31:0] train_stdp_a_row_base;
    logic [31:0] train_stdp_bt_col_base;
    logic [31:0] train_stdp_dividend;
    logic [31:0] train_stdp_divisor;
    logic        train_stdp_div_valid;
    logic [31:0] train_stdp_div_q;
    logic [31:0] train_stdp_div_r;
    logic        train_stdp_div_out_valid;
    logic        train_stdp_div_err;
    logic        train_stdp_div_busy;
    logic        train_stdp_batch_active;
    logic [6:0]  train_stdp_batch_tile_rows;
    logic [6:0]  train_stdp_batch_next_row0;
    logic        train_chunk_active;
    train_chunk_state_t train_chunk_state;
    logic [2:0]  train_chunk_mode; // 0=phase0,1=phase1,2=phase2,3=phase3,4=phase4(retry)
    logic [15:0] train_chunk_samples_left;
    logic [6:0]  train_chunk_tile_rows;
    logic [15:0] train_chunk_steps_left;
    logic [31:0] train_chunk_seed_xin;
    logic [31:0] train_chunk_seed_xexc;
    logic [6:0]  train_chunk_winner;
    logic [9:0]  train_chunk_pre_idx;
    logic [9:0]  train_chunk_pre_from_infer;
    logic [9:0]  train_chunk_pre_scan_idx;
    logic [9:0]  train_chunk_pre_write_count;
    logic [6:0]  train_chunk_winner_scan_idx;
    logic [6:0]  train_chunk_winner_best_idx;
    logic [15:0] train_chunk_winner_best_count;
    logic [31:0] train_chunk_last_infer_spikes;
    logic [31:0] train_chunk_last_blank_spikes;
    logic [15:0] train_chunk_retry_curr_max_fr;
    logic [15:0] train_chunk_retry_accepted_max_fr;
    logic        train_gen_active;
    train_gen_state_t train_gen_state;
    logic [31:0] train_gen_base_word;
    logic [15:0] train_gen_count_total;
    logic [15:0] train_gen_idx;
    logic [31:0] train_gen_lcg_state;
    logic [31:0] train_gen_curr_word;
    logic        train_gen_lcg_enable;
    logic [1:0]  train_gen_cache_mode; // 0=none,1=x_in,2=x_exc
    logic        train_label_stats_active;
    train_label_stats_state_t train_label_stats_state;
    logic [3:0]  train_label_stats_label;
    logic [9:0]  train_label_stats_idx;
    logic [9:0]  train_label_stats_base_idx;
    logic        train_xin_cache_valid;
    logic        train_xexc_cache_valid;
    (* ram_style = "block" *) logic [31:0] train_xin_cache [0:N_IN-1];
    (* ram_style = "block" *) logic [31:0] train_xexc_cache [0:N_NEURONS-1];
    (* ram_style = "block" *) logic [31:0] train_label_spike_sum [0:(10*N_NEURONS)-1];
    (* ram_style = "block" *) logic [31:0] train_label_count [0:9];
    logic [9:0]  train_label_sum_rd_addr;
    logic [31:0] train_label_sum_rd_data;
    logic [3:0]  train_label_count_rd_addr;
    logic [31:0] train_label_count_rd_data;

    logic        infer_active;
    infer_state_t infer_state;
    logic [31:0] infer_steps_target;
    logic [15:0] infer_step_idx;
    logic [6:0]  infer_neuron_idx;
    logic [9:0]  infer_input_idx;
    logic [9:0]  infer_prep_idx;
    logic signed [31:0] infer_accum;
    logic [2:0]  infer_accum_weight_phase;
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
    (* ram_style = "block" *) logic [15:0] train_inj_spike_count_snap [0:N_NEURONS-1];
    logic [6:0]  infer_spike_count_rd_addr;
    logic [15:0] infer_spike_count_rd_data;
    logic [15:0] train_inj_spike_count_snap_rd_data;
    (* ram_style = "block" *) logic [15:0] infer_exc_last_spike_step [0:N_NEURONS-1];
    (* ram_style = "block" *) logic [15:0] infer_inh_last_spike_step [0:N_NEURONS-1];
    logic [6:0]  infer_apply_idx;
    logic signed [31:0] infer_sum_c_inh;
    logic [31:0] infer_total_spikes;
    logic [31:0] infer_rng_state;
    logic [31:0] infer_poisson_num_const_cfg;
    (* ram_style = "block" *) logic [10:0] infer_poisson_thresh [0:N_IN-1];
    logic [9:0]  infer_poisson_thresh_rd_addr;
    logic [10:0] infer_poisson_thresh_rd_data;
    (* ram_style = "block" *) logic [9:0]  infer_pre_active_list [0:N_IN-1];
    logic [9:0]  infer_pre_active_count;
    logic [9:0]  infer_pre_rd_addr;
    logic [9:0]  infer_pre_rd_data;
    logic        infer_input_spike [0:N_IN-1];
    logic [9:0]  infer_last_active_input_idx;
    logic [31:0] infer_dividend;
    logic [31:0] infer_divisor;
    logic        infer_div_valid;
    logic [31:0] infer_div_q;
    logic [31:0] infer_div_r;
    logic        infer_div_out_valid;
    logic        infer_div_err;
    logic        infer_div_busy;
    logic        infer_skip_init_clear;
    logic        infer_force_no_input;
    logic        memrd_pending;
    logic        memrd_wait;
    memrd_kind_t memrd_kind;
    logic [15:0] memrd_idx;

    wire [7:0] r_in = {sw[15:11], 3'b000};
    wire [7:0] g_in = {sw[10:5],  2'b00};
    wire [7:0] b_in = {sw[4:0],   3'b000};

    assign SD_DQ1 = 1'b1;
    assign SD_DQ2 = 1'b1;

    assign rgb0[2] = tx_active;  // blue LED: UART TX active
    assign rgb0[1] = ddr_write_count[0]; // green LED: DDR write activity bit
    assign rgb0[0] = (resp_status == STATUS_OK); // red LED: OK result

    assign rgb1 = 3'b000;
    // Minimal always-on debug for lightweight builds:
    // led[7]=train_chunk_active, [6]=train_label_stats_active, [5]=train_stdp_active,
    // [4]=train_trace_active, [3]=ddr_req_pending_core, [2]=imgload_active,
    // [1]=sd_copy_active, [0]=response_ready
    assign led[7] = train_chunk_active;
    assign led[6] = train_label_stats_active;
    assign led[5] = train_stdp_active;
    assign led[4] = train_trace_active;
    assign led[3] = ddr_req_pending_core;
    assign led[2] = imgload_active;
    assign led[1] = sd_copy_active;
    assign led[0] = response_ready;
    assign led[14] = ddr_clk_wiz_locked;         // DDR clock wizard lock
    assign led[15] = ddr_calib_complete;         // DDR3 calibration done
    // State code (6 bits):
    // chunk active -> {0, train_chunk_state[4:0]}
    // else label-stats active -> {3'b100, train_label_stats_state[2:0]}
    // else STDP active -> {1'b1, train_stdp_state[4:0]}
    assign led[13:8] = train_chunk_active ? {1'b0, train_chunk_state} :
                       (train_label_stats_active ? {3'b100, train_label_stats_state} :
                        (train_stdp_active ? {1'b1, train_stdp_state} : 6'h00));
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

    always_ff @(posedge clk_100mhz_buf) begin
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
        infer_pre_rd_data <= infer_pre_active_list[infer_pre_rd_addr];
        raw_image0_rd_data <= raw_image0_u8[raw_image0_rd_addr];
        infer_spike_count_rd_data <= infer_spike_count[infer_spike_count_rd_addr];
        train_inj_spike_count_snap_rd_data <= train_inj_spike_count_snap[infer_spike_count_rd_addr];
        train_label_sum_rd_data <= train_label_spike_sum[train_label_sum_rd_addr];
        train_label_count_rd_data <= train_label_count[train_label_count_rd_addr];
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

    lab06_clk_wiz u_ddr_clk_wiz (
        .clk_controller  (clk_controller),
        .clk_ddr3        (clk_ddr3),
        .clk_ddr3_90     (clk_ddr3_90),
        .clk_camera      (clk_ref_200), // 200 MHz output reused as DDR IDELAY refclk
        .clk_xc          (),
        .clk_passthrough (clk_100mhz_buf),
        .reset           (btn[0]),
        .locked          (ddr_clk_wiz_locked),
        .clk_in1         (clk_100mhz)
    );

    ddr3_top #(
        .CONTROLLER_CLK_PERIOD(12_000),
        .DDR3_CLK_PERIOD(3_000),
        .ROW_BITS(14),
        .COL_BITS(10),
        .BA_BITS(3),
        .BYTE_LANES(2),
        .AUX_WIDTH(16),
        .WB2_ADDR_BITS(32),
        .WB2_DATA_BITS(32),
        .MICRON_SIM(0),
        .ODELAY_SUPPORTED(0),
        .SECOND_WISHBONE(0),
        .ECC_ENABLE(0),
        .WB_ERROR(0)
    ) u_ddr3_top (
        .i_controller_clk(clk_controller),
        .i_ddr3_clk      (clk_ddr3),
        .i_ref_clk       (clk_ref_200),
        .i_ddr3_clk_90   (clk_ddr3_90),
        .i_rst_n         (!btn[0] && ddr_clk_wiz_locked),

        .i_wb_cyc        (1'b1),
        .i_wb_stb        (ddr_wb_stb),
        .i_wb_we         (ddr_wb_we),
        .i_wb_addr       (ddr_wb_addr),
        .i_wb_data       (ddr_wb_wdata),
        .i_wb_sel        (ddr_wb_sel),
        .i_aux           ({15'd0, ddr_wb_we}),
        .o_wb_stall      (ddr_wb_stall),
        .o_wb_ack        (ddr_wb_ack),
        .o_wb_err        (),
        .o_wb_data       (ddr_wb_rdata),
        .o_aux           (),

        .i_wb2_cyc       (1'b0),
        .i_wb2_stb       (1'b0),
        .i_wb2_we        (1'b0),
        .i_wb2_addr      (32'd0),
        .i_wb2_data      (32'd0),
        .i_wb2_sel       (4'd0),
        .o_wb2_stall     (),
        .o_wb2_ack       (),
        .o_wb2_data      (),

        .o_ddr3_clk_p    (ddr3_clk_p),
        .o_ddr3_clk_n    (ddr3_clk_n),
        .o_ddr3_reset_n  (ddr3_reset_n),
        .o_ddr3_cke      (ddr3_clke),
        .o_ddr3_cs_n     (),
        .o_ddr3_ras_n    (ddr3_ras_n),
        .o_ddr3_cas_n    (ddr3_cas_n),
        .o_ddr3_we_n     (ddr3_we_n),
        .o_ddr3_addr     (ddr3_addr),
        .o_ddr3_ba_addr  (ddr3_ba),
        .io_ddr3_dq      (ddr3_dq),
        .io_ddr3_dqs     (ddr3_dqs_p),
        .io_ddr3_dqs_n   (ddr3_dqs_n),
        .o_ddr3_dm       (ddr3_dm),
        .o_ddr3_odt      (ddr3_odt),
        .o_calib_complete(ddr_calib_complete),
        .o_debug1        (),
        .i_user_self_refresh(1'b0),
        .uart_tx         ()
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

    divider2b #(.WIDTH(32)) u_stdp_divider (
        .clk_in        (core_clk),
        .rst_in        (btn[0]),
        .dividend_in   (train_stdp_dividend),
        .divisor_in    (train_stdp_divisor),
        .data_valid_in (train_stdp_div_valid),
        .quotient_out  (train_stdp_div_q),
        .remainder_out (train_stdp_div_r),
        .data_valid_out(train_stdp_div_out_valid),
        .error_out     (train_stdp_div_err),
        .busy_out      (train_stdp_div_busy)
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

    function automatic [7:0] lane_byte_sel(
        input [31:0] w,
        input [1:0] lane
    );
        begin
            case (lane)
                2'd0: lane_byte_sel = w[7:0];
                2'd1: lane_byte_sel = w[15:8];
                2'd2: lane_byte_sel = w[23:16];
                default: lane_byte_sel = w[31:24];
            endcase
        end
    endfunction

    function automatic [31:0] s32_abs_u(
        input signed [31:0] v
    );
        begin
            if (v < 0) begin
                s32_abs_u = $unsigned(-v);
            end else begin
                s32_abs_u = $unsigned(v);
            end
        end
    endfunction

    function automatic [31:0] train_gen_word_from_state(
        input [31:0] lcg_state_in
    );
        begin
            // Deterministic selfcheck helper values in [0, 32767].
            train_gen_word_from_state = {17'd0, lcg_state_in[22:8]};
        end
    endfunction

    always_ff @(posedge clk_controller) begin
        if (btn[0] || !ddr_clk_wiz_locked) begin
            ddr_wb_stb <= 1'b0;
            ddr_wb_we  <= 1'b0;
            ddr_wb_addr <= 24'd0;
            ddr_wb_wdata <= 128'd0;
            ddr_wb_sel <= 16'd0;
            ddr_req_toggle_ddr_sync1 <= 1'b0;
            ddr_req_toggle_ddr_sync2 <= 1'b0;
            ddr_req_toggle_ddr_seen  <= 1'b0;
            ddr_rsp_toggle_ddr <= 1'b0;
            ddr_bridge_state <= DDRBR_IDLE;
            ddr_req_we_ddr <= 1'b0;
            ddr_req_from_sd_ddr <= 1'b0;
            ddr_req_addr_word_ddr <= 32'd0;
            ddr_req_wdata_ddr <= 32'd0;
            ddr_req_wide_ddr <= 1'b0;
            ddr_req_wdata128_ddr <= 128'd0;
            ddr_req_sel16_ddr <= 16'd0;
            ddr_req_word_count_ddr <= 3'd0;
            ddr_rsp_rdata_ddr <= 32'd0;
            ddr_rsp_status_ddr <= STATUS_BAD_PACKET;
            ddr_lane_sel_ddr <= 2'd0;
            ddr_resp_rdata_async <= 32'd0;
            ddr_resp_status_async <= STATUS_BAD_PACKET;
        end else begin
            ddr_req_toggle_ddr_sync1 <= ddr_req_toggle_core;
            ddr_req_toggle_ddr_sync2 <= ddr_req_toggle_ddr_sync1;
            ddr_wb_stb <= 1'b0;

            case (ddr_bridge_state)
                DDRBR_IDLE: begin
                    if (ddr_req_toggle_ddr_sync2 != ddr_req_toggle_ddr_seen) begin
                        ddr_req_toggle_ddr_seen <= ddr_req_toggle_ddr_sync2;
                        // Core domain holds payload stable until response toggle is observed.
                        ddr_req_we_ddr <= ddr_req_we_core;
                        ddr_req_from_sd_ddr <= ddr_req_from_sd_core;
                        ddr_req_addr_word_ddr <= ddr_req_addr_word_core;
                        ddr_req_wdata_ddr <= ddr_req_wdata_core;
                        ddr_req_wide_ddr <= ddr_req_wide_core;
                        ddr_req_wdata128_ddr <= ddr_req_wdata128_core;
                        ddr_req_sel16_ddr <= ddr_req_sel16_core;
                        ddr_req_word_count_ddr <= ddr_req_word_count_core;
                        ddr_lane_sel_ddr <= ddr_req_addr_word_core[1:0];
                        ddr_bridge_state <= DDRBR_ISSUE;
                    end
                end

                DDRBR_ISSUE: begin
                    if (!ddr_calib_complete) begin
                        ddr_resp_status_async <= STATUS_BAD_PACKET;
                        ddr_resp_rdata_async  <= 32'sd0;
                        ddr_rsp_toggle_ddr    <= ~ddr_rsp_toggle_ddr;
                        ddr_bridge_state      <= DDRBR_IDLE;
                    end else if (!ddr_wb_stall) begin
                        ddr_wb_we   <= ddr_req_we_ddr;
                        ddr_wb_addr <= {2'b00, ddr_req_addr_word_ddr[23:2]};
                        if (ddr_req_we_ddr && ddr_req_wide_ddr) begin
                            ddr_wb_wdata <= ddr_req_wdata128_ddr;
                            ddr_wb_sel   <= ddr_req_sel16_ddr;
                        end else if (ddr_req_we_ddr) begin
                            case (ddr_req_addr_word_ddr[1:0])
                                2'd0: begin ddr_wb_wdata <= {96'd0, ddr_req_wdata_ddr}; ddr_wb_sel <= 16'h000F; end
                                2'd1: begin ddr_wb_wdata <= {64'd0, ddr_req_wdata_ddr, 32'd0}; ddr_wb_sel <= 16'h00F0; end
                                2'd2: begin ddr_wb_wdata <= {32'd0, ddr_req_wdata_ddr, 64'd0}; ddr_wb_sel <= 16'h0F00; end
                                default: begin ddr_wb_wdata <= {ddr_req_wdata_ddr, 96'd0}; ddr_wb_sel <= 16'hF000; end
                            endcase
                        end else begin
                            ddr_wb_wdata <= 128'd0;
                            ddr_wb_sel   <= 16'h0000;
                        end
                        ddr_wb_stb <= 1'b1;
                        ddr_bridge_state <= DDRBR_WAIT_ACK;
                    end
                end

                DDRBR_WAIT_ACK: begin
                    // Hold stb asserted until ack to avoid missing write/read accepts on slaves
                    // that expect stb to remain high while the request is outstanding.
                    ddr_wb_stb <= 1'b1;
                    if (ddr_wb_ack) begin
                        if (ddr_req_we_ddr) begin
                            ddr_resp_status_async <= STATUS_OK;
                            ddr_resp_rdata_async  <= ddr_req_addr_word_ddr;
                        end else begin
                            ddr_resp_status_async <= STATUS_OK;
                            case (ddr_lane_sel_ddr)
                                2'd0: ddr_resp_rdata_async <= ddr_wb_rdata[31:0];
                                2'd1: ddr_resp_rdata_async <= ddr_wb_rdata[63:32];
                                2'd2: ddr_resp_rdata_async <= ddr_wb_rdata[95:64];
                                default: ddr_resp_rdata_async <= ddr_wb_rdata[127:96];
                            endcase
                        end
                        ddr_rsp_toggle_ddr <= ~ddr_rsp_toggle_ddr;
                        ddr_wb_stb <= 1'b0;
                        ddr_bridge_state <= DDRBR_IDLE;
                    end
                end

                default: begin
                    ddr_bridge_state <= DDRBR_IDLE;
                end
            endcase
        end
    end

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
            ddr_req_pending_core <= 1'b0;
            ddr_req_we_core      <= 1'b0;
            ddr_req_from_sd_core <= 1'b0;
            ddr_req_from_imgload_core <= 1'b0;
            ddr_req_from_train_core <= 1'b0;
            ddr_req_addr_word_core <= 32'd0;
            ddr_req_wdata_core   <= 32'd0;
            ddr_req_wide_core    <= 1'b0;
            ddr_req_wdata128_core <= 128'd0;
            ddr_req_sel16_core   <= 16'd0;
            ddr_req_word_count_core <= 3'd0;
            ddr_req_toggle_core  <= 1'b0;
            ddr_rsp_toggle_core_sync1 <= 1'b0;
            ddr_rsp_toggle_core_sync2 <= 1'b0;
            ddr_rsp_toggle_core_seen  <= 1'b0;
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
            sd_sector_ddr_base_word_bank[0] <= 32'd0;
            sd_sector_ddr_base_word_bank[1] <= 32'd0;
            sd_sector_words_queued_bank[0] <= 8'd0;
            sd_sector_words_queued_bank[1] <= 8'd0;
            sd_sector_buf_ready <= 2'b00;
            sd_fill_bank <= 1'b0;
            sd_flush_bank <= 1'b0;
            sd_ddr_flush_active <= 1'b0;
            sd_ddr_flush_idx <= 8'd0;
            sd_wait_counter    <= 24'd0;
            sd_header_done     <= 1'b0;
            sd_file_total_bytes<= 32'd0;
            sd_file_bytes_seen <= 32'd0;
            sd_copy_done_pending <= 1'b0;
            sd_use_sector_limit <= 1'b0;
            sd_copy_raw1_mode <= 1'b1;
            sd_copy_dest_base_word <= 32'd0;
            raw_image0_valid    <= 1'b0;
            raw_num_images      <= 32'd0;
            raw_bytes_per_image <= 32'd0;
            raw_image0_capture_idx <= 10'd0;
            raw_image0_sum_u8   <= 32'd0;
            raw_image0_rd_addr  <= 10'd0;
            imgload_active      <= 1'b0;
            imgload_addr_word    <= 32'd0;
            imgload_lane        <= 2'd0;
            imgload_byte_idx     <= 10'd0;
            imgload_total_bytes  <= 10'd0;
            imgload_sum_u8_accum <= 32'd0;
            imgload_word_valid   <= 1'b0;
            imgload_word_data    <= 32'd0;
            imgload_word_lane    <= 2'd0;
            train_trace_active   <= 1'b0;
            train_trace_state    <= TRK_IDLE;
            train_winner_idx     <= 7'd0;
            train_pre_count      <= 10'd0;
            train_a_idx          <= 10'd0;
            train_pre_idx        <= 10'd0;
            train_b_col_idx      <= 7'd0;
            train_curr_pre       <= 10'd0;
            train_tmp_x_val      <= 32'd0;
            train_tmp_mem_val    <= 32'd0;
            train_trace_a_row_base <= TRAIN_BASE_A_Q16_WORDS;
            train_trace_bt_pre_base <= TRAIN_BASE_BT_Q16_WORDS;
            train_stdp_active    <= 1'b0;
            train_stdp_state     <= TSK_IDLE;
            train_stdp_row0      <= 7'd0;
            train_stdp_row_end   <= 7'd0;
            train_stdp_row_idx   <= 7'd0;
            train_stdp_col_idx   <= 10'd0;
            train_stdp_w_val     <= 32'sd0;
            train_stdp_a_val     <= 32'sd0;
            train_stdp_bt_val    <= 32'sd0;
            train_stdp_w_new     <= 32'sd0;
            train_stdp_w_norm_q16<= 32'sd0;
            train_stdp_dW_q16    <= 32'sd0;
            train_stdp_row_sum_abs <= 32'd0;
            train_stdp_w_row_base <= TRAIN_BASE_W_Q16_WORDS;
            train_stdp_a_row_base <= TRAIN_BASE_A_Q16_WORDS;
            train_stdp_bt_col_base <= TRAIN_BASE_BT_Q16_WORDS;
            train_stdp_dividend  <= 32'd0;
            train_stdp_divisor   <= 32'd1;
            train_stdp_div_valid <= 1'b0;
            train_stdp_batch_active <= 1'b0;
            train_stdp_batch_tile_rows <= 7'd0;
            train_stdp_batch_next_row0 <= 7'd0;
            train_chunk_active   <= 1'b0;
            train_chunk_state    <= TCK_IDLE;
            train_chunk_mode     <= 2'd0;
            train_chunk_samples_left <= 16'd0;
            train_chunk_tile_rows <= 7'd0;
            train_chunk_steps_left <= 16'd0;
            train_chunk_seed_xin <= 32'h13579BDF;
            train_chunk_seed_xexc <= 32'h2468ACE1;
            train_chunk_winner <= 7'd0;
            train_chunk_pre_idx <= 10'd0;
            train_chunk_pre_from_infer <= 10'd0;
            train_chunk_pre_scan_idx <= 10'd0;
            train_chunk_pre_write_count <= 10'd0;
            train_chunk_winner_scan_idx <= 7'd0;
            train_chunk_winner_best_idx <= 7'd0;
            train_chunk_winner_best_count <= 16'd0;
            train_chunk_last_infer_spikes <= 32'd0;
            train_chunk_last_blank_spikes <= 32'd0;
            train_chunk_retry_curr_max_fr <= TRAIN_RETRY_MAX_FR_START;
            train_chunk_retry_accepted_max_fr <= TRAIN_RETRY_MAX_FR_START;
            train_gen_active     <= 1'b0;
            train_gen_state      <= TGK_IDLE;
            train_gen_base_word  <= 32'd0;
            train_gen_count_total<= 16'd0;
            train_gen_idx        <= 16'd0;
            train_gen_lcg_state  <= 32'd0;
            train_gen_curr_word  <= 32'd0;
            train_gen_lcg_enable <= 1'b0;
            train_gen_cache_mode <= 2'd0;
            train_label_stats_active <= 1'b0;
            train_label_stats_state <= TLS_IDLE;
            train_label_stats_label <= 4'd0;
            train_label_stats_idx <= 10'd0;
            train_label_stats_base_idx <= 10'd0;
            train_xin_cache_valid <= 1'b0;
            train_xexc_cache_valid <= 1'b0;
            infer_active        <= 1'b0;
            infer_state         <= INFER_IDLE;
            infer_steps_target  <= 32'd0;
            infer_step_idx      <= 16'd0;
            infer_neuron_idx    <= 7'd0;
            infer_input_idx     <= 10'd0;
            infer_prep_idx      <= 10'd0;
            infer_accum         <= 32'sd0;
            infer_accum_weight_phase <= 3'd0;
            infer_w_rd_addr     <= 17'd0;
            infer_pre_rd_addr   <= 10'd0;
            infer_apply_idx     <= 7'd0;
            infer_sum_c_inh     <= 32'sd0;
            infer_total_spikes  <= 32'd0;
            infer_rng_state     <= 32'd0;
            infer_poisson_num_const_cfg <= POISSON_NUM_CONST;
            infer_dividend      <= 32'd0;
            infer_divisor       <= 32'd1;
            infer_div_valid     <= 1'b0;
            infer_skip_init_clear <= 1'b0;
            infer_force_no_input  <= 1'b0;
            infer_spike_count_rd_addr <= 7'd0;
            train_label_sum_rd_addr <= 10'd0;
            train_label_count_rd_addr <= 4'd0;
            infer_poisson_thresh_rd_addr <= 10'd0;
            infer_pre_active_count <= 10'd0;
            infer_last_active_input_idx <= 10'd0;
            for (int i = 0; i < N_NEURONS; i++) begin
                train_inj_spike_count_snap[i] <= 16'd0;
            end
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
            train_stdp_div_valid <= 1'b0;
            ddr_rsp_toggle_core_sync1 <= ddr_rsp_toggle_ddr;
            ddr_rsp_toggle_core_sync2 <= ddr_rsp_toggle_core_sync1;

            if (ddr_req_pending_core && !response_ready &&
                (ddr_rsp_toggle_core_sync2 != ddr_rsp_toggle_core_seen)) begin
                ddr_rsp_toggle_core_seen <= ddr_rsp_toggle_core_sync2;
                ddr_req_pending_core <= 1'b0;
                if (ddr_req_from_sd_core) begin
                    ddr_req_from_sd_core <= 1'b0;
                    if (ddr_resp_status_async != STATUS_OK) begin
                        sd_ddr_flush_active <= 1'b0;
                        sd_copy_active <= 1'b0;
                        sd_in_read <= 1'b0;
                        resp_status    <= STATUS_BAD_PACKET;
                        resp_result    <= 32'sd0;
                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                        response_ready <= 1'b1;
                    end else begin
                        ddr_write_count <= ddr_write_count + {29'd0, ddr_req_word_count_core};
                        if ((sd_ddr_flush_idx + {5'd0, ddr_req_word_count_core}) >= sd_sector_words_queued_bank[sd_flush_bank]) begin
                            sd_ddr_flush_active <= 1'b0;
                            sd_ddr_flush_idx <= 8'd0;
                            sd_sector_buf_ready[sd_flush_bank] <= 1'b0;
                            if (sd_copy_done_pending && !sd_in_read &&
                                ((sd_flush_bank == 1'b0 && !sd_sector_buf_ready[1]) ||
                                 (sd_flush_bank == 1'b1 && !sd_sector_buf_ready[0]))) begin
                                sd_copy_active <= 1'b0;
                                resp_status    <= STATUS_OK;
                                resp_result    <= sd_copy_words_written;
                                resp_checksum  <= calc_resp_checksum(STATUS_OK, sd_copy_words_written);
                                response_ready <= 1'b1;
                            end
                        end else begin
                            sd_ddr_flush_idx <= sd_ddr_flush_idx + {5'd0, ddr_req_word_count_core};
                        end
                    end
                end else if (ddr_req_from_imgload_core) begin
                    ddr_req_from_imgload_core <= 1'b0;
                    if (ddr_resp_status_async != STATUS_OK) begin
                        imgload_active <= 1'b0;
                        imgload_word_valid <= 1'b0;
                        resp_status    <= STATUS_BAD_PACKET;
                        resp_result    <= 32'sd0;
                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                        response_ready <= 1'b1;
                    end else begin
                        imgload_word_valid <= 1'b1;
                        imgload_word_data  <= ddr_resp_rdata_async;
                        imgload_word_lane  <= imgload_lane;
                        imgload_addr_word <= imgload_addr_word + 32'd1;
                        imgload_lane <= 2'd0;
                    end
                end else if (TRAIN_ENABLE && ddr_req_from_train_core) begin
                    ddr_req_from_train_core <= 1'b0;
                    if (ddr_resp_status_async != STATUS_OK) begin
                        train_trace_active <= 1'b0;
                        train_trace_state <= TRK_IDLE;
                        train_stdp_active <= 1'b0;
                        train_stdp_state  <= TSK_IDLE;
                        train_stdp_batch_active <= 1'b0;
                        train_chunk_active <= 1'b0;
                        train_chunk_state <= TCK_IDLE;
                        train_gen_active  <= 1'b0;
                        train_gen_state   <= TGK_IDLE;
                        resp_status    <= STATUS_BAD_PACKET;
                        resp_result    <= 32'sd0;
                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                        response_ready <= 1'b1;
                    end else if (train_trace_active) begin
                        case (train_trace_state)
                            TRK_A_READ_X_WAIT: begin
                                train_tmp_x_val <= ddr_resp_rdata_async;
                                train_trace_state <= TRK_A_READ_A_REQ;
                            end
                            TRK_A_READ_A_WAIT: begin
                                train_tmp_mem_val <= ddr_resp_rdata_async;
                                train_trace_state <= TRK_A_WRITE_A_REQ;
                            end
                            TRK_A_WRITE_A_WAIT: begin
                                if (train_a_idx == (N_IN - 1)) begin
                                    train_pre_idx <= 10'd0;
                                    train_b_col_idx <= 7'd0;
                                    train_trace_state <= TRK_B_READ_PRE_REQ;
                                end else begin
                                    train_a_idx <= train_a_idx + 10'd1;
                                    train_trace_state <= TRK_A_READ_X_REQ;
                                end
                            end
                            TRK_B_READ_PRE_WAIT: begin
                                train_curr_pre <= ddr_resp_rdata_async[9:0];
                                train_trace_bt_pre_base <= TRAIN_BASE_BT_Q16_WORDS + ({22'd0, ddr_resp_rdata_async[9:0]} * N_NEURONS);
                                train_b_col_idx <= 7'd0;
                                train_trace_state <= TRK_B_READ_X_REQ;
                            end
                            TRK_B_READ_X_WAIT: begin
                                train_tmp_x_val <= ddr_resp_rdata_async;
                                train_trace_state <= TRK_B_READ_BT_REQ;
                            end
                            TRK_B_READ_BT_WAIT: begin
                                train_tmp_mem_val <= ddr_resp_rdata_async;
                                train_trace_state <= TRK_B_WRITE_BT_REQ;
                            end
                            TRK_B_WRITE_BT_WAIT: begin
                                if (train_b_col_idx == (N_NEURONS - 1)) begin
                                    if ((train_pre_idx + 10'd1) >= train_pre_count) begin
                                        train_trace_state <= TRK_DONE;
                                    end else begin
                                        train_pre_idx <= train_pre_idx + 10'd1;
                                        train_trace_state <= TRK_B_READ_PRE_REQ;
                                    end
                                end else begin
                                    train_b_col_idx <= train_b_col_idx + 7'd1;
                                    train_trace_state <= TRK_B_READ_X_REQ;
                                end
                            end
                            default: begin
                                train_trace_active <= 1'b0;
                                train_trace_state <= TRK_IDLE;
                                resp_status    <= STATUS_BAD_PACKET;
                                resp_result    <= 32'sd0;
                                resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                response_ready <= 1'b1;
                            end
                        endcase
                    end else if (train_stdp_active) begin
                        case (train_stdp_state)
                            TSK_SUM_READ_W_WAIT: begin
                                if (train_stdp_col_idx == (N_IN - 1)) begin
                                    train_stdp_row_sum_abs <= train_stdp_row_sum_abs + s32_abs_u($signed(ddr_resp_rdata_async));
                                    train_stdp_col_idx <= 10'd0;
                                    train_stdp_state <= TSK_READ_W_REQ;
                                end else begin
                                    train_stdp_row_sum_abs <= train_stdp_row_sum_abs + s32_abs_u($signed(ddr_resp_rdata_async));
                                    train_stdp_col_idx <= train_stdp_col_idx + 10'd1;
                                    train_stdp_state <= TSK_SUM_READ_W_REQ;
                                end
                            end
                            TSK_READ_W_WAIT: begin
                                train_stdp_w_val <= $signed(ddr_resp_rdata_async);
                                train_stdp_state <= TSK_READ_A_REQ;
                            end
                            TSK_READ_A_WAIT: begin
                                train_stdp_a_val <= $signed(ddr_resp_rdata_async);
                                train_stdp_state <= TSK_READ_BT_REQ;
                            end
                            TSK_READ_BT_WAIT: begin
                                train_stdp_bt_val <= $signed(ddr_resp_rdata_async);
                                train_stdp_state <= TSK_DIV_NORM_START;
                            end
                            TSK_WRITE_W_WAIT: begin
                                if (train_stdp_col_idx == (N_IN - 1)) begin
                                    if ((train_stdp_row_idx + 7'd1) >= train_stdp_row_end) begin
                                        train_stdp_state <= TSK_DONE;
                                    end else begin
                                        train_stdp_row_idx <= train_stdp_row_idx + 7'd1;
                                        train_stdp_col_idx <= 10'd0;
                                        train_stdp_row_sum_abs <= 32'd0;
                                        train_stdp_w_row_base <= train_stdp_w_row_base + N_IN;
                                        train_stdp_a_row_base <= train_stdp_a_row_base + N_IN;
                                        train_stdp_bt_col_base <= TRAIN_BASE_BT_Q16_WORDS + {24'd0, (train_stdp_row_idx + 7'd1)};
                                        train_stdp_state   <= TSK_SUM_READ_W_REQ;
                                    end
                                end else begin
                                    train_stdp_col_idx <= train_stdp_col_idx + 10'd1;
                                    train_stdp_bt_col_base <= train_stdp_bt_col_base + N_NEURONS;
                                    train_stdp_state   <= TSK_READ_W_REQ;
                                end
                            end
                            default: begin
                                train_stdp_active <= 1'b0;
                                train_stdp_state  <= TSK_IDLE;
                                train_stdp_batch_active <= 1'b0;
                                train_chunk_active <= 1'b0;
                                train_chunk_state <= TCK_IDLE;
                                resp_status    <= STATUS_BAD_PACKET;
                                resp_result    <= 32'sd0;
                                resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                response_ready <= 1'b1;
                            end
                        endcase
                    end else if (train_gen_active) begin
                        case (train_gen_state)
                            TGK_WRITE_WAIT: begin
                                if ((train_gen_idx + 16'd1) >= train_gen_count_total) begin
                                    train_gen_state <= TGK_DONE;
                                end else begin
                                    train_gen_idx <= train_gen_idx + 16'd1;
                                    if (train_gen_lcg_enable) begin
                                        train_gen_lcg_state <= ($unsigned(train_gen_lcg_state) * LCG_A) + LCG_C;
                                        train_gen_curr_word <= train_gen_word_from_state(
                                            ($unsigned(train_gen_lcg_state) * LCG_A) + LCG_C
                                        );
                                    end
                                    train_gen_state <= TGK_WRITE_REQ;
                                end
                            end
                            default: begin
                                train_gen_active <= 1'b0;
                                train_gen_state  <= TGK_IDLE;
                                resp_status    <= STATUS_BAD_PACKET;
                                resp_result    <= 32'sd0;
                                resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                response_ready <= 1'b1;
                            end
                        endcase
                    end else begin
                        resp_status    <= STATUS_BAD_PACKET;
                        resp_result    <= 32'sd0;
                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                        response_ready <= 1'b1;
                    end
                end else begin
                    // Generic DDR read/write response path (non-SD, non-imgload, non-train-kernel).
                    // During TRAIN_RUN_CHUNK phase1 prelist staging, a single generic DDR write is issued
                    // intentionally; consume its ACK silently so only the coarse chunk completion response
                    // is visible to the host.
                    if (TRAIN_ENABLE && train_chunk_active && (train_chunk_state == TCK_PRELIST_WRITE_WAIT)) begin
                        if ((ddr_resp_status_async == STATUS_OK) && ddr_req_we_core) begin
                            ddr_write_count <= ddr_write_count + 32'd1;
                        end
                    end else begin
                        if ((ddr_resp_status_async == STATUS_OK) && ddr_req_we_core) begin
                            ddr_write_count <= ddr_write_count + 32'd1;
                        end
                        resp_status    <= ddr_resp_status_async;
                        resp_result    <= ddr_resp_rdata_async;
                        resp_checksum  <= calc_resp_checksum(ddr_resp_status_async, ddr_resp_rdata_async);
                        response_ready <= 1'b1;
                    end
                end
            end
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
                        MEMRD_TRAIN_INJ_SPIKE_COUNT: begin
                            resp_status    <= STATUS_OK;
                            resp_result    <= {16'd0, train_inj_spike_count_snap_rd_data};
                            resp_checksum  <= calc_resp_checksum(STATUS_OK, {16'd0, train_inj_spike_count_snap_rd_data});
                            response_ready <= 1'b1;
                        end
                        MEMRD_TRAIN_LABEL_STAT_SUM: begin
                            resp_status    <= STATUS_OK;
                            resp_result    <= train_label_sum_rd_data;
                            resp_checksum  <= calc_resp_checksum(STATUS_OK, train_label_sum_rd_data);
                            response_ready <= 1'b1;
                        end
                        MEMRD_TRAIN_LABEL_STAT_COUNT: begin
                            resp_status    <= STATUS_OK;
                            resp_result    <= train_label_count_rd_data;
                            resp_checksum  <= calc_resp_checksum(STATUS_OK, train_label_count_rd_data);
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

            if (sd_ddr_flush_active && !ddr_req_pending_core && !response_ready && sd_copy_active) begin
                if (sd_ddr_flush_idx < sd_sector_words_queued_bank[sd_flush_bank]) begin
                    logic [7:0] words_left;
                    logic [2:0] words_this_req;
                    logic [15:0] sel_mask16;
                    logic [127:0] wide_wdata;
                    words_left = sd_sector_words_queued_bank[sd_flush_bank] - sd_ddr_flush_idx;
                    words_this_req = (words_left >= 8'd4) ? 3'd4 : words_left[2:0];
                    wide_wdata = 128'd0;
                    sel_mask16 = 16'd0;
                    if (words_this_req >= 3'd1) begin
                        wide_wdata[31:0] = sd_sector_word_buf[sd_flush_bank][sd_ddr_flush_idx];
                        sel_mask16[3:0] = 4'hF;
                    end
                    if (words_this_req >= 3'd2) begin
                        wide_wdata[63:32] = sd_sector_word_buf[sd_flush_bank][sd_ddr_flush_idx + 8'd1];
                        sel_mask16[7:4] = 4'hF;
                    end
                    if (words_this_req >= 3'd3) begin
                        wide_wdata[95:64] = sd_sector_word_buf[sd_flush_bank][sd_ddr_flush_idx + 8'd2];
                        sel_mask16[11:8] = 4'hF;
                    end
                    if (words_this_req >= 3'd4) begin
                        wide_wdata[127:96] = sd_sector_word_buf[sd_flush_bank][sd_ddr_flush_idx + 8'd3];
                        sel_mask16[15:12] = 4'hF;
                    end

                    ddr_req_pending_core   <= 1'b1;
                    ddr_req_we_core        <= 1'b1;
                    ddr_req_from_sd_core   <= 1'b1;
                    ddr_req_from_imgload_core <= 1'b0;
                    ddr_req_from_train_core <= 1'b0;
                    ddr_req_addr_word_core <= sd_sector_ddr_base_word_bank[sd_flush_bank] + {24'd0, sd_ddr_flush_idx};
                    ddr_req_wdata_core     <= sd_sector_word_buf[sd_flush_bank][sd_ddr_flush_idx];
                    ddr_req_wide_core      <= 1'b1;
                    ddr_req_wdata128_core  <= wide_wdata;
                    ddr_req_sel16_core     <= sel_mask16;
                    ddr_req_word_count_core <= words_this_req;
                    ddr_req_toggle_core    <= ~ddr_req_toggle_core;
                    ddr_last_addr          <= sd_sector_ddr_base_word_bank[sd_flush_bank] + {24'd0, sd_ddr_flush_idx};
                    ddr_last_data          <= sd_sector_word_buf[sd_flush_bank][sd_ddr_flush_idx];
                end else begin
                    sd_ddr_flush_active <= 1'b0;
                end
            end

            if (imgload_active && imgload_word_valid && !response_ready) begin
                if (imgload_byte_idx < imgload_total_bytes) begin
                    raw_image0_u8[imgload_byte_idx] <= lane_byte_sel(imgload_word_data, imgload_word_lane);
                    imgload_sum_u8_accum <= imgload_sum_u8_accum + {24'd0, lane_byte_sel(imgload_word_data, imgload_word_lane)};
                    if ((imgload_byte_idx + 10'd1) >= imgload_total_bytes) begin
                        imgload_active <= 1'b0;
                        imgload_word_valid <= 1'b0;
                        imgload_byte_idx <= imgload_byte_idx + 10'd1;
                        raw_image0_valid <= 1'b1;
                        raw_image0_capture_idx <= imgload_byte_idx + 10'd1;
                        raw_image0_sum_u8 <= imgload_sum_u8_accum + {24'd0, lane_byte_sel(imgload_word_data, imgload_word_lane)};
                        resp_status    <= STATUS_OK;
                        resp_result    <= imgload_byte_idx + 10'd1;
                        resp_checksum  <= calc_resp_checksum(STATUS_OK, imgload_byte_idx + 10'd1);
                        response_ready <= 1'b1;
                    end else begin
                        imgload_byte_idx <= imgload_byte_idx + 10'd1;
                        if (imgload_word_lane == 2'd3) begin
                            imgload_word_valid <= 1'b0;
                            imgload_word_lane <= 2'd0;
                        end else begin
                            imgload_word_lane <= imgload_word_lane + 2'd1;
                        end
                    end
                end else begin
                    imgload_word_valid <= 1'b0;
                end
            end

            if (imgload_active && !imgload_word_valid && !ddr_req_pending_core && !response_ready) begin
                if (imgload_byte_idx < imgload_total_bytes) begin
                    ddr_req_pending_core    <= 1'b1;
                    ddr_req_we_core         <= 1'b0;
                    ddr_req_from_sd_core    <= 1'b0;
                    ddr_req_from_imgload_core <= 1'b1;
                    ddr_req_from_train_core <= 1'b0;
                    ddr_req_addr_word_core  <= imgload_addr_word;
                    ddr_req_wdata_core      <= 32'd0;
                    ddr_req_wide_core       <= 1'b0;
                    ddr_req_wdata128_core   <= 128'd0;
                    ddr_req_sel16_core      <= 16'd0;
                    ddr_req_word_count_core <= 3'd1;
                    ddr_req_toggle_core     <= ~ddr_req_toggle_core;
                    ddr_last_addr           <= imgload_addr_word;
                end
            end

            if (TRAIN_ENABLE && train_trace_active && !ddr_req_pending_core && !response_ready && !sd_ddr_flush_active && !imgload_word_valid && !train_stdp_active) begin
                case (train_trace_state)
                    TRK_A_READ_X_REQ: begin
                        if (train_xin_cache_valid) begin
                            train_tmp_x_val   <= train_xin_cache[train_a_idx];
                            train_trace_state <= TRK_A_READ_A_REQ;
                        end else begin
                            ddr_req_pending_core    <= 1'b1;
                            ddr_req_we_core         <= 1'b0;
                            ddr_req_from_sd_core    <= 1'b0;
                            ddr_req_from_imgload_core <= 1'b0;
                            ddr_req_from_train_core <= 1'b1;
                            ddr_req_addr_word_core  <= TRAIN_BASE_XIN_WORK_WORDS + {22'd0, train_a_idx};
                            ddr_req_wdata_core      <= 32'd0;
                            ddr_req_wide_core       <= 1'b0;
                            ddr_req_wdata128_core   <= 128'd0;
                            ddr_req_sel16_core      <= 16'd0;
                            ddr_req_word_count_core <= 3'd1;
                            ddr_req_toggle_core     <= ~ddr_req_toggle_core;
                            train_trace_state       <= TRK_A_READ_X_WAIT;
                        end
                    end
                    TRK_A_READ_A_REQ: begin
                        ddr_req_pending_core    <= 1'b1;
                        ddr_req_we_core         <= 1'b0;
                        ddr_req_from_sd_core    <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core <= 1'b1;
                        ddr_req_addr_word_core  <= train_trace_a_row_base + {22'd0, train_a_idx};
                        ddr_req_wdata_core      <= 32'd0;
                        ddr_req_wide_core       <= 1'b0;
                        ddr_req_wdata128_core   <= 128'd0;
                        ddr_req_sel16_core      <= 16'd0;
                        ddr_req_word_count_core <= 3'd1;
                        ddr_req_toggle_core     <= ~ddr_req_toggle_core;
                        train_trace_state       <= TRK_A_READ_A_WAIT;
                    end
                    TRK_A_WRITE_A_REQ: begin
                        ddr_req_pending_core    <= 1'b1;
                        ddr_req_we_core         <= 1'b1;
                        ddr_req_from_sd_core    <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core <= 1'b1;
                        ddr_req_addr_word_core  <= train_trace_a_row_base + {22'd0, train_a_idx};
                        ddr_req_wdata_core      <= train_tmp_mem_val + train_tmp_x_val;
                        ddr_req_wide_core       <= 1'b0;
                        ddr_req_wdata128_core   <= 128'd0;
                        ddr_req_sel16_core      <= 16'd0;
                        ddr_req_word_count_core <= 3'd1;
                        ddr_req_toggle_core     <= ~ddr_req_toggle_core;
                        train_trace_state       <= TRK_A_WRITE_A_WAIT;
                    end
                    TRK_B_READ_PRE_REQ: begin
                        if (train_pre_idx >= train_pre_count) begin
                            train_trace_state <= TRK_DONE;
                        end else begin
                            ddr_req_pending_core    <= 1'b1;
                            ddr_req_we_core         <= 1'b0;
                            ddr_req_from_sd_core    <= 1'b0;
                            ddr_req_from_imgload_core <= 1'b0;
                            ddr_req_from_train_core <= 1'b1;
                            ddr_req_addr_word_core  <= TRAIN_BASE_PRELIST_WORK_WORDS + {22'd0, train_pre_idx};
                            ddr_req_wdata_core      <= 32'd0;
                            ddr_req_wide_core       <= 1'b0;
                            ddr_req_wdata128_core   <= 128'd0;
                            ddr_req_sel16_core      <= 16'd0;
                            ddr_req_word_count_core <= 3'd1;
                            ddr_req_toggle_core     <= ~ddr_req_toggle_core;
                            train_trace_state       <= TRK_B_READ_PRE_WAIT;
                        end
                    end
                    TRK_B_READ_X_REQ: begin
                        if (train_xexc_cache_valid) begin
                            train_tmp_x_val   <= train_xexc_cache[train_b_col_idx];
                            train_trace_state <= TRK_B_READ_BT_REQ;
                        end else begin
                            ddr_req_pending_core    <= 1'b1;
                            ddr_req_we_core         <= 1'b0;
                            ddr_req_from_sd_core    <= 1'b0;
                            ddr_req_from_imgload_core <= 1'b0;
                            ddr_req_from_train_core <= 1'b1;
                            ddr_req_addr_word_core  <= TRAIN_BASE_XEXC_WORK_WORDS + {25'd0, train_b_col_idx};
                            ddr_req_wdata_core      <= 32'd0;
                            ddr_req_wide_core       <= 1'b0;
                            ddr_req_wdata128_core   <= 128'd0;
                            ddr_req_sel16_core      <= 16'd0;
                            ddr_req_word_count_core <= 3'd1;
                            ddr_req_toggle_core     <= ~ddr_req_toggle_core;
                            train_trace_state       <= TRK_B_READ_X_WAIT;
                        end
                    end
                    TRK_B_READ_BT_REQ: begin
                        ddr_req_pending_core    <= 1'b1;
                        ddr_req_we_core         <= 1'b0;
                        ddr_req_from_sd_core    <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core <= 1'b1;
                        ddr_req_addr_word_core  <= train_trace_bt_pre_base + {25'd0, train_b_col_idx};
                        ddr_req_wdata_core      <= 32'd0;
                        ddr_req_wide_core       <= 1'b0;
                        ddr_req_wdata128_core   <= 128'd0;
                        ddr_req_sel16_core      <= 16'd0;
                        ddr_req_word_count_core <= 3'd1;
                        ddr_req_toggle_core     <= ~ddr_req_toggle_core;
                        train_trace_state       <= TRK_B_READ_BT_WAIT;
                    end
                    TRK_B_WRITE_BT_REQ: begin
                        ddr_req_pending_core    <= 1'b1;
                        ddr_req_we_core         <= 1'b1;
                        ddr_req_from_sd_core    <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core <= 1'b1;
                        ddr_req_addr_word_core  <= train_trace_bt_pre_base + {25'd0, train_b_col_idx};
                        ddr_req_wdata_core      <= train_tmp_mem_val + train_tmp_x_val;
                        ddr_req_wide_core       <= 1'b0;
                        ddr_req_wdata128_core   <= 128'd0;
                        ddr_req_sel16_core      <= 16'd0;
                        ddr_req_word_count_core <= 3'd1;
                        ddr_req_toggle_core     <= ~ddr_req_toggle_core;
                        train_trace_state       <= TRK_B_WRITE_BT_WAIT;
                    end
                    TRK_DONE: begin
                        train_trace_active <= 1'b0;
                        train_trace_state <= TRK_IDLE;
                        if (!train_chunk_active) begin
                            resp_status    <= STATUS_OK;
                            resp_result    <= {22'd0, train_pre_count};
                            resp_checksum  <= calc_resp_checksum(STATUS_OK, {22'd0, train_pre_count});
                            response_ready <= 1'b1;
                        end
                    end
                    default: begin end
                endcase
            end

            if (TRAIN_ENABLE && train_stdp_active && !ddr_req_pending_core && !response_ready && !sd_ddr_flush_active && !imgload_word_valid && !train_trace_active) begin
                case (train_stdp_state)
                    TSK_SUM_READ_W_REQ: begin
                        ddr_req_pending_core      <= 1'b1;
                        ddr_req_we_core           <= 1'b0;
                        ddr_req_from_sd_core      <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core   <= 1'b1;
                        ddr_req_addr_word_core    <= train_stdp_w_row_base + {22'd0, train_stdp_col_idx};
                        ddr_req_wdata_core        <= 32'd0;
                        ddr_req_wide_core         <= 1'b0;
                        ddr_req_wdata128_core     <= 128'd0;
                        ddr_req_sel16_core        <= 16'd0;
                        ddr_req_word_count_core   <= 3'd1;
                        ddr_req_toggle_core       <= ~ddr_req_toggle_core;
                        train_stdp_state          <= TSK_SUM_READ_W_WAIT;
                    end
                    TSK_READ_W_REQ: begin
                        ddr_req_pending_core      <= 1'b1;
                        ddr_req_we_core           <= 1'b0;
                        ddr_req_from_sd_core      <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core   <= 1'b1;
                        ddr_req_addr_word_core    <= train_stdp_w_row_base + {22'd0, train_stdp_col_idx};
                        ddr_req_wdata_core        <= 32'd0;
                        ddr_req_wide_core         <= 1'b0;
                        ddr_req_wdata128_core     <= 128'd0;
                        ddr_req_sel16_core        <= 16'd0;
                        ddr_req_word_count_core   <= 3'd1;
                        ddr_req_toggle_core       <= ~ddr_req_toggle_core;
                        train_stdp_state          <= TSK_READ_W_WAIT;
                    end
                    TSK_READ_A_REQ: begin
                        ddr_req_pending_core      <= 1'b1;
                        ddr_req_we_core           <= 1'b0;
                        ddr_req_from_sd_core      <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core   <= 1'b1;
                        ddr_req_addr_word_core    <= train_stdp_a_row_base + {22'd0, train_stdp_col_idx};
                        ddr_req_wdata_core        <= 32'd0;
                        ddr_req_wide_core         <= 1'b0;
                        ddr_req_wdata128_core     <= 128'd0;
                        ddr_req_sel16_core        <= 16'd0;
                        ddr_req_word_count_core   <= 3'd1;
                        ddr_req_toggle_core       <= ~ddr_req_toggle_core;
                        train_stdp_state          <= TSK_READ_A_WAIT;
                    end
                    TSK_READ_BT_REQ: begin
                        ddr_req_pending_core      <= 1'b1;
                        ddr_req_we_core           <= 1'b0;
                        ddr_req_from_sd_core      <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core   <= 1'b1;
                        ddr_req_addr_word_core    <= train_stdp_bt_col_base;
                        ddr_req_wdata_core        <= 32'd0;
                        ddr_req_wide_core         <= 1'b0;
                        ddr_req_wdata128_core     <= 128'd0;
                        ddr_req_sel16_core        <= 16'd0;
                        ddr_req_word_count_core   <= 3'd1;
                        ddr_req_toggle_core       <= ~ddr_req_toggle_core;
                        train_stdp_state          <= TSK_READ_BT_WAIT;
                    end
                    TSK_DIV_NORM_START: begin
                        logic [31:0] denom_q16_tmp;
                        logic signed [63:0] norm_num_tmp;
                        denom_q16_tmp = (train_stdp_row_sum_abs == 32'd0) ? 32'd1 : train_stdp_row_sum_abs;
                        norm_num_tmp = $signed(train_stdp_w_val) * $signed(TRAIN_NORM_Q16);
                        if (!train_stdp_div_busy) begin
                            train_stdp_dividend <= norm_num_tmp[31:0];
                            train_stdp_divisor  <= denom_q16_tmp;
                            train_stdp_div_valid <= 1'b1;
                            train_stdp_state <= TSK_DIV_NORM_WAIT;
                        end
                    end
                    TSK_DIV_NORM_WAIT: begin
                        if (train_stdp_div_out_valid) begin
                            if (train_stdp_div_err) begin
                                train_stdp_active <= 1'b0;
                                train_stdp_state  <= TSK_IDLE;
                                train_stdp_batch_active <= 1'b0;
                                train_chunk_active <= 1'b0;
                                train_chunk_state <= TCK_IDLE;
                                resp_status    <= STATUS_BAD_PACKET;
                                resp_result    <= 32'sd0;
                                resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                response_ready <= 1'b1;
                            end else begin
                                train_stdp_w_norm_q16 <= $signed(train_stdp_div_q);
                                train_stdp_state <= TSK_DIV_DW_START;
                            end
                        end
                    end
                    TSK_DIV_DW_START: begin
                        logic signed [31:0] pot_term_tmp;
                        logic signed [31:0] dep_term_tmp;
                        logic signed [31:0] dW_q16_tmp;
                        logic [31:0] dW_abs_tmp;
                        // Event-shortcut: if both traces are zero, dW is exactly zero.
                        // Skip the divider path and directly commit normalized weight.
                        if ((train_stdp_a_val == 32'sd0) && (train_stdp_bt_val == 32'sd0)) begin
                            train_stdp_dW_q16 <= 32'sd0;
                            train_stdp_w_new <= train_stdp_w_norm_q16;
                            train_stdp_state <= TSK_WRITE_W_REQ;
                        end else begin
                        pot_term_tmp = fxp_mul_s16_16(
                            fxp_mul_s16_16(TRAIN_LR_P_Q16, (TRAIN_WMAX_Q16 - train_stdp_w_norm_q16)),
                            train_stdp_a_val
                        );
                        dep_term_tmp = fxp_mul_s16_16(
                            fxp_mul_s16_16(TRAIN_LR_M_Q16, train_stdp_w_norm_q16),
                            train_stdp_bt_val
                        );
                        dW_q16_tmp = pot_term_tmp - dep_term_tmp;
                        if (dW_q16_tmp == 32'sd0) begin
                            train_stdp_dW_q16 <= 32'sd0;
                            train_stdp_w_new <= train_stdp_w_norm_q16;
                            train_stdp_state <= TSK_WRITE_W_REQ;
                        end else begin
                        if (dW_q16_tmp < 0)
                            dW_abs_tmp = $unsigned(-dW_q16_tmp);
                        else
                            dW_abs_tmp = $unsigned(dW_q16_tmp);
                        train_stdp_dW_q16 <= dW_q16_tmp;
                        if (!train_stdp_div_busy) begin
                            train_stdp_dividend <= dW_abs_tmp;
                            train_stdp_divisor  <= TRAIN_UPDATE_NT;
                            train_stdp_div_valid <= 1'b1;
                            train_stdp_state <= TSK_DIV_DW_WAIT;
                        end
                        end
                        end
                    end
                    TSK_DIV_DW_WAIT: begin
                        if (train_stdp_div_out_valid) begin
                            logic signed [31:0] dW_step_q16_tmp;
                            logic signed [31:0] w_next_q16_tmp;
                            if (train_stdp_div_err) begin
                                train_stdp_active <= 1'b0;
                                train_stdp_state  <= TSK_IDLE;
                                train_stdp_batch_active <= 1'b0;
                                train_chunk_active <= 1'b0;
                                train_chunk_state <= TCK_IDLE;
                                resp_status    <= STATUS_BAD_PACKET;
                                resp_result    <= 32'sd0;
                                resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                response_ready <= 1'b1;
                            end else begin
                                dW_step_q16_tmp = $signed(train_stdp_div_q);
                                if (train_stdp_dW_q16 < 0)
                                    dW_step_q16_tmp = -dW_step_q16_tmp;
                                if (dW_step_q16_tmp > TRAIN_CLIP_DW_Q16)
                                    dW_step_q16_tmp = TRAIN_CLIP_DW_Q16;
                                else if (dW_step_q16_tmp < -TRAIN_CLIP_DW_Q16)
                                    dW_step_q16_tmp = -TRAIN_CLIP_DW_Q16;
                                w_next_q16_tmp = train_stdp_w_norm_q16 + dW_step_q16_tmp;
                                if (w_next_q16_tmp > TRAIN_WMAX_Q16)
                                    w_next_q16_tmp = TRAIN_WMAX_Q16;
                                else if (w_next_q16_tmp < TRAIN_WMIN_Q16)
                                    w_next_q16_tmp = TRAIN_WMIN_Q16;
                                train_stdp_w_new <= w_next_q16_tmp;
                                train_stdp_state <= TSK_WRITE_W_REQ;
                            end
                        end
                    end
                    TSK_WRITE_W_REQ: begin
                        ddr_req_pending_core      <= 1'b1;
                        ddr_req_we_core           <= 1'b1;
                        ddr_req_from_sd_core      <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core   <= 1'b1;
                        ddr_req_addr_word_core    <= train_stdp_w_row_base + {22'd0, train_stdp_col_idx};
                        ddr_req_wdata_core        <= train_stdp_w_new;
                        ddr_req_wide_core         <= 1'b0;
                        ddr_req_wdata128_core     <= 128'd0;
                        ddr_req_sel16_core        <= 16'd0;
                        ddr_req_word_count_core   <= 3'd1;
                        ddr_req_toggle_core       <= ~ddr_req_toggle_core;
                        train_stdp_state          <= TSK_WRITE_W_WAIT;
                    end
                    TSK_DONE: begin
                        if (train_stdp_batch_active) begin
                            logic [7:0] next_row_tmp;
                            logic [7:0] next_end_tmp;
                            next_row_tmp = {1'b0, train_stdp_row_end};
                            if (next_row_tmp >= N_NEURONS[7:0]) begin
                                train_stdp_active       <= 1'b0;
                                train_stdp_state        <= TSK_IDLE;
                                train_stdp_batch_active <= 1'b0;
                                if (train_chunk_active) begin
                                    if (train_chunk_samples_left <= 16'd1) begin
                                        if ((train_chunk_mode == 3'd3) || (train_chunk_mode == 3'd4)) begin
                                            // Phase3 continues with a blank-period inference after STDP.
                                            train_chunk_samples_left <= 16'd0;
                                            train_chunk_state        <= TCK_BLANK_INFER_START;
                                        end else begin
                                            train_chunk_active       <= 1'b0;
                                            train_chunk_state        <= TCK_IDLE;
                                            train_chunk_mode         <= 2'd0;
                                            train_chunk_samples_left <= 16'd0;
                                            resp_status    <= STATUS_OK;
                                            if (train_chunk_mode == 3'd2) begin
                                                resp_result <= train_chunk_last_infer_spikes;
                                                resp_checksum <= calc_resp_checksum(STATUS_OK, train_chunk_last_infer_spikes);
                                            end else begin
                                                resp_result <= {16'd0, 16'd1}; // phase0/phase1 returns completed chunk count
                                                resp_checksum <= calc_resp_checksum(STATUS_OK, {16'd0, 16'd1});
                                            end
                                            response_ready <= 1'b1;
                                        end
                                    end else begin
                                        train_chunk_samples_left <= train_chunk_samples_left - 16'd1;
                                        train_stdp_batch_active    <= 1'b1;
                                        train_stdp_batch_tile_rows <= train_chunk_tile_rows;
                                        train_stdp_batch_next_row0 <= 7'd0;
                                        train_stdp_active          <= 1'b1;
                                        train_stdp_row0            <= 7'd0;
                                        if ({1'b0, train_chunk_tile_rows} >= N_NEURONS[7:0])
                                            train_stdp_row_end <= N_NEURONS[6:0];
                                        else
                                            train_stdp_row_end <= train_chunk_tile_rows;
                                        train_stdp_row_idx         <= 7'd0;
                                        train_stdp_col_idx         <= 10'd0;
                                        train_stdp_w_val           <= 32'sd0;
                                        train_stdp_a_val           <= 32'sd0;
                                        train_stdp_bt_val          <= 32'sd0;
                                        train_stdp_w_new           <= 32'sd0;
                                        train_stdp_row_sum_abs     <= 32'd0;
                                        train_stdp_w_row_base      <= TRAIN_BASE_W_Q16_WORDS;
                                        train_stdp_a_row_base      <= TRAIN_BASE_A_Q16_WORDS;
                                        train_stdp_bt_col_base     <= TRAIN_BASE_BT_Q16_WORDS;
                                        train_stdp_state           <= TSK_SUM_READ_W_REQ;
                                    end
                                end else begin
                                    resp_status    <= STATUS_OK;
                                    resp_result    <= N_NEURONS;
                                    resp_checksum  <= calc_resp_checksum(STATUS_OK, N_NEURONS);
                                    response_ready <= 1'b1;
                                end
                            end else begin
                                next_end_tmp = next_row_tmp + {1'b0, train_stdp_batch_tile_rows};
                                train_stdp_active      <= 1'b1;
                                train_stdp_state       <= TSK_SUM_READ_W_REQ;
                                train_stdp_row0        <= next_row_tmp[6:0];
                                train_stdp_row_idx     <= next_row_tmp[6:0];
                                train_stdp_col_idx     <= 10'd0;
                                train_stdp_row_sum_abs <= 32'd0;
                                train_stdp_w_row_base  <= TRAIN_BASE_W_Q16_WORDS + ({24'd0, next_row_tmp[6:0]} * N_IN);
                                train_stdp_a_row_base  <= TRAIN_BASE_A_Q16_WORDS + ({24'd0, next_row_tmp[6:0]} * N_IN);
                                train_stdp_bt_col_base <= TRAIN_BASE_BT_Q16_WORDS + {24'd0, next_row_tmp[6:0]};
                                if (next_end_tmp >= N_NEURONS[7:0])
                                    train_stdp_row_end <= N_NEURONS[6:0];
                                else
                                    train_stdp_row_end <= next_end_tmp[6:0];
                                train_stdp_batch_next_row0 <= next_row_tmp[6:0];
                            end
                        end else begin
                            train_stdp_active <= 1'b0;
                            train_stdp_state  <= TSK_IDLE;
                            resp_status    <= STATUS_OK;
                            resp_result    <= {25'd0, train_stdp_row_end - train_stdp_row0};
                            resp_checksum  <= calc_resp_checksum(STATUS_OK, {25'd0, train_stdp_row_end - train_stdp_row0});
                            response_ready <= 1'b1;
                        end
                    end
                    default: begin end
                endcase
            end

            if (TRAIN_ENABLE && train_chunk_active && !response_ready && !sd_ddr_flush_active && !imgload_word_valid &&
                !ddr_req_pending_core && !train_gen_active && !train_trace_active && !train_stdp_active) begin
                case (train_chunk_state)
                    TCK_INFER_START: begin
                        if (raw_image0_valid && (raw_bytes_per_image == 32'd784) && (raw_image0_sum_u8 != 32'd0)) begin
                            infer_active       <= 1'b1;
                            infer_state        <= INFER_INIT_CLEAR;
                            infer_steps_target <= {16'd0, train_chunk_steps_left};
                            infer_step_idx     <= 16'd0;
                            infer_neuron_idx   <= 7'd0;
                            infer_input_idx    <= 10'd0;
                            infer_prep_idx     <= 10'd0;
                            infer_accum        <= 32'sd0;
                            infer_accum_weight_phase <= 3'd0;
                            infer_apply_idx    <= 7'd0;
                            infer_sum_c_inh    <= 32'd0;
                            infer_total_spikes <= 32'd0;
                            infer_rng_state    <= 32'h12345678;
                            infer_pre_active_count <= 10'd0;
                            infer_skip_init_clear <= 1'b0;
                            infer_force_no_input  <= 1'b0;
                            train_chunk_state <= TCK_INFER_WAIT;
                        end else begin
                            resp_status       <= STATUS_BAD_PACKET;
                            resp_result       <= 32'h36E20001;
                            resp_checksum     <= calc_resp_checksum(STATUS_BAD_PACKET, 32'h36E20001);
                            response_ready    <= 1'b1;
                            train_chunk_active <= 1'b0;
                            train_chunk_mode  <= 2'd0;
                            train_chunk_state <= TCK_IDLE;
                        end
                    end
                    TCK_INFER_WAIT: begin
                        if (!infer_active) begin
                            train_chunk_last_infer_spikes <= infer_total_spikes;
                            for (int i = 0; i < N_NEURONS; i++) begin
                                train_inj_spike_count_snap[i] <= infer_spike_count[i];
                            end
                            if (response_ready && (resp_status == STATUS_OK)) begin
                                response_ready <= 1'b0;
                            end
                            if (train_chunk_mode == 3'd4) begin
                                if (($unsigned(infer_total_spikes) < TRAIN_RETRY_MIN_INJ_SPIKES) &&
                                    (train_chunk_retry_curr_max_fr + TRAIN_RETRY_MAX_FR_STEP <= TRAIN_RETRY_MAX_FR_LIMIT)) begin
                                    logic [15:0] next_max_fr_tmp;
                                    next_max_fr_tmp = train_chunk_retry_curr_max_fr + TRAIN_RETRY_MAX_FR_STEP;
                                    train_chunk_retry_curr_max_fr <= next_max_fr_tmp;
                                    infer_poisson_num_const_cfg <= (POISSON_NUM_CONST * next_max_fr_tmp) >> 5;
                                    train_chunk_state <= TCK_INFER_START;
                                end else begin
                                    train_chunk_retry_accepted_max_fr <= train_chunk_retry_curr_max_fr;
                                    train_chunk_state <= TCK_PICK_WINNER_INIT;
                                end
                            end else begin
                                if ((train_chunk_mode == 3'd2) || (train_chunk_mode == 3'd3)) begin
                                    train_chunk_state <= TCK_PICK_WINNER_INIT;
                                end else begin
                                    train_chunk_state <= TCK_GEN_XIN_START;
                                end
                            end
                        end
                    end
                    TCK_PICK_WINNER_INIT: begin
                        train_chunk_winner_scan_idx  <= 7'd0;
                        train_chunk_winner_best_idx  <= 7'd0;
                        train_chunk_winner_best_count<= infer_spike_count[0];
                        train_chunk_pre_from_infer   <= infer_last_active_input_idx;
                        train_chunk_state <= TCK_PICK_WINNER_SCAN;
                    end
                    TCK_PICK_WINNER_SCAN: begin
                        logic [6:0] next_idx_tmp;
                        next_idx_tmp = train_chunk_winner_scan_idx + 7'd1;
                        if (next_idx_tmp < N_NEURONS[6:0]) begin
                            if (infer_spike_count[next_idx_tmp] >= train_chunk_winner_best_count) begin
                                train_chunk_winner_best_count <= infer_spike_count[next_idx_tmp];
                                train_chunk_winner_best_idx   <= next_idx_tmp;
                            end
                            train_chunk_winner_scan_idx <= next_idx_tmp;
                        end else begin
                            train_chunk_winner <= train_chunk_winner_best_idx;
                            train_chunk_state  <= TCK_GEN_XIN_START;
                        end
                    end
                    TCK_BLANK_INFER_START: begin
                        if (raw_image0_valid && (raw_bytes_per_image == 32'd784) && (raw_image0_sum_u8 != 32'd0)) begin
                            infer_active       <= 1'b1;
                            infer_state        <= INFER_GEN_INPUT_SPIKES; // blank: continue existing state, skip clear/threshold prep
                            infer_steps_target <= TRAIN_MINE_NT_BLANK[31:0];
                            infer_step_idx     <= 16'd0;
                            infer_neuron_idx   <= 7'd0;
                            infer_input_idx    <= 10'd0;
                            infer_prep_idx     <= 10'd0;
                            infer_accum        <= 32'sd0;
                            infer_accum_weight_phase <= 3'd0;
                            infer_apply_idx    <= 7'd0;
                            infer_sum_c_inh    <= 32'd0;
                            infer_total_spikes <= 32'd0;
                            infer_rng_state    <= 32'h12345678;
                            infer_pre_active_count <= 10'd0;
                            infer_skip_init_clear <= 1'b1;
                            infer_force_no_input  <= 1'b1;
                            train_chunk_state <= TCK_BLANK_INFER_WAIT;
                        end else begin
                            resp_status       <= STATUS_BAD_PACKET;
                            resp_result       <= 32'h37E30001;
                            resp_checksum     <= calc_resp_checksum(STATUS_BAD_PACKET, 32'h37E30001);
                            response_ready    <= 1'b1;
                            train_chunk_active <= 1'b0;
                            train_chunk_mode  <= 2'd0;
                            train_chunk_state <= TCK_IDLE;
                        end
                    end
                    TCK_BLANK_INFER_WAIT: begin
                        if (!infer_active) begin
                            train_chunk_last_blank_spikes <= infer_total_spikes;
                            if (response_ready && (resp_status == STATUS_OK)) begin
                                response_ready <= 1'b0;
                            end
                            train_chunk_state <= TCK_DONE;
                        end
                    end
                    TCK_GEN_XIN_START: begin
                        if (train_chunk_steps_left == 16'd0) begin
                            train_chunk_state <= TCK_STDP_START;
                        end else begin
                            train_gen_active      <= 1'b1;
                            train_gen_state       <= TGK_WRITE_REQ;
                            train_gen_idx         <= 16'd0;
                            train_gen_lcg_state   <= train_chunk_seed_xin;
                            train_gen_curr_word   <= train_gen_word_from_state(train_chunk_seed_xin);
                            train_gen_lcg_enable  <= 1'b1;
                            train_gen_cache_mode  <= 2'd1;
                            train_xin_cache_valid <= 1'b0;
                            train_gen_base_word   <= TRAIN_BASE_XIN_WORK_WORDS;
                            train_gen_count_total <= N_IN[15:0];
                            train_chunk_state     <= TCK_GEN_XIN_WAIT;
                        end
                    end
                    TCK_GEN_XIN_WAIT: begin
                        if (!train_gen_active) begin
                            train_chunk_seed_xin <= ($unsigned(train_chunk_seed_xin) * LCG_A) + LCG_C;
                            train_chunk_state <= TCK_GEN_XEXC_START;
                        end
                    end
                    TCK_GEN_XEXC_START: begin
                        train_gen_active       <= 1'b1;
                        train_gen_state        <= TGK_WRITE_REQ;
                        train_gen_idx          <= 16'd0;
                        train_gen_lcg_state    <= train_chunk_seed_xexc;
                        train_gen_curr_word    <= train_gen_word_from_state(train_chunk_seed_xexc);
                        train_gen_lcg_enable   <= 1'b1;
                        train_gen_cache_mode   <= 2'd2;
                        train_xexc_cache_valid <= 1'b0;
                        train_gen_base_word    <= TRAIN_BASE_XEXC_WORK_WORDS;
                        train_gen_count_total  <= N_NEURONS[15:0];
                        train_chunk_state      <= TCK_GEN_XEXC_WAIT;
                    end
                    TCK_GEN_XEXC_WAIT: begin
                        if (!train_gen_active) begin
                            train_chunk_seed_xexc <= ($unsigned(train_chunk_seed_xexc) * LCG_A) + LCG_C;
                            train_chunk_state <= TCK_PRELIST_BUILD_INIT;
                        end
                    end
                    TCK_PRELIST_BUILD_INIT: begin
                        train_chunk_pre_scan_idx    <= 10'd0;
                        train_chunk_pre_write_count <= 10'd0;
                        train_chunk_state           <= TCK_PRELIST_BUILD_SCAN;
                    end
                    TCK_PRELIST_BUILD_SCAN: begin
                        if (train_chunk_pre_scan_idx >= N_IN) begin
                            train_chunk_state <= TCK_TRACE_START;
                        end else if (infer_input_spike[train_chunk_pre_scan_idx]) begin
                            train_chunk_pre_from_infer <= train_chunk_pre_scan_idx;
                            train_chunk_state <= TCK_PRELIST_WRITE_REQ;
                        end else begin
                            train_chunk_pre_scan_idx <= train_chunk_pre_scan_idx + 10'd1;
                        end
                    end
                    TCK_PRELIST_WRITE_REQ: begin
                        ddr_req_pending_core      <= 1'b1;
                        ddr_req_we_core           <= 1'b1;
                        ddr_req_from_sd_core      <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core   <= 1'b0; // generic write path response is fine; chunk waits on ddr_req_pending_core
                        ddr_req_addr_word_core    <= TRAIN_BASE_PRELIST_WORK_WORDS + {22'd0, train_chunk_pre_write_count};
                        ddr_req_wdata_core        <= {22'd0, train_chunk_pre_from_infer};
                        ddr_req_wide_core         <= 1'b0;
                        ddr_req_wdata128_core     <= 128'd0;
                        ddr_req_sel16_core        <= 16'd0;
                        ddr_req_word_count_core   <= 3'd1;
                        ddr_req_toggle_core       <= ~ddr_req_toggle_core;
                        train_chunk_state         <= TCK_PRELIST_WRITE_WAIT;
                    end
                    TCK_PRELIST_WRITE_WAIT: begin
                        if (!ddr_req_pending_core) begin
                            if (response_ready && (resp_status == STATUS_OK)) begin
                                response_ready <= 1'b0;
                            end
                            train_chunk_pre_write_count <= train_chunk_pre_write_count + 10'd1;
                            train_chunk_pre_scan_idx    <= train_chunk_pre_scan_idx + 10'd1;
                            train_chunk_state <= TCK_PRELIST_BUILD_SCAN;
                        end
                    end
                    TCK_TRACE_START: begin
                        train_trace_active <= 1'b1;
                        train_winner_idx   <= train_chunk_winner;
                        train_pre_count    <= train_chunk_pre_write_count;
                        train_a_idx        <= 10'd0;
                        train_pre_idx      <= 10'd0;
                        train_b_col_idx    <= 7'd0;
                        train_curr_pre     <= 10'd0;
                        train_tmp_x_val    <= 32'd0;
                        train_tmp_mem_val  <= 32'd0;
                        train_trace_a_row_base <= TRAIN_BASE_A_Q16_WORDS + ({25'd0, train_chunk_winner} * N_IN);
                        train_trace_bt_pre_base <= TRAIN_BASE_BT_Q16_WORDS;
                        train_trace_state  <= TRK_A_READ_X_REQ;
                        train_chunk_state  <= TCK_TRACE_WAIT;
                    end
                    TCK_TRACE_WAIT: begin
                        if (!train_trace_active) begin
                            train_chunk_winner <= (train_chunk_winner == (N_NEURONS-1)) ? 7'd0 : (train_chunk_winner + 7'd1);
                            train_chunk_pre_idx <= (train_chunk_pre_idx == (N_IN-1)) ? 10'd0 : (train_chunk_pre_idx + 10'd1);
                            if (train_chunk_steps_left > 16'd0)
                                train_chunk_steps_left <= train_chunk_steps_left - 16'd1;
                            train_chunk_state <= TCK_GEN_XIN_START;
                        end
                    end
                    TCK_STDP_START: begin
                        train_stdp_batch_active    <= 1'b1;
                        train_stdp_batch_tile_rows <= train_chunk_tile_rows;
                        train_stdp_batch_next_row0 <= 7'd0;
                        train_stdp_active          <= 1'b1;
                        train_stdp_row0            <= 7'd0;
                        if ({1'b0, train_chunk_tile_rows} >= N_NEURONS[7:0])
                            train_stdp_row_end <= N_NEURONS[6:0];
                        else
                            train_stdp_row_end <= train_chunk_tile_rows;
                        train_stdp_row_idx         <= 7'd0;
                        train_stdp_col_idx         <= 10'd0;
                        train_stdp_w_val           <= 32'sd0;
                        train_stdp_a_val           <= 32'sd0;
                        train_stdp_bt_val          <= 32'sd0;
                        train_stdp_w_new           <= 32'sd0;
                        train_stdp_row_sum_abs     <= 32'd0;
                        train_stdp_w_row_base      <= TRAIN_BASE_W_Q16_WORDS;
                        train_stdp_a_row_base      <= TRAIN_BASE_A_Q16_WORDS;
                        train_stdp_bt_col_base     <= TRAIN_BASE_BT_Q16_WORDS;
                        train_stdp_state           <= TSK_SUM_READ_W_REQ;
                        train_chunk_state          <= TCK_STDP_WAIT;
                    end
                    TCK_STDP_WAIT: begin
                        if (!train_stdp_active && !train_stdp_batch_active) begin
                            train_chunk_state <= TCK_DONE;
                        end
                    end
                    TCK_DONE: begin
                        if ((train_chunk_mode == 3'd3) || (train_chunk_mode == 3'd4)) begin
                            train_chunk_active       <= 1'b0;
                            train_chunk_state        <= TCK_IDLE;
                            train_chunk_mode         <= 2'd0;
                            train_chunk_samples_left <= 16'd0;
                            if (train_chunk_mode == 3'd4) begin
                                infer_poisson_num_const_cfg <= POISSON_NUM_CONST;
                            end
                            // Return inj/blank totals packed as [31:16]=blank, [15:0]=inj (truncated)
                            resp_status    <= STATUS_OK;
                            resp_result    <= {train_chunk_last_blank_spikes[15:0], train_chunk_last_infer_spikes[15:0]};
                            resp_checksum  <= calc_resp_checksum(STATUS_OK, {train_chunk_last_blank_spikes[15:0], train_chunk_last_infer_spikes[15:0]});
                            response_ready <= 1'b1;
                        end else begin
                            // STDP batch completion path returns the response in phase0/phase1/phase2.
                            train_chunk_state <= TCK_IDLE;
                        end
                    end
                    default: begin end
                endcase
            end

            if (TRAIN_ENABLE && train_gen_active && !ddr_req_pending_core && !response_ready && !sd_ddr_flush_active && !imgload_word_valid &&
                !train_trace_active && !train_stdp_active) begin
                case (train_gen_state)
                    TGK_WRITE_REQ: begin
                        if (train_gen_cache_mode == 2'd1) begin
                            train_xin_cache[train_gen_idx] <= train_gen_curr_word;
                        end else if (train_gen_cache_mode == 2'd2) begin
                            train_xexc_cache[train_gen_idx[6:0]] <= train_gen_curr_word;
                        end
                        ddr_req_pending_core      <= 1'b1;
                        ddr_req_we_core           <= 1'b1;
                        ddr_req_from_sd_core      <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core   <= 1'b1;
                        ddr_req_addr_word_core    <= train_gen_base_word + {16'd0, train_gen_idx};
                        ddr_req_wdata_core        <= train_gen_curr_word;
                        ddr_req_wide_core         <= 1'b0;
                        ddr_req_wdata128_core     <= 128'd0;
                        ddr_req_sel16_core        <= 16'd0;
                        ddr_req_word_count_core   <= 3'd1;
                        ddr_req_toggle_core       <= ~ddr_req_toggle_core;
                        train_gen_state           <= TGK_WRITE_WAIT;
                    end
                    TGK_DONE: begin
                        if (train_gen_cache_mode == 2'd1) begin
                            train_xin_cache_valid <= 1'b1;
                        end else if (train_gen_cache_mode == 2'd2) begin
                            train_xexc_cache_valid <= 1'b1;
                        end
                        train_gen_active <= 1'b0;
                        train_gen_state  <= TGK_IDLE;
                        if (!train_chunk_active) begin
                            resp_status    <= STATUS_OK;
                            resp_result    <= {16'd0, train_gen_count_total};
                            resp_checksum  <= calc_resp_checksum(STATUS_OK, {16'd0, train_gen_count_total});
                            response_ready <= 1'b1;
                        end
                    end
                    default: begin end
                endcase
            end

            if (TRAIN_ENABLE && train_label_stats_active && !response_ready && !sd_ddr_flush_active && !imgload_word_valid &&
                !ddr_req_pending_core && !train_trace_active && !train_stdp_active && !train_gen_active && !train_chunk_active && !infer_active) begin
                case (train_label_stats_state)
                    TLS_RESET_SUM: begin
                        train_label_spike_sum[train_label_stats_idx] <= 32'd0;
                        if (train_label_stats_idx == ((10*N_NEURONS)-1)) begin
                            train_label_stats_idx <= 10'd0;
                            train_label_stats_state <= TLS_RESET_COUNT;
                        end else begin
                            train_label_stats_idx <= train_label_stats_idx + 10'd1;
                        end
                    end
                    TLS_RESET_COUNT: begin
                        train_label_count[train_label_stats_idx[3:0]] <= 32'd0;
                        if (train_label_stats_idx[3:0] == 4'd9) begin
                            train_label_stats_state <= TLS_DONE;
                        end else begin
                            train_label_stats_idx <= train_label_stats_idx + 10'd1;
                        end
                    end
                    TLS_ACCUM_READ: begin
                        train_label_sum_rd_addr <= train_label_stats_base_idx + train_label_stats_idx;
                        train_label_stats_state <= TLS_ACCUM_WAIT;
                    end
                    TLS_ACCUM_WAIT: begin
                        train_label_stats_state <= TLS_ACCUM_WRITE;
                    end
                    TLS_ACCUM_WRITE: begin
                        train_label_spike_sum[train_label_stats_base_idx + train_label_stats_idx]
                            <= train_label_sum_rd_data + {16'd0, train_inj_spike_count_snap[train_label_stats_idx[6:0]]};
                        if (train_label_stats_idx == (N_NEURONS-1)) begin
                            train_label_count[train_label_stats_label] <= train_label_count[train_label_stats_label] + 32'd1;
                            train_label_stats_state <= TLS_DONE;
                        end else begin
                            train_label_stats_idx <= train_label_stats_idx + 10'd1;
                            train_label_stats_state <= TLS_ACCUM_READ;
                        end
                    end
                    TLS_DONE: begin
                        train_label_stats_active <= 1'b0;
                        train_label_stats_state <= TLS_IDLE;
                        resp_status    <= STATUS_OK;
                        resp_result    <= {28'd0, train_label_stats_label};
                        resp_checksum  <= calc_resp_checksum(STATUS_OK, {28'd0, train_label_stats_label});
                        response_ready <= 1'b1;
                    end
                    default: begin
                        train_label_stats_active <= 1'b0;
                        train_label_stats_state <= TLS_IDLE;
                        resp_status    <= STATUS_BAD_PACKET;
                        resp_result    <= 32'sd0;
                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                        response_ready <= 1'b1;
                    end
                endcase
            end

            if (rx_dv && !response_ready && !memrd_pending && !sd_copy_active && !infer_active && !imgload_active) begin
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
                            ((req_opcode == OP_ADD_I32) ||
                             (req_opcode == OP_SD_SECTORS_TO_DDR) || (req_opcode == OP_LOAD_IMAGE_FROM_DDR) ||
                             (req_opcode == OP_TRAIN_QUERY_CAPS) ||
                             (req_opcode == OP_TRAIN_RUN_SAMPLE_PHASE3) || (req_opcode == OP_TRAIN_RUN_SAMPLE_PHASE4) ||
                             (req_opcode == OP_TRAIN_LABEL_STATS_RESET) || (req_opcode == OP_TRAIN_LABEL_STATS_ACCUM) ||
                             (req_opcode == OP_READ_TRAIN_LABEL_STAT_SUM) || (req_opcode == OP_READ_TRAIN_LABEL_STAT_COUNT))
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
                        end else if (TRAIN_ENABLE &&
                                     (train_trace_active || train_stdp_active || train_gen_active || train_label_stats_active || train_stdp_batch_active || train_chunk_active)) begin
                            resp_status    <= STATUS_BAD_PACKET;
                            resp_result    <= {8'h31, req_opcode, 16'h0000}; // TRAIN_BUSY debug tag
                            resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {8'h31, req_opcode, 16'h0000});
                            response_ready <= 1'b1;
                        end else if (!TRAIN_ENABLE &&
                                     ((req_opcode == OP_DDR_ZERO32) ||
                                      ((req_opcode >= OP_TRACE_UPDATE) && (req_opcode <= OP_READ_TRAIN_LABEL_STAT_COUNT)))) begin
                            resp_status    <= STATUS_UNSUPPORTED_OP;
                            resp_result    <= 32'sd0;
                            resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                            response_ready <= 1'b1;
                        end else if (!DEV_UART_OPS_ENABLE &&
                                     ((req_opcode == OP_DDR_WRITE32) ||
                                      (req_opcode == OP_SD_TO_DDR_COPY) ||
                                      (req_opcode == OP_DDR_READ32) ||
                                      (req_opcode == OP_DDR_ZERO32) ||
                                      (req_opcode == OP_RUN_SAMPLE_INFER) ||
                                      (req_opcode == OP_READ_SPIKE_COUNT) ||
                                      (req_opcode == OP_READ_RAW_U8) ||
                                      (req_opcode == OP_WRITE_INFER_WEIGHT) ||
                                      (req_opcode == OP_SET_POISSON_MAX_FR) ||
                                      (req_opcode == OP_READ_INFER_DEBUG) ||
                                      (req_opcode == OP_TRACE_UPDATE) ||
                                      (req_opcode == OP_STDP_UPDATE_TILE) ||
                                      (req_opcode == OP_TRAIN_GEN_WORK) ||
                                      (req_opcode == OP_READ_TRAIN_DEBUG) ||
                                      (req_opcode == OP_STDP_UPDATE_ALL) ||
                                      (req_opcode == OP_TRAIN_RUN_CHUNK) ||
                                      (req_opcode == OP_READ_TRAIN_INJ_SPIKE_COUNT) ||
                                      (req_opcode == OP_READ_POISSON_THRESH))) begin
                            resp_status    <= STATUS_UNSUPPORTED_OP;
                            resp_result    <= 32'sd0;
                            resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
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
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < DDR_ADDR_WORD_LIMIT) &&
                                        ddr_calib_complete && !ddr_req_pending_core) begin
                                        train_xin_cache_valid  <= 1'b0;
                                        train_xexc_cache_valid <= 1'b0;
                                        ddr_req_pending_core   <= 1'b1;
                                        ddr_req_we_core        <= 1'b1;
                                        ddr_req_from_sd_core   <= 1'b0;
                                        ddr_req_from_imgload_core <= 1'b0;
                                        ddr_req_from_train_core <= 1'b0;
                                        ddr_req_addr_word_core <= arg0;
                                        ddr_req_wdata_core     <= arg1;
                                        ddr_req_wide_core      <= 1'b0;
                                        ddr_req_wdata128_core  <= 128'd0;
                                        ddr_req_sel16_core     <= 16'd0;
                                        ddr_req_word_count_core <= 3'd1;
                                        ddr_req_toggle_core    <= ~ddr_req_toggle_core;
                                        ddr_last_addr          <= arg0;
                                        ddr_last_data          <= arg1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_DDR_READ32: begin
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < DDR_ADDR_WORD_LIMIT) &&
                                        ddr_calib_complete && !ddr_req_pending_core) begin
                                        ddr_req_pending_core   <= 1'b1;
                                        ddr_req_we_core        <= 1'b0;
                                        ddr_req_from_sd_core   <= 1'b0;
                                        ddr_req_from_imgload_core <= 1'b0;
                                        ddr_req_from_train_core <= 1'b0;
                                        ddr_req_addr_word_core <= arg0;
                                        ddr_req_wdata_core     <= 32'd0;
                                        ddr_req_wide_core      <= 1'b0;
                                        ddr_req_wdata128_core  <= 128'd0;
                                        ddr_req_sel16_core     <= 16'd0;
                                        ddr_req_word_count_core <= 3'd1;
                                        ddr_req_toggle_core    <= ~ddr_req_toggle_core;
                                        ddr_last_addr          <= arg0;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_SD_TO_DDR_COPY: begin
                                    // Legacy RAW1 auto-header copy is disabled to save LUTs.
                                    // Use OP_SD_SECTORS_TO_DDR + OP_LOAD_IMAGE_FROM_DDR instead.
                                    resp_status    <= STATUS_UNSUPPORTED_OP;
                                    resp_result    <= 32'sd0;
                                    resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                    response_ready <= 1'b1;
                                end
                                OP_SD_SECTORS_TO_DDR: begin
                                    if (
                                        (req_nargs == 8'd2) &&
                                        (arg0 >= 0) &&
                                        (arg1 > 0) &&
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
                                        sd_sector_ddr_base_word_bank[0] <= 32'd0;
                                        sd_sector_ddr_base_word_bank[1] <= 32'd0;
                                        sd_sector_words_queued_bank[0] <= 8'd0;
                                        sd_sector_words_queued_bank[1] <= 8'd0;
                                        sd_sector_buf_ready <= 2'b00;
                                        sd_fill_bank <= 1'b0;
                                        sd_flush_bank <= 1'b0;
                                        sd_ddr_flush_active <= 1'b0;
                                        sd_ddr_flush_idx <= 8'd0;
                                        sd_wait_counter      <= 24'd0;
                                        sd_header_done       <= 1'b0;
                                        sd_file_total_bytes  <= 32'd0;
                                        sd_file_bytes_seen   <= 32'd0;
                                        sd_copy_done_pending <= 1'b0;
                                        sd_use_sector_limit  <= 1'b1;
                                        sd_copy_raw1_mode    <= 1'b0;
                                        sd_copy_dest_base_word <= 32'd0;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= {BADDBG_SD_REQ_ARG, req_opcode, arg0[15:0]};
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {BADDBG_SD_REQ_ARG, req_opcode, arg0[15:0]});
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_LOAD_IMAGE_FROM_DDR: begin
                                    if ((req_nargs == 8'd2) &&
                                        (arg0 >= 0) &&
                                        (arg1 > 0) && (arg1 <= N_IN) &&
                                        ddr_calib_complete && !ddr_req_pending_core &&
                                        !imgload_active) begin
                                        // Clear residual SD DMA flush flags so imgload issue is not blocked.
                                        sd_ddr_flush_active <= 1'b0;
                                        sd_sector_buf_ready <= 2'b00;
                                        imgload_active <= 1'b1;
                                        imgload_addr_word <= {2'b00, arg0[31:2]};
                                        imgload_lane <= arg0[1:0];
                                        imgload_byte_idx <= 10'd0;
                                        imgload_total_bytes <= arg1[9:0];
                                        imgload_sum_u8_accum <= 32'd0;
                                        imgload_word_valid <= 1'b0;
                                        imgload_word_data <= 32'd0;
                                        imgload_word_lane <= 2'd0;
                                        raw_image0_valid <= 1'b0;
                                        raw_image0_capture_idx <= 10'd0;
                                        raw_image0_sum_u8 <= 32'd0;
                                        raw_bytes_per_image <= arg1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_DDR_ZERO32: begin
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) &&
                                        (arg0 >= 0) &&
                                        (arg1 > 0) && (arg1 <= 32'sd65535) &&
                                        ddr_calib_complete &&
                                        !ddr_req_pending_core &&
                                        !train_trace_active &&
                                        !train_stdp_active &&
                                        !train_gen_active) begin
                                        train_gen_active      <= 1'b1;
                                        train_gen_state       <= TGK_WRITE_REQ;
                                        train_gen_base_word   <= arg0[31:0];
                                        train_gen_count_total <= arg1[15:0];
                                        train_gen_idx         <= 16'd0;
                                        train_gen_lcg_state   <= 32'd0;
                                        train_gen_curr_word   <= 32'd0;
                                        train_gen_lcg_enable  <= 1'b0;
                                        train_gen_cache_mode  <= 2'd0;
                                        train_xin_cache_valid <= 1'b0;
                                        train_xexc_cache_valid<= 1'b0;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_RUN_SAMPLE_INFER: begin
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if (
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
                                        infer_accum_weight_phase <= 3'd0;
                                        infer_apply_idx    <= 7'd0;
                                        infer_sum_c_inh    <= 32'sd0;
                                        infer_total_spikes <= 32'd0;
                                        infer_rng_state    <= arg0;
                                        infer_pre_active_count <= 10'd0;
                                        infer_skip_init_clear <= 1'b0;
                                        infer_force_no_input  <= 1'b0;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_READ_SPIKE_COUNT: begin
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < N_NEURONS)) begin
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
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) && raw_image0_valid && (arg0 >= 0) && (arg0 < N_IN)) begin
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
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < N_IN)) begin
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
                                OP_READ_TRAIN_INJ_SPIKE_COUNT: begin
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < N_NEURONS)) begin
                                        infer_spike_count_rd_addr <= arg0[6:0];
                                        memrd_idx     <= arg0[15:0];
                                        memrd_kind    <= MEMRD_TRAIN_INJ_SPIKE_COUNT;
                                        memrd_wait    <= 1'b1;
                                        memrd_pending <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= {8'h16, req_opcode, arg0[15:0]};
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {8'h16, req_opcode, arg0[15:0]});
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRAIN_LABEL_STATS_RESET: begin
                                    if ((req_nargs == 8'd2) &&
                                        !train_trace_active && !train_stdp_active && !train_gen_active &&
                                        !train_label_stats_active && !train_chunk_active && !infer_active) begin
                                        train_label_stats_active <= 1'b1;
                                        train_label_stats_state <= TLS_RESET_SUM;
                                        train_label_stats_idx <= 10'd0;
                                        train_label_stats_label <= 4'd0;
                                        train_label_stats_base_idx <= 10'd0;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRAIN_LABEL_STATS_ACCUM: begin
                                    if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < 32'sd10) &&
                                        !train_trace_active && !train_stdp_active && !train_gen_active &&
                                        !train_label_stats_active && !train_chunk_active && !infer_active) begin
                                        logic [9:0] base_tmp;
                                        base_tmp = ({6'd0, arg0[3:0]} * N_NEURONS);
                                        train_label_stats_active <= 1'b1;
                                        train_label_stats_state <= TLS_ACCUM_READ;
                                        train_label_stats_label <= arg0[3:0];
                                        train_label_stats_idx <= 10'd0;
                                        train_label_stats_base_idx <= base_tmp;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_READ_TRAIN_LABEL_STAT_SUM: begin
                                    if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < 32'sd10) &&
                                        (arg1 >= 0) && (arg1 < N_NEURONS)) begin
                                        train_label_sum_rd_addr <= ({6'd0, arg0[3:0]} * N_NEURONS) + arg1[6:0];
                                        memrd_idx     <= arg1[15:0];
                                        memrd_kind    <= MEMRD_TRAIN_LABEL_STAT_SUM;
                                        memrd_wait    <= 1'b1;
                                        memrd_pending <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= {8'h17, req_opcode, arg0[15:0]};
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {8'h17, req_opcode, arg0[15:0]});
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_READ_TRAIN_LABEL_STAT_COUNT: begin
                                    if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < 32'sd10)) begin
                                        train_label_count_rd_addr <= arg0[3:0];
                                        memrd_idx     <= arg0[15:0];
                                        memrd_kind    <= MEMRD_TRAIN_LABEL_STAT_COUNT;
                                        memrd_wait    <= 1'b1;
                                        memrd_pending <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= {8'h18, req_opcode, arg0[15:0]};
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, {8'h18, req_opcode, arg0[15:0]});
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_READ_INFER_DEBUG: begin
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < 32'sd13)) begin
                                        logic signed [31:0] dbg_value;
                                        resp_status <= STATUS_OK;
                                        case (arg0[4:0])
                                            5'd0: dbg_value = 32'd0;
                                            5'd1: dbg_value = 32'd0;
                                            5'd2: dbg_value = 32'd0;
                                            5'd3: dbg_value = 32'd0;
                                            5'd4: dbg_value = 32'd0;
                                            5'd5: dbg_value = 32'd0;
                                            5'd6: dbg_value = 32'd0;
                                            5'd7: dbg_value = infer_total_spikes;
                                            5'd8: dbg_value = infer_steps_target;
                                            5'd9: dbg_value = {16'd0, infer_step_idx};
                                            5'd10: dbg_value = {29'd0, infer_state};
                                            5'd11: dbg_value = raw_image0_sum_u8;
                                            default: dbg_value = infer_poisson_num_const_cfg;
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
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < N_WEIGHTS) && !infer_active) begin
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
                                OP_SET_POISSON_MAX_FR: begin
                                    // arg0=max_fr (positive integer), arg1 reserved.
                                    // Runtime Poisson numerator scales linearly from the default 32 Hz base.
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) && (arg0 > 0) && (arg0 <= 32'sd4096) && !infer_active) begin
                                        infer_poisson_num_const_cfg <= (POISSON_NUM_CONST * arg0[15:0]) >> 5; // *max_fr/32
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= (POISSON_NUM_CONST * arg0[15:0]) >> 5;
                                        resp_checksum  <= calc_resp_checksum(STATUS_OK, (POISSON_NUM_CONST * arg0[15:0]) >> 5);
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRAIN_QUERY_CAPS: begin
                                    if (req_nargs == 8'd2) begin
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= TRAIN_ENABLE ? TRAIN_CAPS_VALUE : 32'd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_OK, TRAIN_ENABLE ? TRAIN_CAPS_VALUE : 32'd0);
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRACE_UPDATE: begin
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) &&
                                        (arg0 >= -1) && (arg0 < N_NEURONS) &&
                                        (arg1 >= 0) && (arg1 <= N_IN) &&
                                        ddr_calib_complete &&
                                        !ddr_req_pending_core &&
                                        !train_trace_active) begin
                                        train_trace_active <= 1'b1;
                                        train_winner_idx   <= (arg0 >= 0) ? arg0[6:0] : 7'd0;
                                        train_pre_count    <= arg1[9:0];
                                        train_a_idx        <= 10'd0;
                                        train_pre_idx      <= 10'd0;
                                        train_b_col_idx    <= 7'd0;
                                        train_curr_pre     <= 10'd0;
                                        train_tmp_x_val    <= 32'd0;
                                        train_tmp_mem_val  <= 32'd0;
                                        train_trace_a_row_base <= TRAIN_BASE_A_Q16_WORDS + ({25'd0, ((arg0 >= 0) ? arg0[6:0] : 7'd0)} * N_IN);
                                        train_trace_bt_pre_base <= TRAIN_BASE_BT_Q16_WORDS;
                                        if (arg0 >= 0) begin
                                            train_trace_state <= TRK_A_READ_X_REQ;
                                        end else begin
                                            train_trace_state <= TRK_B_READ_PRE_REQ;
                                        end
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= TRAIN_CAPS_VALUE;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, TRAIN_CAPS_VALUE);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_STDP_UPDATE_TILE: begin
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) &&
                                        (arg0 >= 0) && (arg0 < N_NEURONS) &&
                                        (arg1 > 0) && ((arg0 + arg1) <= N_NEURONS) &&
                                        ddr_calib_complete &&
                                        !ddr_req_pending_core &&
                                        !train_trace_active &&
                                        !train_stdp_active) begin
                                        train_stdp_active  <= 1'b1;
                                        train_stdp_row0    <= arg0[6:0];
                                        train_stdp_row_end <= arg0[6:0] + arg1[6:0];
                                        train_stdp_row_idx <= arg0[6:0];
                                        train_stdp_col_idx <= 10'd0;
                                        train_stdp_w_val   <= 32'sd0;
                                        train_stdp_a_val   <= 32'sd0;
                                        train_stdp_bt_val  <= 32'sd0;
                                        train_stdp_w_new   <= 32'sd0;
                                        train_stdp_row_sum_abs <= 32'd0;
                                        train_stdp_w_row_base <= TRAIN_BASE_W_Q16_WORDS + ({25'd0, arg0[6:0]} * N_IN);
                                        train_stdp_a_row_base <= TRAIN_BASE_A_Q16_WORDS + ({25'd0, arg0[6:0]} * N_IN);
                                        train_stdp_bt_col_base <= TRAIN_BASE_BT_Q16_WORDS + {25'd0, arg0[6:0]};
                                        train_stdp_state   <= TSK_SUM_READ_W_REQ;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= TRAIN_CAPS_VALUE;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, TRAIN_CAPS_VALUE);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_STDP_UPDATE_ALL: begin
                                    // arg0 = tile_rows (1..N_NEURONS), arg1 reserved
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) &&
                                        (arg0 > 0) && (arg0 <= N_NEURONS) &&
                                        ddr_calib_complete &&
                                        !ddr_req_pending_core &&
                                        !train_trace_active &&
                                        !train_stdp_active &&
                                        !train_stdp_batch_active) begin
                                        train_stdp_batch_active    <= 1'b1;
                                        train_stdp_batch_tile_rows <= arg0[6:0];
                                        train_stdp_batch_next_row0 <= 7'd0;
                                        train_stdp_active          <= 1'b1;
                                        train_stdp_row0            <= 7'd0;
                                        if (arg0 >= N_NEURONS)
                                            train_stdp_row_end <= N_NEURONS[6:0];
                                        else
                                            train_stdp_row_end <= arg0[6:0];
                                        train_stdp_row_idx         <= 7'd0;
                                        train_stdp_col_idx         <= 10'd0;
                                        train_stdp_w_val           <= 32'sd0;
                                        train_stdp_a_val           <= 32'sd0;
                                        train_stdp_bt_val          <= 32'sd0;
                                        train_stdp_w_new           <= 32'sd0;
                                        train_stdp_row_sum_abs     <= 32'd0;
                                        train_stdp_w_row_base      <= TRAIN_BASE_W_Q16_WORDS;
                                        train_stdp_a_row_base      <= TRAIN_BASE_A_Q16_WORDS;
                                        train_stdp_bt_col_base     <= TRAIN_BASE_BT_Q16_WORDS;
                                        train_stdp_state           <= TSK_SUM_READ_W_REQ;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= TRAIN_CAPS_VALUE;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, TRAIN_CAPS_VALUE);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRAIN_RUN_CHUNK: begin
                                    // Phase0: arg0>0,arg1>0 => repeat full-row STDP batch `arg0` times.
                                    // Phase1: arg0<0,arg1>0 => synthetic trace loop (`-arg0` steps) + STDP batch.
                                    // Phase2: arg0<0,arg1<0 => run infer (`-arg0` steps), then phase1 synthetic trace+STDP with tile_rows=`-arg1`.
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) &&
                                        (arg0 != 0) && ((arg0 <= 32'sd65535) && (arg0 >= -32'sd65535)) &&
                                        (((arg1 > 0) && (arg1 <= N_NEURONS)) || ((arg1 < 0) && ((-arg1) <= N_NEURONS))) &&
                                        ddr_calib_complete &&
                                        !ddr_req_pending_core &&
                                        !train_trace_active &&
                                        !train_stdp_active &&
                                        !train_stdp_batch_active &&
                                        !train_chunk_active) begin
                                        train_chunk_active       <= 1'b1;
                                        if (arg0 > 0)
                                            train_chunk_mode <= 3'd0;
                                        else if (arg1 < 0)
                                            train_chunk_mode <= 3'd2;
                                        else
                                            train_chunk_mode <= 3'd1;
                                        train_chunk_state        <= (arg0 < 0) ? ((arg1 < 0) ? TCK_INFER_START : TCK_GEN_XIN_START) : TCK_IDLE;
                                        train_chunk_samples_left <= (arg0 > 0) ? arg0[15:0] : 16'd1;
                                        train_chunk_tile_rows    <= (arg1 > 0) ? arg1 : -arg1;
                                        train_chunk_steps_left    <= (arg0 < 0) ? (-arg0) : 16'd0;
                                        train_chunk_seed_xin      <= 32'h13579BDF;
                                        train_chunk_seed_xexc     <= 32'h2468ACE1;
                                        train_chunk_winner        <= 7'd0;
                                        train_chunk_pre_idx       <= 10'd0;
                                        train_chunk_last_infer_spikes <= 32'd0;
                                        train_chunk_last_blank_spikes <= 32'd0;
                                        train_chunk_retry_curr_max_fr <= TRAIN_RETRY_MAX_FR_START;
                                        train_chunk_retry_accepted_max_fr <= TRAIN_RETRY_MAX_FR_START;
                                        if (arg0 > 0) begin
                                            // Start first STDP batch immediately (phase0)
                                            train_stdp_batch_active    <= 1'b1;
                                            train_stdp_batch_tile_rows <= (arg1 > 0) ? arg1 : -arg1;
                                            train_stdp_batch_next_row0 <= 7'd0;
                                            train_stdp_active          <= 1'b1;
                                            train_stdp_row0            <= 7'd0;
                                            if (((arg1 > 0) ? arg1 : -arg1) >= N_NEURONS)
                                                train_stdp_row_end <= N_NEURONS[6:0];
                                            else
                                                train_stdp_row_end <= (arg1 > 0) ? arg1 : -arg1;
                                            train_stdp_row_idx         <= 7'd0;
                                            train_stdp_col_idx         <= 10'd0;
                                            train_stdp_w_val           <= 32'sd0;
                                            train_stdp_a_val           <= 32'sd0;
                                            train_stdp_bt_val          <= 32'sd0;
                                            train_stdp_w_new           <= 32'sd0;
                                            train_stdp_row_sum_abs     <= 32'd0;
                                            train_stdp_w_row_base      <= TRAIN_BASE_W_Q16_WORDS;
                                            train_stdp_a_row_base      <= TRAIN_BASE_A_Q16_WORDS;
                                            train_stdp_bt_col_base     <= TRAIN_BASE_BT_Q16_WORDS;
                                            train_stdp_state           <= TSK_SUM_READ_W_REQ;
                                        end
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= TRAIN_CAPS_VALUE;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, TRAIN_CAPS_VALUE);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRAIN_RUN_SAMPLE_PHASE3: begin
                                    // arg0 = inj steps (>0), arg1 = tile_rows (>0); blank steps fixed to TRAIN_MINE_NT_BLANK
                                    if ((req_nargs == 8'd2) &&
                                        (arg0 > 0) && (arg0 <= 32'sd65535) &&
                                        (arg1 > 0) && (arg1 <= N_NEURONS) &&
                                        ddr_calib_complete &&
                                        !ddr_req_pending_core &&
                                        !train_trace_active &&
                                        !train_stdp_active &&
                                        !train_stdp_batch_active &&
                                        !train_chunk_active) begin
                                        train_chunk_active       <= 1'b1;
                                        train_chunk_mode         <= 3'd3;
                                        train_chunk_state        <= TCK_INFER_START;
                                        train_chunk_samples_left <= 16'd1;
                                        train_chunk_tile_rows    <= arg1;
                                        train_chunk_steps_left   <= arg0[15:0];
                                        train_chunk_seed_xin     <= 32'h13579BDF;
                                        train_chunk_seed_xexc    <= 32'h2468ACE1;
                                        train_chunk_winner       <= 7'd0;
                                        train_chunk_pre_idx      <= 10'd0;
                                        train_chunk_last_infer_spikes <= 32'd0;
                                        train_chunk_last_blank_spikes <= 32'd0;
                                        train_chunk_retry_curr_max_fr <= TRAIN_RETRY_MAX_FR_START;
                                        train_chunk_retry_accepted_max_fr <= TRAIN_RETRY_MAX_FR_START;
                                        infer_poisson_num_const_cfg <= POISSON_NUM_CONST;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= TRAIN_CAPS_VALUE;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, TRAIN_CAPS_VALUE);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRAIN_RUN_SAMPLE_PHASE4: begin
                                    // arg0 = inj steps (>0), arg1 = tile_rows (>0); retries max_fr in-FPGA before synthetic trace/STDP + blank.
                                    if ((req_nargs == 8'd2) &&
                                        (arg0 > 0) && (arg0 <= 32'sd65535) &&
                                        (arg1 > 0) && (arg1 <= N_NEURONS) &&
                                        ddr_calib_complete &&
                                        !ddr_req_pending_core &&
                                        !train_trace_active &&
                                        !train_stdp_active &&
                                        !train_stdp_batch_active &&
                                        !train_chunk_active) begin
                                        train_chunk_active       <= 1'b1;
                                        train_chunk_mode         <= 3'd4;
                                        train_chunk_state        <= TCK_INFER_START;
                                        train_chunk_samples_left <= 16'd1;
                                        train_chunk_tile_rows    <= arg1;
                                        train_chunk_steps_left   <= arg0[15:0];
                                        train_chunk_seed_xin     <= 32'h13579BDF;
                                        train_chunk_seed_xexc    <= 32'h2468ACE1;
                                        train_chunk_winner       <= 7'd0;
                                        train_chunk_pre_idx      <= 10'd0;
                                        train_chunk_last_infer_spikes <= 32'd0;
                                        train_chunk_last_blank_spikes <= 32'd0;
                                        train_chunk_retry_curr_max_fr <= TRAIN_RETRY_MAX_FR_START;
                                        train_chunk_retry_accepted_max_fr <= TRAIN_RETRY_MAX_FR_START;
                                        infer_poisson_num_const_cfg <= POISSON_NUM_CONST;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= TRAIN_CAPS_VALUE;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, TRAIN_CAPS_VALUE);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRAIN_GEN_WORK: begin
                                    // arg1 mode: 0=x_in_work (N_IN), 1=x_exc_work (N_NEURONS)
                                    if (!DEV_UART_OPS_ENABLE) begin
                                        resp_status    <= STATUS_UNSUPPORTED_OP;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
                                        response_ready <= 1'b1;
                                    end else if ((req_nargs == 8'd2) &&
                                        ddr_calib_complete &&
                                        !ddr_req_pending_core &&
                                        !train_trace_active &&
                                        !train_stdp_active &&
                                        !train_gen_active &&
                                        ((arg1 == 32'sd0) || (arg1 == 32'sd1))) begin
                                        train_gen_active      <= 1'b1;
                                        train_gen_state       <= TGK_WRITE_REQ;
                                        train_gen_idx         <= 16'd0;
                                        train_gen_lcg_state   <= arg0;
                                        train_gen_curr_word   <= train_gen_word_from_state(arg0);
                                        train_gen_lcg_enable  <= 1'b1;
                                        train_gen_cache_mode  <= (arg1 == 32'sd0) ? 2'd1 : 2'd2;
                                        if (arg1 == 32'sd0) begin
                                            train_xin_cache_valid <= 1'b0;
                                        end else begin
                                            train_xexc_cache_valid <= 1'b0;
                                        end
                                        if (arg1 == 32'sd0) begin
                                            train_gen_base_word   <= TRAIN_BASE_XIN_WORK_WORDS;
                                            train_gen_count_total <= N_IN[15:0];
                                        end else begin
                                            train_gen_base_word   <= TRAIN_BASE_XEXC_WORK_WORDS;
                                            train_gen_count_total <= N_NEURONS[15:0];
                                        end
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= TRAIN_CAPS_VALUE;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, TRAIN_CAPS_VALUE);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_READ_TRAIN_DEBUG: begin
                                    resp_status    <= STATUS_UNSUPPORTED_OP;
                                    resp_result    <= 32'sd0;
                                    resp_checksum  <= calc_resp_checksum(STATUS_UNSUPPORTED_OP, 32'sd0);
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
                if (!sd_ddr_flush_active) begin
                    if (sd_sector_buf_ready[sd_flush_bank]) begin
                        sd_ddr_flush_active <= 1'b1;
                        sd_ddr_flush_idx <= 8'd0;
                    end else if (sd_sector_buf_ready[~sd_flush_bank]) begin
                        sd_flush_bank <= ~sd_flush_bank;
                        sd_ddr_flush_active <= 1'b1;
                        sd_ddr_flush_idx <= 8'd0;
                    end
                end

                if (!sd_in_read && !sd_sector_buf_ready[sd_fill_bank] &&
                    (!sd_use_sector_limit || (sd_copy_sectors_left != 0)) && !sd_copy_done_pending) begin
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
                        sd_sector_ddr_base_word_bank[sd_fill_bank] <= sd_copy_dest_base_word + sd_copy_words_written;
                        sd_sector_words_queued_bank[sd_fill_bank] <= 8'd0;
                        sd_ddr_flush_idx <= 8'd0;
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
                    case (sd_pack_idx)
                        2'd0: sd_pack_word[7:0]   <= sd_dout;
                        2'd1: sd_pack_word[15:8]  <= sd_dout;
                        2'd2: sd_pack_word[23:16] <= sd_dout;
                        default: begin
                            sd_pack_word[31:24] <= sd_dout;
                            sd_sector_word_buf[sd_fill_bank][sd_sector_words_queued_bank[sd_fill_bank]] <= {sd_dout, sd_pack_word[23:0]};
                            sd_copy_words_written <= sd_copy_words_written + 32'd1;
                            sd_sector_words_queued_bank[sd_fill_bank] <= sd_sector_words_queued_bank[sd_fill_bank] + 8'd1;
                        end
                    endcase

                    sd_pack_idx <= sd_pack_idx + 2'd1;

                    if (sd_byte_count == 9'd511) begin
                        sd_in_read   <= 1'b0;
                        sd_copy_lba  <= sd_copy_lba + 32'd1;
                        if (sd_use_sector_limit && (sd_copy_sectors_left != 0)) begin
                            sd_copy_sectors_left <= sd_copy_sectors_left - 32'd1;
                        end
                        if (sd_pack_idx != 2'd0) begin
                            sd_sector_word_buf[sd_fill_bank][sd_sector_words_queued_bank[sd_fill_bank]] <= sd_pack_word;
                            sd_copy_words_written <= sd_copy_words_written + 32'd1;
                            sd_sector_words_queued_bank[sd_fill_bank] <= sd_sector_words_queued_bank[sd_fill_bank] + 8'd1;
                        end
                        if ((sd_sector_words_queued_bank[sd_fill_bank] != 8'd0) || (sd_pack_idx != 2'd0)) begin
                            sd_sector_buf_ready[sd_fill_bank] <= 1'b1;
                            if (!sd_ddr_flush_active) begin
                                sd_flush_bank <= sd_fill_bank;
                                sd_ddr_flush_active <= 1'b1;
                                sd_ddr_flush_idx <= 8'd0;
                            end
                        end else if (sd_copy_done_pending || (sd_use_sector_limit && (sd_copy_sectors_left == 32'd1))) begin
                            sd_copy_active <= 1'b0;
                            resp_status    <= STATUS_OK;
                            resp_result    <= sd_copy_words_written;
                            resp_checksum  <= calc_resp_checksum(STATUS_OK, sd_copy_words_written);
                            response_ready <= 1'b1;
                        end
                        if (sd_use_sector_limit && (sd_copy_sectors_left == 32'd1)) begin
                            sd_copy_done_pending <= 1'b1;
                        end
                        sd_fill_bank <= ~sd_fill_bank;
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
                                infer_dividend <= infer_poisson_num_const_cfg * {24'd0, raw_image0_u8[infer_prep_idx]};
                                infer_divisor  <= raw_image0_sum_u8;
                                infer_div_valid <= 1'b1;
                                infer_state <= INFER_PREP_DIV_WAIT;
                            end
                        end else begin
	                            infer_state <= INFER_GEN_INPUT_SPIKES;
	                            infer_input_idx <= 10'd0;
	                            infer_neuron_idx <= 7'd0;
	                            infer_accum <= 32'sd0;
                                infer_accum_weight_phase <= 3'd0;
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
                        if (infer_force_no_input) begin
                            spike_in_now = 1'b0;
                        end else begin
                            spike_in_now = (rng_next[31:21] < infer_poisson_thresh[infer_input_idx]);
                        end
                        infer_rng_state <= rng_next;
                        infer_input_spike[infer_input_idx] <= spike_in_now;
                        if (spike_in_now) begin
                            if (infer_input_idx == 10'd0) begin
                                infer_pre_active_list[10'd0] <= infer_input_idx;
                                infer_pre_active_count <= 10'd1;
                            end else begin
                                infer_pre_active_list[infer_pre_active_count] <= infer_input_idx;
                                infer_pre_active_count <= infer_pre_active_count + 10'd1;
                            end
                            infer_last_active_input_idx <= infer_input_idx;
                        end else if (infer_input_idx == 10'd0) begin
                            infer_pre_active_count <= 10'd0;
                        end
                        if (infer_input_idx == (N_IN - 1)) begin
                            infer_input_idx <= 10'd0;
                            infer_state <= INFER_ACCUM_NEURON;
                        end else begin
                            infer_input_idx <= infer_input_idx + 10'd1;
                        end
                    end

                    INFER_ACCUM_NEURON: begin
                        if (infer_input_idx < infer_pre_active_count) begin
                            logic [16:0] w_idx;
                            if (infer_accum_weight_phase == 3'd0) begin
                                infer_pre_rd_addr <= infer_input_idx;
                                infer_accum_weight_phase <= 3'd1;
                            end else if (infer_accum_weight_phase == 3'd1) begin
                                // prelist BRAM synchronous read latency fill cycle.
                                infer_accum_weight_phase <= 3'd2;
                            end else if (infer_accum_weight_phase == 3'd2) begin
                                w_idx = (infer_neuron_idx * N_IN) + infer_pre_rd_data;
                                infer_w_rd_addr <= w_idx;
                                infer_accum_weight_phase <= 3'd3;
                            end else if (infer_accum_weight_phase == 3'd3) begin
                                // weight BRAM synchronous read latency fill cycle.
                                infer_accum_weight_phase <= 3'd4;
                            end else begin
                                if (infer_w_rd_data != 16'd0) begin
                                    infer_accum <= infer_accum + $signed({16'd0, infer_w_rd_data});
                                end
                                infer_input_idx <= infer_input_idx + 10'd1;
                                infer_accum_weight_phase <= 3'd0;
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
                            infer_accum_weight_phase <= 3'd0;
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
                                    infer_skip_init_clear <= 1'b0;
                                    infer_force_no_input  <= 1'b0;
                                    if (TRAIN_ENABLE && train_chunk_active &&
                                        ((train_chunk_state == TCK_INFER_WAIT) || (train_chunk_state == TCK_BLANK_INFER_WAIT))) begin
                                        // Sub-step completion for TRAIN_RUN_CHUNK phase2: do not emit host response here.
                                    end else begin
	                                    resp_status <= STATUS_OK;
	                                    resp_result <= infer_total_spikes;
	                                    resp_checksum <= calc_resp_checksum(STATUS_OK, infer_total_spikes);
	                                    response_ready <= 1'b1;
                                    end
	                            end else begin
	                                infer_step_idx <= infer_step_idx + 16'd1;
	                                infer_state <= INFER_GEN_INPUT_SPIKES;
	                            end
	                            infer_apply_idx <= 7'd0;
	                            infer_neuron_idx <= 7'd0;
	                            infer_input_idx <= 10'd0;
	                            infer_accum <= 32'sd0;
                                infer_accum_weight_phase <= 3'd0;
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
