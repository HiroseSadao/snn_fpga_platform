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

    localparam int CLKS_PER_BIT = 868; // 100_000_000 / 115_200 ~= 868

    localparam logic [7:0] REQ_SYNC   = 8'hA5;
    localparam logic [7:0] RESP_SYNC  = 8'h5A;
    localparam logic [7:0] PROTO_VER  = 8'h01;
    localparam logic [7:0] OP_SD_SECTORS_TO_DDR = 8'h13;
    localparam logic [7:0] OP_LOAD_IMAGE_FROM_DDR = 8'h14;
    localparam logic [7:0] OP_RUN_SAMPLE_INFER = 8'h20;
    localparam logic [7:0] OP_READ_SPIKE_COUNT = 8'h21;
    localparam logic [7:0] OP_TRAIN_QUERY_CAPS = 8'h30;
    localparam logic [7:0] OP_TRAIN_RUN_SAMPLE_PHASE3 = 8'h37;
    localparam logic [7:0] OP_TRAIN_RUN_SAMPLE_PHASE4 = 8'h38;
    localparam logic [7:0] OP_TRAIN_LABEL_STATS_RESET = 8'h39;
    localparam logic [7:0] OP_TRAIN_LABEL_STATS_ACCUM = 8'h3A;
    localparam logic [7:0] OP_READ_TRAIN_LABEL_STAT_SUM = 8'h3B;
    localparam logic [7:0] OP_READ_TRAIN_LABEL_STAT_COUNT = 8'h3C;
    localparam logic [7:0] OP_BATCH_CONFIG0 = 8'h40;
    localparam logic [7:0] OP_BATCH_CONFIG1 = 8'h41;
    localparam logic [7:0] OP_BATCH_START = 8'h42;
    localparam logic [7:0] OP_BATCH_STATUS = 8'h43;
    localparam logic [7:0] OP_BATCH_READ_SUMMARY = 8'h44;
    localparam logic [7:0] OP_BATCH_CONFIG2 = 8'h45;
    localparam logic [7:0] OP_BATCH_LABEL_WRITE = 8'h46;
    localparam logic [7:0] OP_BATCH_ASSIGN_WRITE = 8'h47;
    localparam logic [31:0] DDR_ADDR_WORD_LIMIT = 32'd16777216; // 64MiB / 4
    localparam logic [7:0] MAX_SUPPORTED_NARGS = 8'd2;
    // Increase RX timeout margin to tolerate host-side inter-byte gaps on UART.
    localparam int RX_TIMEOUT_CLKS = CLKS_PER_BIT * 2000;
    localparam int N_IN = 784;
    localparam int N_NEURONS = 50;
    localparam int N_WEIGHTS = N_IN * N_NEURONS;
    // Phase3 sparse connectivity target: keep only 30% of dense edges.
    localparam int N_EDGES = (N_WEIGHTS * 3) / 10;
    localparam int TRAIN_DENSE_ADDR_W = $clog2(N_WEIGHTS);
    localparam int W_ADDR_W = $clog2(N_EDGES);
    localparam int CSR_ROW_PTR_W = $clog2(N_EDGES + 1);
    localparam int EDGE_ADDR_W = $clog2(N_EDGES);
    localparam int COL_IDX_W = $clog2(N_IN);
    localparam int ROW_IDX_W = $clog2(N_NEURONS);
    localparam int TRAIN_MINE_NT_BLANK = 150;
    localparam logic signed [31:0] FXP_ALPHA = 32'sd62259; // legacy/simple model coeff (unused in mine-style LIF)
    localparam logic signed [31:0] FXP_ALPHA_INH = 32'sd58982; // legacy/simple model coeff (unused in mine-style LIF)
    localparam logic signed [31:0] FXP_INPUT_W = 32'sd8192; // 0.125 in S16.16
    localparam logic signed [31:0] FXP_THRESH = 32'sd65536; // 1.0 in S16.16
    localparam logic signed [31:0] FXP_BIAS_LSB = 32'sd512; // 0.0078125 in S16.16
    localparam logic signed [31:0] FXP_ONE = 32'sd65536; // 1.0 in S16.16
    localparam logic signed [31:0] FXP_HALF = 32'sd32768; // 0.5 in S16.16
    localparam logic signed [31:0] FXP_WEXC = 32'sd147456; // 2.25 in S16.16
    localparam logic signed [31:0] FXP_INH_COEFF = 32'sd563; // (0.85/99) in S16.16, aligned to mine.py main()
    localparam logic signed [31:0] FXP_INH_THRESH = -32'sd2621440; // -40.0 in S16.16
    localparam logic signed [31:0] FXP_SCALE_1000 = 32'sd65536000;   // 1000.0 in S16.16 (1/1ms)
    localparam logic signed [31:0] FXP_SCALE_500  = 32'sd32768000;   // 500.0 in S16.16 (1/2ms)
    localparam logic signed [31:0] FXP_TRACE_PRE_DECAY = 32'sd62339;   // exp(-1/20) in S16.16
    localparam logic signed [31:0] FXP_TRACE_POST1_DECAY = 32'sd62339; // exp(-1/20) in S16.16
    localparam logic signed [31:0] FXP_TRACE_POST2_DECAY = 32'sd63917; // exp(-1/40) in S16.16
    localparam logic signed [31:0] FXP_TRACE_EVENT_SET = 32'sd65536;   // 1.0 in S16.16
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
    localparam logic [31:0] POISSON_NUM_CONST = 32'd2048; // Brian2: threshold_num = raw_u8 * 2048
    localparam logic [31:0] POISSON_DEN_CONST = 32'd4000; // Brian2: p = (raw_u8 / 8) * 2e-3 = raw_u8 / 4000
    // Poisson threshold upper bound. 2048 means always-fire against rand11 in [0..2047].
    localparam logic [11:0] RNG_MAX = 12'd2048;
    localparam logic [31:0] LCG_A = 32'd1664525;
    localparam logic [31:0] LCG_C = 32'd1013904223;
    // Build switch: keep training kernels enabled for mine.py-aligned learning builds.
    // Set to 1'b0 only for inference-only fast-build iteration.
    localparam logic TRAIN_ENABLE = 1'b1;
    localparam logic [7:0] STATUS_OK             = 8'h00;
    localparam logic [7:0] STATUS_BAD_PACKET     = 8'hE1;
    localparam logic [7:0] STATUS_UNSUPPORTED_OP = 8'hE2;
    localparam logic [7:0] BADDBG_SD_REQ_ARG     = 8'h20;
    localparam logic [7:0] BADDBG_SD_CD_N        = 8'h21;
    localparam logic [7:0] BADDBG_SD_WAIT_TO     = 8'h22;
    localparam logic [31:0] IMGLOAD_DDR_WAIT_TIMEOUT_CLKS = 32'd25000000; // 250ms @100MHz
    // Training kernel capability bits (host-visible via OP_TRAIN_QUERY_CAPS)
    // [0]=query_caps impl, [1]=logical DDR map fixed, [2]=trace opcode present,
    // [3]=tile opcode present, [8]=trace kernel exec impl, [9]=tile kernel exec impl,
    // [10]=train work generation helper impl, [11]=stdp all-rows batch impl,
    // [12]=train chunk runner (phase0 skeleton) impl, [13]=phase3 sample flow impl,
    // [14]=phase4 retry sample flow impl.
    localparam logic [31:0] TRAIN_CAPS_VALUE = 32'h00017F0F;
    localparam logic [7:0] BATCH_ERR_NONE = 8'h00;
    localparam logic [7:0] BATCH_ERR_NOT_READY = 8'h01;
    localparam logic [7:0] BATCH_ERR_UNIMPLEMENTED = 8'h02;
    localparam logic [7:0] BATCH_PHASE_IDLE = 8'd0;
    localparam logic [7:0] BATCH_PHASE_CONFIGURED = 8'd1;
    localparam logic [7:0] BATCH_PHASE_LOADING = 8'd2;
    localparam logic [7:0] BATCH_PHASE_RUNNING = 8'd3;
    localparam logic [7:0] BATCH_PHASE_DONE = 8'd4;
    // Step1 logical DDR word map contract (future external DDR integration target).
    localparam logic [31:0] TRAIN_BASE_W_Q16_WORDS  = 32'd0;
    localparam logic [31:0] TRAIN_BASE_A_Q16_WORDS  = TRAIN_BASE_W_Q16_WORDS + N_WEIGHTS;
    localparam logic [31:0] TRAIN_BASE_BT_Q16_WORDS = TRAIN_BASE_A_Q16_WORDS + N_WEIGHTS;
    localparam logic [31:0] TRAIN_BASE_THETA_WORDS  = TRAIN_BASE_BT_Q16_WORDS + N_WEIGHTS;
    localparam logic [31:0] TRAIN_BASE_VSTATE_WORDS = TRAIN_BASE_THETA_WORDS + N_NEURONS;
    localparam logic [31:0] TRAIN_BASE_DELAY_WORDS  = TRAIN_BASE_VSTATE_WORDS + N_NEURONS;
    localparam logic [31:0] TRAIN_BASE_GIN_WORDS    = TRAIN_BASE_DELAY_WORDS + (N_NEURONS * 8);
    // Fixed training kernel workspaces (host preloads before training kernels)
    localparam logic [31:0] TRAIN_BASE_XIN_WORK_WORDS    = TRAIN_BASE_GIN_WORDS + N_NEURONS;
    localparam logic [31:0] TRAIN_BASE_XEXC_WORK_WORDS   = TRAIN_BASE_XIN_WORK_WORDS + N_IN;
    localparam logic [31:0] TRAIN_BASE_PRELIST_WORK_WORDS= TRAIN_BASE_XEXC_WORK_WORDS + N_NEURONS;
    // Dedicated SD image staging base in DDR (must not overlap training logical map).
    localparam logic [31:0] IMG_STAGING_BASE_WORD = TRAIN_BASE_PRELIST_WORK_WORDS + 32'd262144; // +1MiB bytes margin
    // Dedicated batch-image cache region in DDR. Batch train/infer preloads the requested range here once.
    localparam logic [31:0] IMG_CACHE_BASE_WORD   = IMG_STAGING_BASE_WORD + 32'd262144; // +1MiB bytes margin
    // STDP tile kernel (phase1) fixed-point params, q16.16.
    localparam logic signed [31:0] TRAIN_WMAX_Q16    = 32'sd65535; // ~1.0, max representable with stored UQ0.16 weights
    localparam logic signed [31:0] TRAIN_WMIN_Q16    = 32'sd0;
    localparam logic signed [31:0] TRAIN_NORM_Q16    = 32'sd6554; // 0.1
    localparam logic signed [31:0] TRAIN_LR_P_Q16    = 32'sd655;  // 1e-2
    localparam logic signed [31:0] TRAIN_LR_M_Q16    = 32'sd7;    // 1e-4
    localparam logic signed [31:0] TRAIN_CLIP_DW_Q16 = 32'sd66;   // 1e-3
    localparam logic [31:0]        TRAIN_UPDATE_NT   = 32'd100;
    localparam logic [31:0]        TRAIN_RETRY_MIN_INJ_SPIKES = 32'd5;
    localparam logic [31:0]        TRAIN_RETRY_MAX_FR_START   = 32'd32;
    localparam logic [31:0]        TRAIN_RETRY_MAX_FR_STEP    = 32'd16;
    localparam logic [31:0] RAW1_HEADER_BYTES = 32'd20;
    localparam logic [31:0] RAW1_NUM_IMAGES = 32'd10000;
    localparam logic [31:0] RAW1_BYTES_PER_IMAGE = 32'd784;

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
        MEMRD_TRAIN_LABEL_STAT_SUM,
        MEMRD_TRAIN_LABEL_STAT_COUNT
    } memrd_kind_t;
    typedef enum logic [6:0] {
        INFER_IDLE,
        INFER_INIT_CLEAR,
        INFER_CLEAR_SPIKE_COUNT,
        INFER_PREP_DIV_START,
        INFER_PREP_DIV_MUL,
        INFER_PREP_DIV_WAIT,
        INFER_GEN_INPUT_SPIKES,
        INFER_ACCUM_NEURON,
        INFER_ACCUM_NEURON_GIN_MUL,
        INFER_ACCUM_NEURON_GIN_MUL_ROUND,
        INFER_ACCUM_NEURON_GIN_COMB,
        INFER_ACCUM_NEURON_PIPE,
        INFER_NEURON_DV_PRE,
        INFER_NEURON_DV_DRIVE,
        INFER_NEURON_DV_SYN,
        INFER_NEURON_DV_SYN_ROUND,
        INFER_NEURON_VNEXT,
        INFER_NEURON_SPIKE,
        INFER_NEURON_THETA_PRE,
        INFER_NEURON_THETA_ROUND,
        INFER_NEURON_COMMIT,
        INFER_NEURON_WRITE,
        INFER_APPLY_WTA,
        INFER_APPLY_WTA_PRE,
        INFER_APPLY_WTA_PRE_MUL,
        INFER_APPLY_WTA_PRE_ROUND,
        INFER_APPLY_WTA_INH,
        INFER_APPLY_WTA_INH_PROD,
        INFER_APPLY_WTA_INH_MUL,
        INFER_APPLY_WTA_INH_DV,
        INFER_APPLY_WTA_INH_VPROP,
        INFER_APPLY_WTA_INH_POST,
        INFER_APPLY_WTA_ACCUM,
        INFER_WTA_PASS2_PRE,
        INFER_WTA_PASS2,
        INFER_WTA_PASS2_WRITE,
        INFER_EVT_PRE_PRELIST_REQ,
        INFER_EVT_PRE_PRELIST_WAIT,
        INFER_EVT_PRE_PTR0_REQ,
        INFER_EVT_PRE_PTR0_WAIT,
        INFER_EVT_PRE_PTR1_REQ,
        INFER_EVT_PRE_PTR1_WAIT,
        INFER_EVT_PRE_EDGE_REQ,
        INFER_EVT_PRE_EDGE_WAIT,
        INFER_EVT_PRE_TRACE_REQ,
        INFER_EVT_PRE_TRACE_WAIT,
        INFER_EVT_PRE_W_WAIT,
        INFER_EVT_PRE_APPLY,
        INFER_EVT_PRE_APPLY_MUL1,
        INFER_EVT_PRE_APPLY_MUL2,
        INFER_EVT_PRE_APPLY_CLIP,
        INFER_EVT_PRE_APPLY_WNEXT,
        INFER_EVT_POST_PTR0_REQ,
        INFER_EVT_POST_PTR0_WAIT,
        INFER_EVT_POST_PTR1_REQ,
        INFER_EVT_POST_PTR1_WAIT,
        INFER_EVT_POST_EDGE_REQ,
        INFER_EVT_POST_EDGE_WAIT,
        INFER_EVT_POST_TRACE_REQ,
        INFER_EVT_POST_TRACE_WAIT,
        INFER_EVT_POST_W_WAIT,
        INFER_EVT_POST_APPLY,
        INFER_EVT_POST_APPLY_MUL1,
        INFER_EVT_POST_APPLY_MUL2,
        INFER_EVT_POST_APPLY_CLIP,
        INFER_EVT_POST_APPLY_WNEXT,
        INFER_EVT_DONE
    } infer_state_t;
    typedef enum logic [2:0] {
        DDRBR_IDLE,
        DDRBR_ISSUE,
        DDRBR_WAIT_ACK,
        DDRBR_READ_CAPTURE,
        DDRBR_RESP
    } ddr_bridge_state_t;
    typedef enum logic [2:0] {
        DDR_REQ_NONE,
        DDR_REQ_SD,
        DDR_REQ_IMGLOAD,
        DDR_REQ_TRAIN,
        DDR_REQ_GENERIC
    } ddr_req_kind_t;
    typedef enum logic [3:0] {
        TMI_IDLE,
        TMI_W_READ_REQ,
        TMI_W_READ_WAIT,
        TMI_W_WRITE_REQ,
        TMI_W_WRITE_WAIT,
        TMI_A_WRITE_REQ,
        TMI_A_WRITE_WAIT,
        TMI_BT_WRITE_REQ,
        TMI_BT_WRITE_WAIT,
        TMI_DONE
    } train_mem_init_state_t;
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
        TRK_B_READ_PRE_BRAM_WAIT,
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
        TSK_DIV_DW_PREP,
        TSK_DIV_DW_PREP_MUL,
        TSK_DIV_DW_PREP_ROUND,
        TSK_DIV_DW_PIPE,
        TSK_DIV_DW_TERM,
        TSK_DIV_DW_TERM_ROUND,
        TSK_DIV_DW_COMB,
        TSK_DIV_DW_ABS,
        TSK_DIV_DW_START,
        TSK_DIV_DW_WAIT,
        TSK_DIV_DW_CLIP,
        TSK_DIV_DW_WNEXT,
        TSK_WRITE_W_REQ,
        TSK_WRITE_W_WAIT,
        TSK_CLR_A_REQ,
        TSK_CLR_A_WAIT,
        TSK_CLR_BT_REQ,
        TSK_CLR_BT_WAIT,
        TSK_DONE
    } train_stdp_state_t;
    typedef enum logic [2:0] {
        TGK_IDLE,
        TGK_READ_WAIT,
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
        TLS_ACCUM_SAMPLE,
        TLS_ACCUM_WRITE,
        TLS_DONE
    } train_label_stats_state_t;
    typedef enum logic [3:0] {
        BIE_IDLE,
        BIE_RESET,
        BIE_READ,
        BIE_WAIT,
        BIE_ACCUM,
        BIE_SELECT_PREP,
        BIE_SELECT_MUL,
        BIE_SELECT,
        BIE_DONE
    } batch_infer_eval_state_t;
    typedef enum logic [4:0] {
        TCK_IDLE,
        TCK_INFER_START,
        TCK_INFER_WAIT,
        TCK_SNAP_COPY_INIT,
        TCK_SNAP_COPY_WAIT,
        TCK_SNAP_COPY_WRITE,
        TCK_BLANK_INFER_START,
        TCK_BLANK_INFER_WAIT,
        TCK_REBASE_GIN_INIT,
        TCK_REBASE_GIN_RUN,
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
    logic [31:0] batch_status_word;
    logic [31:0] batch_summary_word;
    logic        train_busy_uart_blocked;
    logic [20:0] rx_timeout_counter;
    logic [1:0]  clk_div;
    wire         clk_25mhz = clk_div[1];
    wire         core_clk = clk_100mhz_buf;
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
    logic        ddr_calib_complete_core;

    // Core<->DDR bridge for UART DDR read/write smoke test
    logic        ddr_req_pending_core;
    logic        ddr_req_we_core;
    logic [31:0] ddr_req_addr_word_core;
    logic [31:0] ddr_req_wdata_core;
    logic        ddr_req_wide_core;
    logic [127:0] ddr_req_wdata128_core;
    logic [15:0] ddr_req_sel16_core;
    logic [2:0]  ddr_req_word_count_core;
    logic        ddr_req_toggle_core;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) logic ddr_rsp_toggle_core_sync1, ddr_rsp_toggle_core_sync2;
    logic        ddr_rsp_toggle_core_seen;
    logic        ddr_rsp_capture_pending_core;
    logic        ddr_rsp_payload_ready_core;
    logic [1:0]  ddr_rsp_payload_settle_core;
    logic        ddr_rsp_drain_active_core;
    logic [1:0]  ddr_rsp_drain_quiet_core;
    logic [31:0] ddr_resp_rdata_core;
    logic [7:0]  ddr_resp_status_core;
    logic [72:0]  ddr_req_payload_core;
    logic [72:0]  ddr_req_payload_core_reg;
    logic [72:0]  ddr_req_payload_ddr_sync;
    logic [49:0]  ddr_rsp_payload_ddr;
    logic [49:0]  ddr_rsp_payload_ddr_reg;
    logic [49:0]  ddr_rsp_payload_core_sync;
    logic [15:0]  ddr_req_tag_ddr;
    logic [15:0]  ddr_rsp_req_tag_ddr;
    logic        ddr_rsp_was_write_ddr;
    logic        ddr_resp_was_write_core;
    logic [15:0]  ddr_req_tag_core;
    logic [15:0]  ddr_req_tag_expect_core;
    logic        ddr_req_pending_core_prev;

    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) logic ddr_req_toggle_ddr_sync1, ddr_req_toggle_ddr_sync2;
    logic        ddr_req_toggle_ddr_seen;
    logic        ddr_rsp_toggle_ddr;
    ddr_bridge_state_t ddr_bridge_state;
    logic        ddr_req_we_ddr;
    logic [31:0] ddr_req_addr_word_ddr;
    logic [31:0] ddr_req_wdata_ddr;
    logic [31:0] ddr_rsp_rdata_ddr;
    logic [7:0]  ddr_rsp_status_ddr;
    ddr_req_kind_t ddr_req_kind_core;
    ddr_req_kind_t ddr_rsp_kind_core;
    logic        ddr_req_from_sd_core;
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
    logic        raw_image0_valid;
    logic        raw_image1_valid;
    logic [31:0] raw_num_images;
    logic [31:0] raw_bytes_per_image;
    logic [9:0]  raw_image0_capture_idx;
    logic [9:0]  raw_image1_capture_idx;
    logic [31:0] raw_image0_sum_u8;
    logic [31:0] raw_image1_sum_u8;
    logic [9:0]  raw_image0_rd_addr;
    logic [7:0]  raw_image0_rd_data;
    logic [7:0]  raw_image1_rd_data;
    logic        raw_image0_wr_en;
    logic [9:0]  raw_image0_wr_addr;
    logic [7:0]  raw_image0_wr_data;
    (* ram_style = "block" *) logic [7:0] raw_image0_mem [0:N_IN-1];
    (* ram_style = "block" *) logic [7:0] raw_image1_mem [0:N_IN-1];
    logic        batch_compute_buf_sel;
    logic        batch_fill_buf_sel;
    logic        imgload_target_buf_sel;
    logic        batch_prefetch_active;
    logic        batch_prefetch_issue_pending;
    logic        batch_prefetch_ready;
    logic [31:0] batch_prefetch_sample_idx;
    logic [31:0] batch_prefetch_img_byte_off;
    logic [31:0] batch_prefetch_img_sector_off;
    logic [31:0] batch_prefetch_img_byte_in_sector;
    logic [31:0] batch_prefetch_img_sectors_needed;
    logic        sd_copy_resp_pending;
    logic [7:0]  sd_copy_resp_status;
    logic signed [31:0] sd_copy_resp_result;
    logic        raw_image_compute_valid;
    logic [31:0] raw_image_compute_sum_u8;
    logic [7:0]  raw_image_compute_rd_data;
    logic        raw_image_fill_valid;
    (* ram_style = "block" *) logic [7:0] batch_label_mem [0:RAW1_NUM_IMAGES-1];
    logic        imgload_active;
    logic [31:0] imgload_addr_word;
    logic [1:0]  imgload_lane;
    logic [9:0]  imgload_byte_idx;
    logic [9:0]  imgload_total_bytes;
    logic [31:0] imgload_sum_u8_accum;
    logic        imgload_word_valid;
    logic [31:0] imgload_word_data;
    logic [1:0]  imgload_word_lane;
    logic [31:0] imgload_ddr_wait_counter;
    logic        imgload_start_pending;
    logic [31:0] imgload_start_addr_word;
    logic [1:0]  imgload_start_lane;
    logic [9:0]  imgload_start_total_bytes;
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
    logic        train_trace_use_infer_prelist;
    logic        train_trace_skip_a;
    logic        train_trace_skip_b;
    logic        train_trace_multi_post_active;
    logic [6:0]  train_trace_post_scan_idx;
    logic        infer_trace_next_post_found;
    logic [6:0]  infer_trace_next_post_idx;
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
    logic [31:0] train_stdp_div_q_holdfix;
    logic signed [31:0] train_stdp_dW_q16;
    logic [31:0] train_stdp_dW_abs;
    logic signed [31:0] train_stdp_pot_term_q16;
    logic signed [31:0] train_stdp_dep_term_q16;
    logic signed [31:0] train_stdp_pot_mid_q16;
    logic signed [31:0] train_stdp_dep_mid_q16;
    logic signed [31:0] train_stdp_pot_mid_pipe_q16;
    logic signed [31:0] train_stdp_dep_mid_pipe_q16;
    logic signed [31:0] train_stdp_a_val_pipe;
    logic signed [31:0] train_stdp_bt_val_pipe;
    (* use_dsp = "yes" *) logic signed [63:0] train_stdp_pot_prod_q32;
    (* use_dsp = "yes" *) logic signed [63:0] train_stdp_dep_prod_q32;
    (* use_dsp = "yes" *) logic signed [63:0] train_stdp_pot_mid_prod_q32;
    (* use_dsp = "yes" *) logic signed [63:0] train_stdp_dep_mid_prod_q32;
    logic [31:0] train_stdp_row_sum_abs;
    logic [31:0] train_stdp_w_row_base;
    logic [31:0] train_stdp_a_row_base;
    logic [31:0] train_stdp_bt_col_base;
    logic [31:0] train_stdp_dividend;
    logic [31:0] train_stdp_divisor;
    logic [31:0] train_stdp_update_nt;
    logic        train_stdp_div_valid;
    logic [31:0] train_stdp_div_q;
    logic [31:0] train_stdp_div_r;
    logic        train_stdp_div_out_valid;
    logic        train_stdp_div_err;
    logic        train_stdp_div_busy;
    logic        train_stdp_batch_active;
    logic        train_chunk_active;
    train_chunk_state_t train_chunk_state;
    logic [2:0]  train_chunk_mode; // phase3 runtime mode (fixed to 3 while active)
    logic [15:0] train_chunk_samples_left;
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
    logic        train_chunk_winner_scan_phase;
    logic [6:0]  train_chunk_snap_copy_idx;
    logic [31:0] train_chunk_last_infer_spikes;
    logic [31:0] train_chunk_last_blank_spikes;
    logic [31:0] train_chunk_retry_curr_max_fr;
    logic [31:0] train_chunk_retry_accepted_max_fr;
    logic        train_chunk_retry_continue_infer;
    logic [6:0]  train_rebase_neuron_idx;
    logic [9:0]  train_rebase_input_idx;
    logic [W_ADDR_W-1:0] train_rebase_edge_idx;
    logic [W_ADDR_W-1:0] train_rebase_edge_end;
    logic signed [31:0] train_rebase_accum;
    logic [2:0]  train_rebase_phase;
    logic        train_gen_active;
    train_gen_state_t train_gen_state;
    logic [31:0] train_gen_base_word;
    logic [15:0] train_gen_count_total;
    logic [15:0] train_gen_idx;
    logic [31:0] train_gen_lcg_state;
    logic [31:0] train_gen_curr_word;
    logic        train_gen_lcg_enable;
    logic [1:0]  train_gen_cache_mode; // 0=none,1=x_in,2=x_exc
    logic        train_mem_init_active;
    logic        train_mem_init_done;
    train_mem_init_state_t train_mem_init_state;
    logic [TRAIN_DENSE_ADDR_W-1:0] train_mem_init_idx;
    logic        train_label_stats_active;
    train_label_stats_state_t train_label_stats_state;
    logic [3:0]  train_label_stats_label;
    logic [9:0]  train_label_stats_idx;
    logic [9:0]  train_label_stats_base_idx;
    logic        train_xin_cache_valid;
    logic        train_xexc_cache_valid;
    logic        train_xin_wr_en;
    logic [9:0]  train_xin_wr_addr;
    logic [31:0] train_xin_wr_data;
    logic        train_xin_wr_en_pipe;
    logic [9:0]  train_xin_wr_addr_pipe;
    logic [31:0] train_xin_wr_data_pipe;
    logic [9:0]  train_xin_rd_addr;
    logic [31:0] train_xin_rd_data;
    (* ram_style = "block" *) logic [31:0] train_xin_mem [0:N_IN-1];
    logic        train_xexc_wr_en;
    logic [6:0]  train_xexc_wr_addr;
    logic [31:0] train_xexc_wr_data;
    logic [6:0]  train_xexc_rd_addr;
    logic [31:0] train_xexc_rd_data;
    (* ram_style = "block" *) logic [31:0] train_xexc_mem [0:N_NEURONS-1];
    logic        train_xpost2_wr_en;
    logic [6:0]  train_xpost2_wr_addr;
    logic [31:0] train_xpost2_wr_data;
    logic [6:0]  train_xpost2_rd_addr;
    logic [31:0] train_xpost2_rd_data;
    (* ram_style = "block" *) logic [31:0] train_xpost2_mem [0:N_NEURONS-1];
    (* ram_style = "block" *) logic [31:0] train_label_count [0:9];
    logic [9:0]  train_label_sum_rd_addr;
    logic [31:0] train_label_sum_rd_data;
    logic        train_label_sum_wr_en;
    logic [9:0]  train_label_sum_wr_addr;
    logic [31:0] train_label_sum_wr_data;
    (* ram_style = "block" *) logic [31:0] train_label_sum_mem [0:(10*N_NEURONS)-1];
    logic [3:0]  train_label_count_rd_addr;
    logic [31:0] train_label_count_rd_data;
    logic [15:0] train_label_stats_spike_q;
    logic        batch_cfg0_valid;
    logic        batch_cfg1_valid;
    logic        batch_cfg_mode_train;
    logic [31:0] batch_cfg_start_sample_idx;
    logic [31:0] batch_cfg_num_samples;
    logic [31:0] batch_cfg_seed;
    logic [31:0] batch_cfg_start_lba;
    logic [31:0] batch_cfg_start_byte_off;
    logic [31:0] batch_cfg_start_sector_off;
    logic [31:0] batch_cfg_start_byte_in_sector;
    logic [31:0] batch_cfg_start_sectors_needed;
    logic [31:0] batch_cfg_num_samples_bytes;
    logic [31:0] batch_cfg_cache_total_sectors;
    logic [31:0] batch_cfg_cache_total_words;
    logic        batch_cfg_cache_fits;
    logic        batch_active;
    logic        batch_done;
    logic        batch_error;
    logic [7:0]  batch_error_code;
    logic [7:0]  batch_phase;
    logic [31:0] batch_processed_samples;
    logic [31:0] batch_current_sample_idx;
    logic [31:0] batch_total_spikes;
    logic [31:0] batch_correct_count;
    logic [63:0] batch_elapsed_cycles;
    logic [63:0] batch_load_cycles;
    logic [63:0] batch_train_core_cycles;
    logic [63:0] batch_infer_core_cycles;
    logic [63:0] batch_label_stats_cycles;
    logic [63:0] batch_infer_eval_cycles;
    logic [63:0] batch_other_cycles;
    logic [63:0] batch_train_inject_infer_cycles;
    logic [63:0] batch_train_blank_infer_cycles;
    logic [63:0] batch_train_snap_cycles;
    logic [63:0] batch_train_rebase_cycles;
    logic [63:0] batch_train_evt_pre_cycles;
    logic [63:0] batch_train_evt_post_cycles;
    logic [63:0] batch_train_accum_cycles;
    logic [31:0] batch_img_byte_off;
    logic [31:0] batch_img_sector_off;
    logic [31:0] batch_img_byte_in_sector;
    logic [31:0] batch_img_sectors_needed;
    logic        batch_use_cached_images;
    logic [31:0] batch_cache_img_byte_off;
    logic        batch_label_wr_en;
    logic [13:0] batch_label_wr_addr;
    logic [7:0]  batch_label_wr_data;
    logic [13:0] batch_label_rd_addr;
    logic [7:0]  batch_label_rd_data;
    logic        batch_assign_wr_en;
    logic [6:0]  batch_assign_wr_addr;
    logic [3:0]  batch_assign_wr_data;
    logic [6:0]  batch_assign_rd_addr;
    logic [3:0]  batch_assign_rd_data;
    (* ram_style = "block" *) logic [3:0] batch_assign_mem [0:N_NEURONS-1];
    logic        batch_infer_eval_active;
    batch_infer_eval_state_t batch_infer_eval_state;
    logic [6:0]  batch_infer_eval_idx;
    logic [3:0]  batch_infer_eval_label_idx;
    logic [15:0] batch_infer_eval_spike_q;
    logic [3:0]  batch_infer_eval_assign_q;
    logic [3:0]  batch_infer_eval_pred_label;
    logic [3:0]  batch_infer_eval_best_label;
    logic [31:0] batch_infer_eval_best_sum;
    logic [15:0] batch_infer_eval_best_count;
    logic [31:0] batch_infer_eval_curr_sum_q;
    logic [15:0] batch_infer_eval_curr_count_q;
    logic [15:0] batch_infer_eval_best_count_eff_q;
    (* use_dsp = "yes" *) logic [63:0] batch_infer_eval_cmp_lhs;
    (* use_dsp = "yes" *) logic [63:0] batch_infer_eval_cmp_rhs;
    logic [31:0] batch_infer_eval_sum [0:9];
    logic [15:0] batch_infer_eval_count [0:9];
    integer batch_eval_i;

    logic        infer_active;
    infer_state_t infer_state;
    logic [31:0] infer_steps_target;
    logic [15:0] infer_step_idx;
    logic [6:0]  infer_neuron_idx;
    logic [9:0]  infer_input_idx;
    logic [9:0]  infer_prep_idx;
    logic signed [31:0] infer_accum;
    logic [2:0]  infer_accum_weight_phase;
    logic [W_ADDR_W-1:0] infer_w_rd_addr;
    logic [15:0] infer_w_rd_data;
    logic [15:0] infer_w_rd_data_q;
    logic [W_ADDR_W-1:0] infer_w_rd_addr_lane1;
    logic [15:0] infer_w_rd_data_lane1;
    logic [15:0] infer_w_rd_data_q_lane1;
    logic        infer_w_wr_en;
    logic [W_ADDR_W-1:0] infer_w_wr_addr;
    logic [15:0] infer_w_wr_data;
    (* ram_style = "block" *) logic signed [31:0] infer_g_in_state [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_v_state [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_exc_theta [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_in_delay0 [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_in_delay1 [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_in_delay2 [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_in_delay3 [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_in_delay4 [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_v_inh_state [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_c_exc_state [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_c_inh_state [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_inh_state [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_exc_delay0 [0:N_NEURONS-1];
    (* ram_style = "block" *) logic signed [31:0] infer_g_exc_delay1 [0:N_NEURONS-1];
    logic        infer_s_exc [0:N_NEURONS-1];
    localparam int SPIKE_CNT_ADDR_W = 7;
    logic        spike_count_we;
    logic [SPIKE_CNT_ADDR_W-1:0] spike_count_waddr;
    logic [15:0] spike_count_wdata;
    logic [15:0] spike_count_wdata_holdfix;
    logic [6:0]  infer_spike_rd_addr;
    logic [15:0] infer_spike_rd_data;
    logic [15:0] spike_count_rdata;
    logic        snap_count_we;
    logic [SPIKE_CNT_ADDR_W-1:0] snap_count_waddr;
    logic [15:0] snap_count_wdata;
    logic [SPIKE_CNT_ADDR_W-1:0] snap_count_raddr;
    logic [15:0] snap_count_rdata;
    logic [ROW_IDX_W:0] csr_row_ptr_rd_addr;
    logic [CSR_ROW_PTR_W-1:0] csr_row_ptr_rd_data;
    logic [EDGE_ADDR_W-1:0] csr_col_idx_rd_addr;
    logic [COL_IDX_W-1:0] csr_col_idx_rd_data;
    logic [EDGE_ADDR_W-1:0] csr_col_idx_rd_addr_lane1;
    logic [COL_IDX_W-1:0] csr_col_idx_rd_data_lane1;
    logic [COL_IDX_W:0] csc_col_ptr_rd_addr;
    logic [CSR_ROW_PTR_W-1:0] csc_col_ptr_rd_data;
    logic [EDGE_ADDR_W-1:0] csc_row_idx_rd_addr;
    logic [ROW_IDX_W-1:0] csc_row_idx_rd_data;
    logic [EDGE_ADDR_W-1:0] csc_edge_idx_rd_addr;
    logic [EDGE_ADDR_W-1:0] csc_edge_idx_rd_data;
    (* ram_style = "block" *) logic [15:0] infer_exc_last_spike_step [0:N_NEURONS-1];
    (* ram_style = "block" *) logic [15:0] infer_inh_last_spike_step [0:N_NEURONS-1];
    logic [6:0]  infer_apply_idx;
    logic [2:0]  infer_trace_phase;
    logic        infer_trace_spike_latched;
    (* use_dsp = "yes" *) logic signed [63:0] infer_apply_xin_prod;
    logic signed [31:0] infer_apply_xin_decay;
    logic signed [31:0] infer_apply_xin_next;
    logic signed [31:0] infer_apply_xexc_trace_q;
    logic signed [31:0] infer_apply_xpost2_trace_q;
    (* use_dsp = "yes" *) logic signed [63:0] infer_apply_xexc_prod;
    (* use_dsp = "yes" *) logic signed [63:0] infer_apply_xpost2_prod;
    logic signed [31:0] infer_apply_xexc_decay;
    logic signed [31:0] infer_apply_xexc_next;
    logic signed [31:0] infer_apply_xpost2_decay;
    logic signed [31:0] infer_apply_xpost2_next;
    logic signed [31:0] infer_post1_before [0:N_NEURONS-1];
    logic signed [31:0] infer_post2_before [0:N_NEURONS-1];
    logic signed [31:0] infer_sum_c_inh;
    logic signed [31:0] infer_pass2_diff_c_inh;
    logic signed [31:0] infer_pass2_g_inh_next;
    logic signed [31:0] infer_apply_c_inh_next;
    logic signed [31:0] infer_apply_v_inh_write;
    logic signed [31:0] infer_apply_v_inh_cur;
    logic signed [31:0] infer_apply_c_inh_cur;
    logic [15:0] infer_apply_inh_last_spike;
    logic signed [31:0] infer_apply_delayed_g_exc;
    logic signed [31:0] infer_apply_inh_eexc_minus_v;
    logic signed [31:0] infer_apply_inh_vrest_minus_v;
    logic signed [31:0] infer_apply_exc_drive_dt_inh;
    logic signed [31:0] infer_apply_leak_dt_inh;
    (* use_dsp = "yes" *) logic signed [63:0] infer_apply_exc_drive_prod_q32;
    (* use_dsp = "yes" *) logic signed [63:0] infer_apply_leak_prod_q32;
    logic signed [31:0] infer_apply_i_syn_exc_mul_a;
    logic signed [31:0] infer_apply_i_syn_exc_mul_b;
    logic signed [31:0] infer_apply_i_syn_exc_step_inh;
    (* use_dsp = "yes" *) logic signed [63:0] infer_apply_i_syn_exc_prod_q32;
    logic signed [31:0] infer_apply_dv_inh_step;
    logic signed [31:0] infer_apply_v_inh_prop;
    logic        infer_apply_inh_refractory_ok;
    logic        infer_apply_s_inh_now;
    logic        infer_step_winner_valid;
    logic [6:0]  infer_step_winner_idx;
    logic        infer_trace_wait_last_step;
    logic [9:0]  infer_evt_prelist_idx;
    logic [9:0]  infer_evt_pre_idx;
    logic [6:0]  infer_evt_post_idx;
    logic [9:0]  infer_evt_post_input_idx;
    logic        infer_evt_has_winner;
    logic [6:0]  infer_evt_winner_idx;
    logic signed [31:0] infer_evt_trace_val;
    logic signed [31:0] infer_evt_post2_before_q;
    logic signed [31:0] infer_evt_w_cur;
    logic signed [31:0] infer_evt_mid_q16;
    (* use_dsp = "yes" *) logic signed [63:0] infer_evt_term_prod_q32;
    logic signed [31:0] infer_evt_term_q16;
    logic signed [31:0] infer_evt_dw_q16;
    logic signed [31:0] infer_evt_w_next_q16;
    logic [EDGE_ADDR_W-1:0] infer_evt_edge_idx;
    logic [EDGE_ADDR_W-1:0] infer_evt_edge_end;
    logic [EDGE_ADDR_W-1:0] infer_evt_edge_ptr;
    logic [1:0]  infer_accum_pair_count;
    logic        infer_accum_lane0_fire;
    logic        infer_accum_lane1_fire;
    logic [31:0] infer_total_spikes;
    logic [31:0] infer_rng_state;
    (* use_dsp = "yes" *) logic [63:0] infer_rng_mul_prod_q32;
    logic [31:0] infer_poisson_num_const_cfg;
    logic [9:0]  infer_poisson_thresh_rd_addr;
    logic [11:0] infer_poisson_thresh_rd_data;
    logic        infer_poisson_thresh_wr_en;
    logic [9:0]  infer_poisson_thresh_wr_addr;
    logic [11:0] infer_poisson_thresh_wr_data;
    (* ram_style = "block" *) logic [11:0] infer_poisson_thresh_mem [0:N_IN-1];
    logic        infer_pre_wr_en;
    logic [9:0]  infer_pre_wr_addr;
    logic [9:0]  infer_pre_wr_data;
    logic [9:0]  infer_pre_active_count;
    logic [9:0]  infer_pre_rd_addr;
    logic [9:0]  infer_pre_rd_data;
    logic        infer_pre_spike_wr_en;
    logic [9:0]  infer_pre_spike_wr_addr;
    logic        infer_pre_spike_wr_data;
    logic [9:0]  infer_pre_spike_rd_addr;
    logic        infer_pre_spike_rd_data;
    logic [9:0]  infer_pre_spike_rd_addr_lane1;
    logic        infer_pre_spike_rd_data_lane1;
    logic [31:0] infer_dividend;
    logic [31:0] infer_divisor;
    (* use_dsp = "yes" *) logic [63:0] infer_prep_div_prod_q32;
    logic        infer_div_valid;
    logic [31:0] infer_div_q;
    logic [31:0] infer_div_r;
    logic        infer_div_out_valid;
    logic        infer_div_err;
    logic        infer_div_busy;
    logic [6:0]  infer_eval_idx;
    logic signed [31:0] infer_eval_v_cur;
    logic signed [31:0] infer_eval_theta_cur;
    logic signed [31:0] infer_eval_g_inh_cur;
    logic signed [31:0] infer_eval_delayed_g_in;
    logic signed [31:0] infer_eval_exc_drive_dt;
    logic signed [31:0] infer_eval_inh_drive_dt;
    logic signed [31:0] infer_eval_leak_dt;
    logic signed [31:0] infer_eval_i_syn_exc_step;
    logic signed [31:0] infer_eval_i_syn_inh_step;
    (* use_dsp = "yes" *) logic signed [63:0] infer_eval_i_syn_exc_prod_q32;
    (* use_dsp = "yes" *) logic signed [63:0] infer_eval_i_syn_inh_prod_q32;
    logic signed [31:0] infer_eval_eexc_minus_v;
    logic signed [31:0] infer_eval_einh_minus_v;
    logic signed [31:0] infer_eval_vrest_minus_v;
    logic signed [31:0] infer_eval_dv_exc_step;
    logic        infer_eval_exc_refractory_ok;
    logic [15:0] infer_eval_last_spike_step;
    logic        infer_delay_pipe_valid;
    logic [6:0]  infer_delay_pipe_idx;
    logic signed [31:0] infer_delay_pipe_g_in_curr;
    logic signed [31:0] infer_delay_pipe_mul_term;
    (* use_dsp = "yes" *) logic signed [63:0] infer_delay_pipe_mul_prod_q32;
    logic signed [31:0] infer_delay_pipe_d0;
    logic signed [31:0] infer_delay_pipe_d1;
    logic signed [31:0] infer_delay_pipe_d2;
    logic signed [31:0] infer_delay_pipe_d3;
    logic signed [31:0] infer_delay_pipe_delayed_g_in;
    logic [6:0]  infer_commit_idx;
    logic signed [31:0] infer_commit_v_next;
    logic signed [31:0] infer_commit_thresh;
    logic signed [31:0] infer_commit_theta_next;
    logic signed [31:0] infer_commit_theta_decay;
    (* use_dsp = "yes" *) logic signed [63:0] infer_commit_theta_prod;
    logic        infer_commit_spike_now;
    logic        infer_skip_init_clear;
    logic        infer_force_no_input;
    logic        infer_model_state_valid;
    logic        memrd_pending;
    logic        memrd_wait;
    memrd_kind_t memrd_kind;

    assign SD_DQ1 = 1'b1;
    assign SD_DQ2 = 1'b1;

    assign rgb0[2] = tx_active;  // blue LED: UART TX active
    assign rgb0[1] = 1'b0; // green LED unused
    assign rgb0[0] = (resp_status == STATUS_OK); // red LED: OK result

    assign rgb1 = 3'b000;
    assign led = {ddr_calib_complete, ddr_clk_wiz_locked, 14'd0};
    assign pmoda = {rgb0[0], rgb0[1], rgb0[2]};

    always_ff @(posedge clk_100mhz_buf) begin
        if (btn[0]) begin
            clk_div <= 2'b00;
        end else begin
            clk_div <= clk_div + 2'b01;
        end
    end

    // Register CDC payload buses to reduce long comb fanout into synchronizers.
    always_ff @(posedge core_clk) begin
        ddr_req_payload_core_reg <= ddr_req_payload_core;
    end

    always_ff @(posedge clk_controller) begin
        ddr_rsp_payload_ddr_reg <= ddr_rsp_payload_ddr;
    end

    always_comb begin
        ddr_req_payload_core = {
            ddr_req_tag_core,              // [72:57]
            ddr_req_wdata_core,            // [56:25]
            ddr_req_addr_word_core[23:0],  // [24:1]
            ddr_req_we_core                // [0]
        };
        ddr_rsp_payload_ddr = {
            (ddr_rsp_status_ddr == STATUS_OK), // [49]
            ddr_rsp_was_write_ddr,             // [48]
            ddr_rsp_req_tag_ddr,               // [47:32]
            ddr_rsp_rdata_ddr                  // [31:0]
        };
    end

    // Cross-domain payload transfer is synchronized independently from toggle handshakes.
    xpm_cdc_array_single #(
        .DEST_SYNC_FF(2),
        .INIT_SYNC_FF(0),
        .SIM_ASSERT_CHK(0),
        .SRC_INPUT_REG(1),
        .WIDTH(73)
    ) u_cdc_req_payload (
        .src_clk (core_clk),
        .src_in  (ddr_req_payload_core_reg),
        .dest_clk(clk_controller),
        .dest_out(ddr_req_payload_ddr_sync)
    );

    xpm_cdc_array_single #(
        .DEST_SYNC_FF(2),
        .INIT_SYNC_FF(0),
        .SIM_ASSERT_CHK(0),
        .SRC_INPUT_REG(1),
        .WIDTH(50)
    ) u_cdc_rsp_payload (
        .src_clk (clk_controller),
        .src_in  (ddr_rsp_payload_ddr_reg),
        .dest_clk(core_clk),
        .dest_out(ddr_rsp_payload_core_sync)
    );

    xpm_cdc_single #(
        .DEST_SYNC_FF(2),
        .INIT_SYNC_FF(0),
        .SIM_ASSERT_CHK(0),
        .SRC_INPUT_REG(1)
    ) u_cdc_ddr_calib_complete (
        .src_clk (clk_controller),
        .src_in  (ddr_calib_complete),
        .dest_clk(core_clk),
        .dest_out(ddr_calib_complete_core)
    );

    // Explicit synchronous read port for infer weight RAM to push Vivado toward BRAM
    // inference (instead of LUTRAM/distributed RAM).
    always_ff @(posedge core_clk) begin
        infer_w_rd_data_q <= infer_w_rd_data;
        infer_w_rd_data_q_lane1 <= infer_w_rd_data_lane1;

        if (raw_image0_wr_en && !imgload_target_buf_sel) begin
            raw_image0_mem[raw_image0_wr_addr] <= raw_image0_wr_data;
        end
        if (raw_image0_wr_en && imgload_target_buf_sel) begin
            raw_image1_mem[raw_image0_wr_addr] <= raw_image0_wr_data;
        end
        raw_image0_rd_data <= raw_image0_mem[raw_image0_rd_addr];
        raw_image1_rd_data <= raw_image1_mem[raw_image0_rd_addr];

        if (batch_label_wr_en) begin
            batch_label_mem[batch_label_wr_addr] <= batch_label_wr_data;
        end
        batch_label_rd_data <= batch_label_mem[batch_label_rd_addr];
        if (batch_assign_wr_en) begin
            batch_assign_mem[batch_assign_wr_addr] <= batch_assign_wr_data;
        end
        batch_assign_rd_data <= batch_assign_mem[batch_assign_rd_addr];

        if (train_label_sum_wr_en) begin
            train_label_sum_mem[train_label_sum_wr_addr] <= train_label_sum_wr_data;
        end
        train_label_sum_rd_data <= train_label_sum_mem[train_label_sum_rd_addr];

        if (infer_poisson_thresh_wr_en) begin
            infer_poisson_thresh_mem[infer_poisson_thresh_wr_addr] <= infer_poisson_thresh_wr_data;
        end
        infer_poisson_thresh_rd_data <= infer_poisson_thresh_mem[infer_poisson_thresh_rd_addr];

        if (train_xin_wr_en_pipe) begin
            train_xin_mem[train_xin_wr_addr_pipe] <= train_xin_wr_data_pipe;
        end
        if (btn[0]) begin
            train_xin_wr_en_pipe <= 1'b0;
            train_xin_wr_addr_pipe <= 10'd0;
            train_xin_wr_data_pipe <= 32'd0;
        end else begin
            train_xin_wr_en_pipe <= train_xin_wr_en;
            train_xin_wr_addr_pipe <= train_xin_wr_addr;
            train_xin_wr_data_pipe <= train_xin_wr_data;
        end
        train_xin_rd_data <= train_xin_mem[train_xin_rd_addr];

        if (train_xexc_wr_en) begin
            train_xexc_mem[train_xexc_wr_addr] <= train_xexc_wr_data;
        end
        train_xexc_rd_data <= train_xexc_mem[train_xexc_rd_addr];

        if (train_xpost2_wr_en) begin
            train_xpost2_mem[train_xpost2_wr_addr] <= train_xpost2_wr_data;
        end
        train_xpost2_rd_data <= train_xpost2_mem[train_xpost2_rd_addr];

        train_label_count_rd_data <= train_label_count[train_label_count_rd_addr];
    end

    always_comb begin
        if (batch_active) begin
            raw_image_compute_valid = batch_compute_buf_sel ? raw_image1_valid : raw_image0_valid;
            raw_image_compute_sum_u8 = batch_compute_buf_sel ? raw_image1_sum_u8 : raw_image0_sum_u8;
            raw_image_compute_rd_data = batch_compute_buf_sel ? raw_image1_rd_data : raw_image0_rd_data;
            raw_image_fill_valid = batch_fill_buf_sel ? raw_image1_valid : raw_image0_valid;
        end else begin
            raw_image_compute_valid = raw_image0_valid;
            raw_image_compute_sum_u8 = raw_image0_sum_u8;
            raw_image_compute_rd_data = raw_image0_rd_data;
            raw_image_fill_valid = raw_image0_valid;
        end
    end

    assign infer_spike_rd_data = spike_count_rdata;

    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(10),
        .ADDR_WIDTH_B(10),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(10),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(N_IN * 10),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(10),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_EMBEDDED_CONSTRAINT(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(10),
        .WRITE_MODE_B("read_first")
    ) u_infer_prelist_mem (
        .sleep          (1'b0),
        .clka           (core_clk),
        .ena            (infer_pre_wr_en),
        .wea            (infer_pre_wr_en),
        .addra          (infer_pre_wr_addr),
        .dina           (infer_pre_wr_data),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .clkb           (core_clk),
        .rstb           (1'b0),
        .enb            (1'b1),
        .regceb         (1'b1),
        .addrb          (infer_pre_rd_addr),
        .doutb          (infer_pre_rd_data),
        .sbiterrb       (),
        .dbiterrb       ()
    );

    // Per-step pre spike bitmap for sparse CSR inference.
    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(10),
        .ADDR_WIDTH_B(10),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(1),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("distributed"),
        .MEMORY_SIZE(N_IN),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(1),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_EMBEDDED_CONSTRAINT(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(1),
        .WRITE_MODE_B("read_first")
    ) u_infer_pre_spike_map (
        .sleep          (1'b0),
        .clka           (core_clk),
        .ena            (infer_pre_spike_wr_en),
        .wea            (infer_pre_spike_wr_en),
        .addra          (infer_pre_spike_wr_addr),
        .dina           (infer_pre_spike_wr_data),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .clkb           (core_clk),
        .rstb           (1'b0),
        .enb            (1'b1),
        .regceb         (1'b1),
        .addrb          (infer_pre_spike_rd_addr),
        .doutb          (infer_pre_spike_rd_data),
        .sbiterrb       (),
        .dbiterrb       ()
    );

    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(10),
        .ADDR_WIDTH_B(10),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(1),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("distributed"),
        .MEMORY_SIZE(N_IN),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(1),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_EMBEDDED_CONSTRAINT(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(1),
        .WRITE_MODE_B("read_first")
    ) u_infer_pre_spike_map_lane1 (
        .sleep          (1'b0),
        .clka           (core_clk),
        .ena            (infer_pre_spike_wr_en),
        .wea            (infer_pre_spike_wr_en),
        .addra          (infer_pre_spike_wr_addr),
        .dina           (infer_pre_spike_wr_data),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .clkb           (core_clk),
        .rstb           (1'b0),
        .enb            (1'b1),
        .regceb         (1'b1),
        .addrb          (infer_pre_spike_rd_addr_lane1),
        .doutb          (infer_pre_spike_rd_data_lane1),
        .sbiterrb       (),
        .dbiterrb       ()
    );

    // Sparse inference weight table: csr_weight[N_EDGES].
    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(W_ADDR_W),
        .ADDR_WIDTH_B(W_ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(16),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("data/csr_weight_q16.mem"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(N_EDGES * 16),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(16),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_EMBEDDED_CONSTRAINT(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(16),
        .WRITE_MODE_B("read_first")
    ) u_infer_w_rom_bram (
        .sleep          (1'b0),
        .clka           (core_clk),
        .ena            (infer_w_wr_en),
        .wea            (infer_w_wr_en),
        .addra          (infer_w_wr_addr),
        .dina           (infer_w_wr_data),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .clkb           (core_clk),
        .rstb           (1'b0),
        .enb            (1'b1),
        .regceb         (1'b1),
        .addrb          (infer_w_rd_addr),
        .doutb          (infer_w_rd_data),
        .sbiterrb       (),
        .dbiterrb       ()
    );

    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(W_ADDR_W),
        .ADDR_WIDTH_B(W_ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(16),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("data/csr_weight_q16.mem"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(N_EDGES * 16),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(16),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_EMBEDDED_CONSTRAINT(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(16),
        .WRITE_MODE_B("read_first")
    ) u_infer_w_rom_bram_lane1 (
        .sleep          (1'b0),
        .clka           (core_clk),
        .ena            (infer_w_wr_en),
        .wea            (infer_w_wr_en),
        .addra          (infer_w_wr_addr),
        .dina           (infer_w_wr_data),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .clkb           (core_clk),
        .rstb           (1'b0),
        .enb            (1'b1),
        .regceb         (1'b1),
        .addrb          (infer_w_rd_addr_lane1),
        .doutb          (infer_w_rd_data_lane1),
        .sbiterrb       (),
        .dbiterrb       ()
    );

    // Hold-fix buffer chain for spike_count BRAM write data path.
    // The explicit LUT1 stage increases minimum data-path delay into RAMB DI pins.
    genvar spike_hold_i;
    generate
        for (spike_hold_i = 0; spike_hold_i < 16; spike_hold_i = spike_hold_i + 1) begin : g_spike_count_holdfix
            (* keep = "true", dont_touch = "true" *)
            LUT1 #(
                .INIT(2'b10)
            ) u_lut1_holdfix (
                .I0(spike_count_wdata[spike_hold_i]),
                .O(spike_count_wdata_holdfix[spike_hold_i])
            );
        end
    endgenerate

    // Spike count storage in explicit BRAM (replaces inferred array RAM).
    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(SPIKE_CNT_ADDR_W),
        .ADDR_WIDTH_B(SPIKE_CNT_ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(16),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(N_NEURONS * 16),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(16),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_EMBEDDED_CONSTRAINT(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(16),
        .WRITE_MODE_B("read_first")
    ) u_spike_count_bram (
        .sleep          (1'b0),
        .clka           (core_clk),
        .ena            (1'b1),
        .wea            (spike_count_we),
        .addra          (spike_count_waddr),
        .dina           (spike_count_wdata_holdfix),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .clkb           (core_clk),
        .rstb           (1'b0),
        .enb            (1'b1),
        .regceb         (1'b1),
        .addrb          (infer_spike_rd_addr),
        .doutb          (spike_count_rdata),
        .sbiterrb       (),
        .dbiterrb       ()
    );

    // Snapshot spike count storage in explicit BRAM (replaces inferred array RAM).
    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(SPIKE_CNT_ADDR_W),
        .ADDR_WIDTH_B(SPIKE_CNT_ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(16),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(N_NEURONS * 16),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(16),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_EMBEDDED_CONSTRAINT(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(16),
        .WRITE_MODE_B("read_first")
    ) u_spike_snap_bram (
        .sleep          (1'b0),
        .clka           (core_clk),
        .ena            (1'b1),
        .wea            (snap_count_we),
        .addra          (snap_count_waddr),
        .dina           (snap_count_wdata),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .clkb           (core_clk),
        .rstb           (1'b0),
        .enb            (1'b1),
        .regceb         (1'b1),
        .addrb          (snap_count_raddr),
        .doutb          (snap_count_rdata),
        .sbiterrb       (),
        .dbiterrb       ()
    );

    // CSR row pointer: [N_NEURONS+1], entries are edge offsets into csr_w/csr_col_idx.
    xpm_memory_sprom #(
        .ADDR_WIDTH_A(ROW_IDX_W + 1),
        .AUTO_SLEEP_TIME(0),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("data/csr_row_ptr.mem"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE((N_NEURONS + 1) * CSR_ROW_PTR_W),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(CSR_ROW_PTR_W),
        .READ_LATENCY_A(1),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep")
    ) u_csr_row_ptr_bram (
        .sleep          (1'b0),
        .clka           (core_clk),
        .rsta           (1'b0),
        .ena            (1'b1),
        .regcea         (1'b1),
        .addra          (csr_row_ptr_rd_addr),
        .douta          (csr_row_ptr_rd_data),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .sbiterra       (),
        .dbiterra       ()
    );

    // CSR col index: [N_EDGES], input index for each edge.
    xpm_memory_sprom #(
        .ADDR_WIDTH_A(EDGE_ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("data/csr_col_idx.mem"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(N_EDGES * COL_IDX_W),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(COL_IDX_W),
        .READ_LATENCY_A(1),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep")
    ) u_csr_col_idx_bram (
        .sleep          (1'b0),
        .clka           (core_clk),
        .rsta           (1'b0),
        .ena            (1'b1),
        .regcea         (1'b1),
        .addra          (csr_col_idx_rd_addr),
        .douta          (csr_col_idx_rd_data),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .sbiterra       (),
        .dbiterra       ()
    );

    xpm_memory_sprom #(
        .ADDR_WIDTH_A(EDGE_ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("data/csr_col_idx.mem"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(N_EDGES * COL_IDX_W),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(COL_IDX_W),
        .READ_LATENCY_A(1),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep")
    ) u_csr_col_idx_bram_lane1 (
        .sleep          (1'b0),
        .clka           (core_clk),
        .rsta           (1'b0),
        .ena            (1'b1),
        .regcea         (1'b1),
        .addra          (csr_col_idx_rd_addr_lane1),
        .douta          (csr_col_idx_rd_data_lane1),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .sbiterra       (),
        .dbiterra       ()
    );

    // CSC col pointer: [N_IN+1], reverse index range for each pre neuron.
    xpm_memory_sprom #(
        .ADDR_WIDTH_A(COL_IDX_W + 1),
        .AUTO_SLEEP_TIME(0),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("data/csc_col_ptr.mem"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE((N_IN + 1) * CSR_ROW_PTR_W),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(CSR_ROW_PTR_W),
        .READ_LATENCY_A(1),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep")
    ) u_csc_col_ptr_bram (
        .sleep          (1'b0),
        .clka           (core_clk),
        .rsta           (1'b0),
        .ena            (1'b1),
        .regcea         (1'b1),
        .addra          (csc_col_ptr_rd_addr),
        .douta          (csc_col_ptr_rd_data),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .sbiterra       (),
        .dbiterra       ()
    );

    // CSC row index: [N_EDGES], post neuron index for each reverse entry.
    xpm_memory_sprom #(
        .ADDR_WIDTH_A(EDGE_ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("data/csc_row_idx.mem"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(N_EDGES * ROW_IDX_W),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(ROW_IDX_W),
        .READ_LATENCY_A(1),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep")
    ) u_csc_row_idx_bram (
        .sleep          (1'b0),
        .clka           (core_clk),
        .rsta           (1'b0),
        .ena            (1'b1),
        .regcea         (1'b1),
        .addra          (csc_row_idx_rd_addr),
        .douta          (csc_row_idx_rd_data),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .sbiterra       (),
        .dbiterra       ()
    );

    // CSC edge index: [N_EDGES], mapping reverse entry -> edge index in infer_w (CSR w array).
    xpm_memory_sprom #(
        .ADDR_WIDTH_A(EDGE_ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("data/csc_edge_idx.mem"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(N_EDGES * EDGE_ADDR_W),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_A(EDGE_ADDR_W),
        .READ_LATENCY_A(1),
        .READ_RESET_VALUE_A("0"),
        .RST_MODE_A("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(1),
        .WAKEUP_TIME("disable_sleep")
    ) u_csc_edge_idx_bram (
        .sleep          (1'b0),
        .clka           (core_clk),
        .rsta           (1'b0),
        .ena            (1'b1),
        .regcea         (1'b1),
        .addra          (csc_edge_idx_rd_addr),
        .douta          (csc_edge_idx_rd_data),
        .injectsbiterra (1'b0),
        .injectdbiterra (1'b0),
        .sbiterra       (),
        .dbiterra       ()
    );

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

    // Hold-fix buffer chain for divider->STDP normalized weight register path.
    genvar stdp_div_hold_i;
    generate
        for (stdp_div_hold_i = 0; stdp_div_hold_i < 32; stdp_div_hold_i = stdp_div_hold_i + 1) begin : g_stdp_div_holdfix
            (* keep = "true", dont_touch = "true" *)
            LUT1 #(
                .INIT(2'b10)
            ) u_lut1_holdfix (
                .I0(train_stdp_div_q[stdp_div_hold_i]),
                .O(train_stdp_div_q_holdfix[stdp_div_hold_i])
            );
        end
    endgenerate
    
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

    function automatic [31:0] batch_summary_select(input logic [5:0] field_idx);
        begin
            case (field_idx)
                6'd0: batch_summary_select = batch_cfg_start_sample_idx;
                6'd1: batch_summary_select = batch_cfg_num_samples;
                6'd2: batch_summary_select = batch_cfg_seed;
                6'd3: batch_summary_select = batch_current_sample_idx;
                6'd4: batch_summary_select = batch_processed_samples;
                6'd5: batch_summary_select = batch_total_spikes;
                6'd6: batch_summary_select = batch_correct_count;
                6'd7: batch_summary_select = batch_elapsed_cycles[31:0];
                6'd8: batch_summary_select = batch_load_cycles[31:0];
                6'd9: batch_summary_select = batch_train_core_cycles[31:0];
                6'd10: batch_summary_select = batch_infer_core_cycles[31:0];
                6'd11: batch_summary_select = batch_label_stats_cycles[31:0];
                6'd12: batch_summary_select = batch_infer_eval_cycles[31:0];
                6'd13: batch_summary_select = batch_other_cycles[31:0];
                6'd14: batch_summary_select = batch_train_inject_infer_cycles[31:0];
                6'd15: batch_summary_select = batch_train_blank_infer_cycles[31:0];
                6'd16: batch_summary_select = batch_train_snap_cycles[31:0];
                6'd17: batch_summary_select = batch_train_rebase_cycles[31:0];
                6'd18: batch_summary_select = batch_train_evt_pre_cycles[31:0];
                6'd19: batch_summary_select = batch_train_evt_post_cycles[31:0];
                6'd20: batch_summary_select = batch_train_accum_cycles[31:0];
                6'd21: batch_summary_select = batch_elapsed_cycles[63:32];
                6'd22: batch_summary_select = batch_load_cycles[63:32];
                6'd23: batch_summary_select = batch_train_core_cycles[63:32];
                6'd24: batch_summary_select = batch_infer_core_cycles[63:32];
                6'd25: batch_summary_select = batch_label_stats_cycles[63:32];
                6'd26: batch_summary_select = batch_infer_eval_cycles[63:32];
                6'd27: batch_summary_select = batch_other_cycles[63:32];
                6'd28: batch_summary_select = batch_train_inject_infer_cycles[63:32];
                6'd29: batch_summary_select = batch_train_blank_infer_cycles[63:32];
                6'd30: batch_summary_select = batch_train_snap_cycles[63:32];
                6'd31: batch_summary_select = batch_train_rebase_cycles[63:32];
                6'd32: batch_summary_select = batch_train_evt_pre_cycles[63:32];
                6'd33: batch_summary_select = batch_train_evt_post_cycles[63:32];
                6'd34: batch_summary_select = batch_train_accum_cycles[63:32];
                default: batch_summary_select = 32'd0;
            endcase
        end
    endfunction

    function automatic logic infer_state_is_evt_pre(input infer_state_t state_in);
        begin
            case (state_in)
                INFER_EVT_PRE_PRELIST_REQ,
                INFER_EVT_PRE_PRELIST_WAIT,
                INFER_EVT_PRE_PTR0_REQ,
                INFER_EVT_PRE_PTR0_WAIT,
                INFER_EVT_PRE_PTR1_REQ,
                INFER_EVT_PRE_PTR1_WAIT,
                INFER_EVT_PRE_EDGE_REQ,
                INFER_EVT_PRE_EDGE_WAIT,
                INFER_EVT_PRE_TRACE_REQ,
                INFER_EVT_PRE_TRACE_WAIT,
                INFER_EVT_PRE_W_WAIT,
                INFER_EVT_PRE_APPLY,
                INFER_EVT_PRE_APPLY_MUL1,
                INFER_EVT_PRE_APPLY_MUL2,
                INFER_EVT_PRE_APPLY_CLIP,
                INFER_EVT_PRE_APPLY_WNEXT: infer_state_is_evt_pre = 1'b1;
                default: infer_state_is_evt_pre = 1'b0;
            endcase
        end
    endfunction

    function automatic logic infer_state_is_evt_post(input infer_state_t state_in);
        begin
            case (state_in)
                INFER_EVT_POST_PTR0_REQ,
                INFER_EVT_POST_PTR0_WAIT,
                INFER_EVT_POST_PTR1_REQ,
                INFER_EVT_POST_PTR1_WAIT,
                INFER_EVT_POST_EDGE_REQ,
                INFER_EVT_POST_EDGE_WAIT,
                INFER_EVT_POST_TRACE_REQ,
                INFER_EVT_POST_TRACE_WAIT,
                INFER_EVT_POST_W_WAIT,
                INFER_EVT_POST_APPLY,
                INFER_EVT_POST_APPLY_MUL1,
                INFER_EVT_POST_APPLY_MUL2,
                INFER_EVT_POST_APPLY_CLIP,
                INFER_EVT_POST_APPLY_WNEXT: infer_state_is_evt_post = 1'b1;
                default: infer_state_is_evt_post = 1'b0;
            endcase
        end
    endfunction

    function automatic logic infer_state_is_accum_core(input infer_state_t state_in);
        begin
            case (state_in)
                INFER_GEN_INPUT_SPIKES,
                INFER_ACCUM_NEURON,
                INFER_ACCUM_NEURON_GIN_MUL,
                INFER_ACCUM_NEURON_GIN_MUL_ROUND,
                INFER_ACCUM_NEURON_GIN_COMB,
                INFER_ACCUM_NEURON_PIPE,
                INFER_NEURON_DV_PRE,
                INFER_NEURON_DV_DRIVE,
                INFER_NEURON_DV_SYN,
                INFER_NEURON_DV_SYN_ROUND,
                INFER_NEURON_VNEXT,
                INFER_NEURON_SPIKE,
                INFER_NEURON_THETA_PRE,
                INFER_NEURON_THETA_ROUND,
                INFER_NEURON_COMMIT,
                INFER_NEURON_WRITE,
                INFER_APPLY_WTA,
                INFER_APPLY_WTA_PRE,
                INFER_APPLY_WTA_PRE_MUL,
                INFER_APPLY_WTA_PRE_ROUND,
                INFER_APPLY_WTA_INH,
                INFER_APPLY_WTA_INH_PROD,
                INFER_APPLY_WTA_INH_MUL,
                INFER_APPLY_WTA_INH_DV,
                INFER_APPLY_WTA_INH_VPROP,
                INFER_APPLY_WTA_INH_POST,
                INFER_APPLY_WTA_ACCUM,
                INFER_WTA_PASS2_PRE,
                INFER_WTA_PASS2,
                INFER_WTA_PASS2_WRITE: infer_state_is_accum_core = 1'b1;
                default: infer_state_is_accum_core = 1'b0;
            endcase
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
            ddr_req_addr_word_ddr <= 32'd0;
            ddr_req_wdata_ddr <= 32'd0;
            ddr_rsp_rdata_ddr <= 32'd0;
            ddr_rsp_status_ddr <= STATUS_BAD_PACKET;
            ddr_req_tag_ddr <= 16'd0;
            ddr_rsp_req_tag_ddr <= 16'd0;
            ddr_rsp_was_write_ddr <= 1'b0;
        end else begin
            ddr_req_toggle_ddr_sync1 <= ddr_req_toggle_core;
            ddr_req_toggle_ddr_sync2 <= ddr_req_toggle_ddr_sync1;
            ddr_wb_stb <= 1'b0;

            case (ddr_bridge_state)
                DDRBR_IDLE: begin
                    if (ddr_req_toggle_ddr_sync2 != ddr_req_toggle_ddr_seen) begin
                        ddr_req_toggle_ddr_seen <= ddr_req_toggle_ddr_sync2;
                        ddr_req_tag_ddr <= ddr_req_payload_ddr_sync[72:57];
                        ddr_req_we_ddr <= ddr_req_payload_ddr_sync[0];
                        ddr_req_addr_word_ddr <= {8'd0, ddr_req_payload_ddr_sync[24:1]};
                        ddr_req_wdata_ddr <= ddr_req_payload_ddr_sync[56:25];
                        ddr_bridge_state <= DDRBR_ISSUE;
                    end
                end

                DDRBR_ISSUE: begin
                    if (!ddr_calib_complete) begin
                        ddr_rsp_status_ddr <= STATUS_BAD_PACKET;
                        ddr_rsp_was_write_ddr <= ddr_req_we_ddr;
                        ddr_rsp_req_tag_ddr <= ddr_req_tag_ddr;
                        ddr_rsp_rdata_ddr  <= 32'sd0;
                        ddr_bridge_state   <= DDRBR_RESP;
                    end else if (!ddr_wb_stall) begin
                        ddr_wb_we   <= ddr_req_we_ddr;
                        ddr_wb_addr <= {2'b00, ddr_req_addr_word_ddr[23:2]};
                        if (ddr_req_we_ddr) begin
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
                        ddr_rsp_was_write_ddr <= ddr_req_we_ddr;
                        ddr_rsp_req_tag_ddr <= ddr_req_tag_ddr;
                        ddr_wb_stb <= 1'b0;
                        if (ddr_req_we_ddr) begin
                            ddr_rsp_status_ddr <= STATUS_OK;
                            ddr_rsp_rdata_ddr  <= ddr_req_addr_word_ddr;
                            ddr_bridge_state <= DDRBR_RESP;
                        end else begin
                            // Some DDR controller wrappers present read data one cycle after ACK.
                            ddr_bridge_state <= DDRBR_READ_CAPTURE;
                        end
                    end
                end

                DDRBR_READ_CAPTURE: begin
                    ddr_rsp_status_ddr <= STATUS_OK;
                    case (ddr_req_addr_word_ddr[1:0])
                        2'd0: ddr_rsp_rdata_ddr <= ddr_wb_rdata[31:0];
                        2'd1: ddr_rsp_rdata_ddr <= ddr_wb_rdata[63:32];
                        2'd2: ddr_rsp_rdata_ddr <= ddr_wb_rdata[95:64];
                        default: ddr_rsp_rdata_ddr <= ddr_wb_rdata[127:96];
                    endcase
                    ddr_bridge_state <= DDRBR_RESP;
                end

                DDRBR_RESP: begin
                    ddr_rsp_toggle_ddr <= ~ddr_rsp_toggle_ddr;
                    ddr_bridge_state <= DDRBR_IDLE;
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
            batch_status_word <= 32'd0;
            batch_summary_word <= 32'd0;
            train_busy_uart_blocked <= 1'b0;
            tx_dv             <= 1'b0;
            tx_byte           <= 8'h00;
            ddr_req_pending_core <= 1'b0;
            ddr_req_we_core      <= 1'b0;
            ddr_req_from_sd_core <= 1'b0;
            ddr_req_from_imgload_core <= 1'b0;
            ddr_req_from_train_core <= 1'b0;
            ddr_req_kind_core <= DDR_REQ_NONE;
            ddr_rsp_kind_core <= DDR_REQ_NONE;
            ddr_req_addr_word_core <= 32'd0;
            ddr_req_wdata_core   <= 32'd0;
            ddr_req_wide_core    <= 1'b0;
            ddr_req_wdata128_core <= 128'd0;
            ddr_req_sel16_core   <= 16'd0;
            ddr_req_word_count_core <= 3'd0;
            ddr_req_toggle_core  <= 1'b0;
            ddr_req_tag_core <= 16'd0;
            ddr_req_tag_expect_core <= 16'd0;
            ddr_req_pending_core_prev <= 1'b0;
            ddr_rsp_toggle_core_sync1 <= 1'b0;
            ddr_rsp_toggle_core_sync2 <= 1'b0;
            ddr_rsp_toggle_core_seen  <= 1'b0;
            ddr_rsp_capture_pending_core <= 1'b0;
            ddr_rsp_payload_ready_core <= 1'b0;
            ddr_rsp_payload_settle_core <= 2'd0;
            ddr_rsp_drain_active_core <= 1'b0;
            ddr_rsp_drain_quiet_core <= 2'd0;
            ddr_resp_rdata_core <= 32'd0;
            ddr_resp_status_core <= STATUS_BAD_PACKET;
            ddr_resp_was_write_core <= 1'b0;
            rx_timeout_counter<= '0;
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
            sd_copy_dest_base_word <= IMG_STAGING_BASE_WORD;
            raw_image0_valid    <= 1'b0;
            raw_image1_valid    <= 1'b0;
            raw_num_images      <= 32'd0;
            raw_bytes_per_image <= 32'd0;
            raw_image0_capture_idx <= 10'd0;
            raw_image1_capture_idx <= 10'd0;
            raw_image0_sum_u8   <= 32'd0;
            raw_image1_sum_u8   <= 32'd0;
            raw_image0_rd_addr  <= 10'd0;
            raw_image0_wr_en    <= 1'b0;
            raw_image0_wr_addr  <= 10'd0;
            raw_image0_wr_data  <= 8'd0;
            batch_compute_buf_sel <= 1'b0;
            batch_fill_buf_sel <= 1'b1;
            imgload_target_buf_sel <= 1'b0;
            batch_prefetch_active <= 1'b0;
            batch_prefetch_issue_pending <= 1'b0;
            batch_prefetch_ready <= 1'b0;
            batch_prefetch_sample_idx <= 32'd0;
            batch_prefetch_img_byte_off <= 32'd0;
            batch_prefetch_img_sector_off <= 32'd0;
            batch_prefetch_img_byte_in_sector <= 32'd0;
            batch_prefetch_img_sectors_needed <= 32'd0;
            sd_copy_resp_pending <= 1'b0;
            sd_copy_resp_status <= STATUS_OK;
            sd_copy_resp_result <= 32'sd0;
            imgload_active      <= 1'b0;
            imgload_addr_word    <= 32'd0;
            imgload_lane        <= 2'd0;
            imgload_byte_idx     <= 10'd0;
            imgload_total_bytes  <= 10'd0;
            imgload_sum_u8_accum <= 32'd0;
            imgload_word_valid   <= 1'b0;
            imgload_word_data    <= 32'd0;
            imgload_word_lane    <= 2'd0;
            imgload_ddr_wait_counter <= 32'd0;
            imgload_start_pending <= 1'b0;
            imgload_start_addr_word <= 32'd0;
            imgload_start_lane <= 2'd0;
            imgload_start_total_bytes <= 10'd0;
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
            train_trace_use_infer_prelist <= 1'b0;
            train_trace_skip_a   <= 1'b0;
            train_trace_skip_b   <= 1'b0;
            train_trace_multi_post_active <= 1'b0;
            train_trace_post_scan_idx <= 7'd0;
            infer_trace_next_post_found <= 1'b0;
            infer_trace_next_post_idx <= 7'd0;
            train_stdp_active    <= 1'b0;
            train_stdp_state     <= TSK_IDLE;
            train_stdp_row0      <= 7'd0;
            train_stdp_row_end   <= 7'd0;
            train_stdp_row_idx   <= 7'd0;
            train_stdp_col_idx   <= 10'd0;
            train_stdp_w_new     <= 32'sd0;
            train_stdp_row_sum_abs <= 32'd0;
            train_stdp_w_row_base <= TRAIN_BASE_W_Q16_WORDS;
            train_stdp_a_row_base <= TRAIN_BASE_A_Q16_WORDS;
            train_stdp_bt_col_base <= TRAIN_BASE_BT_Q16_WORDS;
            train_stdp_dividend  <= 32'd0;
            train_stdp_divisor   <= 32'd1;
            train_stdp_update_nt <= TRAIN_UPDATE_NT;
            train_stdp_div_valid <= 1'b0;
            train_stdp_batch_active <= 1'b0;
            train_chunk_active   <= 1'b0;
            train_chunk_state    <= TCK_IDLE;
            train_chunk_mode     <= 2'd0;
            train_chunk_samples_left <= 16'd0;
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
            train_chunk_winner_scan_phase <= 1'b0;
            train_chunk_snap_copy_idx <= 7'd0;
            train_chunk_last_infer_spikes <= 32'd0;
            train_chunk_last_blank_spikes <= 32'd0;
            train_chunk_retry_curr_max_fr <= TRAIN_RETRY_MAX_FR_START;
            train_chunk_retry_accepted_max_fr <= TRAIN_RETRY_MAX_FR_START;
            train_chunk_retry_continue_infer <= 1'b0;
            train_rebase_neuron_idx <= 7'd0;
            train_rebase_input_idx <= 10'd0;
            train_rebase_edge_idx <= '0;
            train_rebase_edge_end <= '0;
            train_rebase_accum <= 32'sd0;
            train_rebase_phase <= 3'd0;
            train_gen_active     <= 1'b0;
            train_gen_state      <= TGK_IDLE;
            train_gen_base_word  <= 32'd0;
            train_gen_count_total<= 16'd0;
            train_gen_idx        <= 16'd0;
            train_gen_lcg_state  <= 32'd0;
            train_gen_curr_word  <= 32'd0;
            train_gen_lcg_enable <= 1'b0;
            train_gen_cache_mode <= 2'd0;
            train_mem_init_active <= 1'b0;
            // Phase2 no longer uses A/BT matrix initialization.
            train_mem_init_done <= 1'b1;
            train_mem_init_state <= TMI_IDLE;
            train_mem_init_idx <= '0;
            train_label_stats_active <= 1'b0;
            train_label_stats_state <= TLS_IDLE;
            train_label_stats_label <= 4'd0;
            train_label_stats_idx <= 10'd0;
            train_label_stats_base_idx <= 10'd0;
            train_label_stats_spike_q <= 16'd0;
            batch_cfg0_valid <= 1'b0;
            batch_cfg1_valid <= 1'b0;
            batch_cfg_mode_train <= 1'b0;
            batch_cfg_start_sample_idx <= 32'd0;
            batch_cfg_num_samples <= 32'd0;
            batch_cfg_seed <= 32'd0;
            batch_cfg_start_lba <= 32'd2048;
            batch_cfg_start_byte_off <= 32'd0;
            batch_cfg_start_sector_off <= 32'd0;
            batch_cfg_start_byte_in_sector <= 32'd0;
            batch_cfg_start_sectors_needed <= 32'd0;
            batch_cfg_num_samples_bytes <= 32'd0;
            batch_cfg_cache_total_sectors <= 32'd0;
            batch_cfg_cache_total_words <= 32'd0;
            batch_cfg_cache_fits <= 1'b0;
            batch_active <= 1'b0;
            batch_done <= 1'b0;
            batch_error <= 1'b0;
            batch_error_code <= BATCH_ERR_NONE;
            batch_phase <= BATCH_PHASE_IDLE;
            batch_processed_samples <= 32'd0;
            batch_current_sample_idx <= 32'd0;
            batch_total_spikes <= 32'd0;
            batch_correct_count <= 32'd0;
            batch_elapsed_cycles <= 64'd0;
            batch_load_cycles <= 64'd0;
            batch_train_core_cycles <= 64'd0;
            batch_infer_core_cycles <= 64'd0;
            batch_label_stats_cycles <= 64'd0;
            batch_infer_eval_cycles <= 64'd0;
            batch_other_cycles <= 64'd0;
            batch_train_inject_infer_cycles <= 64'd0;
            batch_train_blank_infer_cycles <= 64'd0;
            batch_train_snap_cycles <= 64'd0;
            batch_train_rebase_cycles <= 64'd0;
            batch_train_evt_pre_cycles <= 64'd0;
            batch_train_evt_post_cycles <= 64'd0;
            batch_train_accum_cycles <= 64'd0;
            batch_img_byte_off <= 32'd0;
            batch_img_sector_off <= 32'd0;
            batch_img_byte_in_sector <= 32'd0;
            batch_img_sectors_needed <= 32'd0;
            batch_use_cached_images <= 1'b0;
            batch_cache_img_byte_off <= 32'd0;
            batch_label_wr_en <= 1'b0;
            batch_label_wr_addr <= 14'd0;
            batch_label_wr_data <= 8'd0;
            batch_label_rd_addr <= 14'd0;
            batch_assign_wr_en <= 1'b0;
            batch_assign_wr_addr <= 7'd0;
            batch_assign_wr_data <= 4'd0;
            batch_assign_rd_addr <= 7'd0;
            batch_infer_eval_active <= 1'b0;
            batch_infer_eval_state <= BIE_IDLE;
            batch_infer_eval_idx <= 7'd0;
            batch_infer_eval_label_idx <= 4'd0;
            batch_infer_eval_spike_q <= 16'd0;
            batch_infer_eval_assign_q <= 4'd0;
            batch_infer_eval_pred_label <= 4'd0;
            batch_infer_eval_best_label <= 4'd0;
            batch_infer_eval_best_sum <= 32'd0;
            batch_infer_eval_best_count <= 16'd0;
            batch_infer_eval_curr_sum_q <= 32'd0;
            batch_infer_eval_curr_count_q <= 16'd0;
            batch_infer_eval_best_count_eff_q <= 16'd0;
            batch_infer_eval_cmp_lhs <= 64'd0;
            batch_infer_eval_cmp_rhs <= 64'd0;
            for (batch_eval_i = 0; batch_eval_i < 10; batch_eval_i = batch_eval_i + 1) begin
                batch_infer_eval_sum[batch_eval_i] <= 32'd0;
                batch_infer_eval_count[batch_eval_i] <= 16'd0;
            end
            train_xin_cache_valid <= 1'b0;
            train_xexc_cache_valid <= 1'b0;
            train_xin_wr_en <= 1'b0;
            train_xin_wr_addr <= 10'd0;
            train_xin_wr_data <= 32'd0;
            train_xin_rd_addr <= 10'd0;
            train_xexc_wr_en <= 1'b0;
            train_xexc_wr_addr <= 7'd0;
            train_xexc_wr_data <= 32'd0;
            train_xexc_rd_addr <= 7'd0;
            train_xpost2_wr_en <= 1'b0;
            train_xpost2_wr_addr <= 7'd0;
            train_xpost2_wr_data <= 32'd0;
            train_xpost2_rd_addr <= 7'd0;
            infer_active        <= 1'b0;
            infer_state         <= INFER_IDLE;
            infer_steps_target  <= 32'd0;
            infer_step_idx      <= 16'd0;
            infer_neuron_idx    <= 7'd0;
            infer_input_idx     <= 10'd0;
            infer_prep_idx      <= 10'd0;
            infer_accum         <= 32'sd0;
            infer_accum_weight_phase <= 3'd0;
            infer_w_rd_addr     <= '0;
            infer_w_wr_en       <= 1'b0;
            infer_w_wr_addr     <= '0;
            infer_w_wr_data     <= 16'd0;
            spike_count_we      <= 1'b0;
            spike_count_waddr   <= '0;
            spike_count_wdata   <= 16'd0;
            infer_pre_rd_addr   <= 10'd0;
            infer_apply_idx     <= 7'd0;
            infer_trace_phase <= 3'd0;
            infer_trace_spike_latched <= 1'b0;
            infer_apply_xexc_trace_q <= 32'sd0;
            infer_apply_xpost2_trace_q <= 32'sd0;
            infer_apply_xpost2_prod <= 64'sd0;
            infer_apply_xpost2_decay <= 32'sd0;
            infer_apply_xpost2_next <= 32'sd0;
            infer_sum_c_inh     <= 32'sd0;
            infer_pass2_g_inh_next <= 32'sd0;
            infer_step_winner_valid <= 1'b0;
            infer_step_winner_idx <= 7'd0;
            infer_trace_wait_last_step <= 1'b0;
            infer_evt_prelist_idx <= 10'd0;
            infer_evt_pre_idx <= 10'd0;
            infer_evt_post_idx <= 7'd0;
            infer_evt_post_input_idx <= 10'd0;
            infer_evt_has_winner <= 1'b0;
            infer_evt_winner_idx <= 7'd0;
            infer_evt_trace_val <= 32'sd0;
            infer_evt_post2_before_q <= 32'sd0;
            infer_evt_w_cur <= 32'sd0;
            infer_evt_mid_q16 <= 32'sd0;
            infer_evt_term_prod_q32 <= 64'sd0;
            infer_evt_term_q16 <= 32'sd0;
            infer_evt_dw_q16 <= 32'sd0;
            infer_evt_w_next_q16 <= 32'sd0;
            infer_evt_edge_idx <= '0;
            infer_evt_edge_end <= '0;
            infer_evt_edge_ptr <= '0;
            infer_total_spikes  <= 32'd0;
            infer_rng_state     <= 32'd0;
            infer_rng_mul_prod_q32 <= 64'd0;
            infer_poisson_num_const_cfg <= POISSON_NUM_CONST;
            infer_dividend      <= 32'd0;
            infer_divisor       <= 32'd1;
            infer_prep_div_prod_q32 <= 64'd0;
            infer_div_valid     <= 1'b0;
            infer_delay_pipe_valid <= 1'b0;
            infer_skip_init_clear <= 1'b0;
            infer_force_no_input  <= 1'b0;
            infer_model_state_valid <= 1'b0;
            infer_spike_rd_addr <= 7'd0;
            snap_count_we <= 1'b0;
            snap_count_waddr <= '0;
            snap_count_wdata <= 16'd0;
            snap_count_raddr <= '0;
            csr_row_ptr_rd_addr <= '0;
            csr_col_idx_rd_addr <= '0;
            csr_col_idx_rd_addr_lane1 <= '0;
            csc_col_ptr_rd_addr <= '0;
            csc_row_idx_rd_addr <= '0;
            csc_edge_idx_rd_addr <= '0;
            train_label_sum_rd_addr <= 10'd0;
            train_label_count_rd_addr <= 4'd0;
            train_label_sum_wr_en <= 1'b0;
            train_label_sum_wr_addr <= 10'd0;
            train_label_sum_wr_data <= 32'd0;
            infer_poisson_thresh_rd_addr <= 10'd0;
            infer_poisson_thresh_wr_en <= 1'b0;
            infer_poisson_thresh_wr_addr <= 10'd0;
            infer_poisson_thresh_wr_data <= 12'd0;
            infer_pre_active_count <= 10'd0;
            infer_pre_wr_en <= 1'b0;
            infer_pre_wr_addr <= 10'd0;
            infer_pre_wr_data <= 10'd0;
            infer_pre_spike_wr_en <= 1'b0;
            infer_pre_spike_wr_addr <= 10'd0;
            infer_pre_spike_wr_data <= 1'b0;
            infer_pre_spike_rd_addr <= 10'd0;
            infer_pre_spike_rd_addr_lane1 <= 10'd0;
            infer_w_rd_addr_lane1 <= '0;
            infer_w_rd_data_q_lane1 <= 16'd0;
            infer_accum_pair_count <= 2'd0;
            infer_accum_lane0_fire <= 1'b0;
            infer_accum_lane1_fire <= 1'b0;
            memrd_pending <= 1'b0;
            memrd_wait    <= 1'b0;
            memrd_kind    <= MEMRD_NONE;
            // Large state arrays are cleared by a sequential init phase before inference
            // to reduce control sets and allow BRAM inference.
        end else begin
            tx_dv <= 1'b0;
            sd_rd <= 1'b0;
            sd_wr <= 1'b0;
            infer_div_valid <= 1'b0;
            train_stdp_div_valid <= 1'b0;
            raw_image0_wr_en <= 1'b0;
            train_label_sum_wr_en <= 1'b0;
            batch_label_wr_en <= 1'b0;
            batch_assign_wr_en <= 1'b0;
            infer_poisson_thresh_wr_en <= 1'b0;
            infer_pre_wr_en <= 1'b0;
            infer_pre_spike_wr_en <= 1'b0;
            infer_w_wr_en <= 1'b0;
            spike_count_we <= 1'b0;
            snap_count_we <= 1'b0;
            train_xin_wr_en <= 1'b0;
            train_xexc_wr_en <= 1'b0;
            train_xpost2_wr_en <= 1'b0;
            batch_status_word <= {batch_phase, batch_error_code, batch_error, batch_done, batch_active, batch_cfg1_valid, batch_cfg0_valid, 6'd0};
            batch_summary_word <= batch_summary_select(arg0[5:0]);
            train_busy_uart_blocked <= TRAIN_ENABLE &&
                                       (train_label_stats_active || train_chunk_active) &&
                                       (req_opcode != OP_BATCH_STATUS) &&
                                       (req_opcode != OP_BATCH_READ_SUMMARY);
            ddr_rsp_toggle_core_sync1 <= ddr_rsp_toggle_ddr;
            ddr_rsp_toggle_core_sync2 <= ddr_rsp_toggle_core_sync1;
            if (sd_copy_resp_pending && !response_ready) begin
                sd_copy_resp_pending <= 1'b0;
                resp_status <= sd_copy_resp_status;
                resp_result <= sd_copy_resp_result;
                resp_checksum <= 8'h00;
                response_ready <= 1'b1;
            end
            if (!ddr_req_pending_core_prev && ddr_req_pending_core) begin
                ddr_req_tag_expect_core <= ddr_req_tag_core;
                ddr_req_tag_core <= ddr_req_tag_core + 16'd1;
            end
            ddr_req_pending_core_prev <= ddr_req_pending_core;
            if (batch_active) begin
                batch_elapsed_cycles <= batch_elapsed_cycles + 64'd1;
                if ((batch_phase == BATCH_PHASE_LOADING) ||
                    sd_copy_active || imgload_active || imgload_start_pending ||
                    ddr_req_pending_core || ddr_rsp_drain_active_core) begin
                    batch_load_cycles <= batch_load_cycles + 64'd1;
                end else if (train_chunk_active) begin
                    batch_train_core_cycles <= batch_train_core_cycles + 64'd1;
                end else if (infer_active) begin
                    batch_infer_core_cycles <= batch_infer_core_cycles + 64'd1;
                end else if (train_label_stats_active) begin
                    batch_label_stats_cycles <= batch_label_stats_cycles + 64'd1;
                end else if (batch_infer_eval_active) begin
                    batch_infer_eval_cycles <= batch_infer_eval_cycles + 64'd1;
                end else begin
                    batch_other_cycles <= batch_other_cycles + 64'd1;
                end

                if (batch_cfg_mode_train && train_chunk_active) begin
                    if (infer_active) begin
                        if (infer_force_no_input) begin
                            batch_train_blank_infer_cycles <= batch_train_blank_infer_cycles + 64'd1;
                        end else begin
                            batch_train_inject_infer_cycles <= batch_train_inject_infer_cycles + 64'd1;
                        end
                        if (infer_state_is_evt_pre(infer_state)) begin
                            batch_train_evt_pre_cycles <= batch_train_evt_pre_cycles + 64'd1;
                        end else if (infer_state_is_evt_post(infer_state)) begin
                            batch_train_evt_post_cycles <= batch_train_evt_post_cycles + 64'd1;
                        end else if (infer_state_is_accum_core(infer_state)) begin
                            batch_train_accum_cycles <= batch_train_accum_cycles + 64'd1;
                        end
                    end else begin
                        case (train_chunk_state)
                            TCK_SNAP_COPY_INIT,
                            TCK_SNAP_COPY_WAIT,
                            TCK_SNAP_COPY_WRITE: batch_train_snap_cycles <= batch_train_snap_cycles + 64'd1;
                            TCK_REBASE_GIN_INIT,
                            TCK_REBASE_GIN_RUN: batch_train_rebase_cycles <= batch_train_rebase_cycles + 64'd1;
                            default: begin end
                        endcase
                    end
                end
            end
            if (ddr_rsp_drain_active_core && !ddr_req_pending_core) begin
                if (ddr_rsp_toggle_core_sync2 != ddr_rsp_toggle_core_seen) begin
                    ddr_rsp_toggle_core_seen <= ddr_rsp_toggle_core_sync2;
                    ddr_rsp_drain_quiet_core <= 2'd0;
                end else if (ddr_rsp_drain_quiet_core >= 2'd2) begin
                    ddr_rsp_drain_active_core <= 1'b0;
                    ddr_rsp_drain_quiet_core <= 2'd0;
                end else begin
                    ddr_rsp_drain_quiet_core <= ddr_rsp_drain_quiet_core + 2'd1;
                end
            end
            if (imgload_start_pending && !imgload_active && !ddr_req_pending_core && !ddr_rsp_drain_active_core) begin
                imgload_start_pending <= 1'b0;
                imgload_active <= 1'b1;
                imgload_addr_word <= imgload_start_addr_word;
                imgload_lane <= imgload_start_lane;
                imgload_byte_idx <= 10'd0;
                imgload_total_bytes <= imgload_start_total_bytes;
                imgload_sum_u8_accum <= 32'd0;
                imgload_word_valid <= 1'b0;
                imgload_word_data <= 32'd0;
                imgload_word_lane <= 2'd0;
                imgload_ddr_wait_counter <= 32'd0;
                if (!imgload_target_buf_sel) begin
                    raw_image0_valid <= 1'b0;
                    raw_image0_capture_idx <= 10'd0;
                    raw_image0_sum_u8 <= 32'd0;
                end else begin
                    raw_image1_valid <= 1'b0;
                    raw_image1_capture_idx <= 10'd0;
                    raw_image1_sum_u8 <= 32'd0;
                end
                raw_bytes_per_image <= {22'd0, imgload_start_total_bytes};
            end

            if (batch_active && (batch_phase == BATCH_PHASE_LOADING) &&
                !response_ready && !sd_copy_active && !imgload_active && !imgload_start_pending &&
                !raw_image_compute_valid && !ddr_req_pending_core && !ddr_rsp_drain_active_core) begin
                imgload_start_pending <= 1'b1;
                if (batch_use_cached_images) begin
                    imgload_start_addr_word <= IMG_CACHE_BASE_WORD + {2'b00, batch_cache_img_byte_off[31:2]};
                    imgload_start_lane <= batch_cache_img_byte_off[1:0];
                end else begin
                    imgload_start_addr_word <= IMG_STAGING_BASE_WORD + {2'b00, batch_img_byte_in_sector[31:2]};
                    imgload_start_lane <= batch_img_byte_in_sector[1:0];
                end
                imgload_start_total_bytes <= RAW1_BYTES_PER_IMAGE[9:0];
                imgload_target_buf_sel <= batch_compute_buf_sel;
                imgload_word_valid <= 1'b0;
                if (!batch_compute_buf_sel) begin
                    raw_image0_valid <= 1'b0;
                    raw_image0_capture_idx <= 10'd0;
                    raw_image0_sum_u8 <= 32'd0;
                end else begin
                    raw_image1_valid <= 1'b0;
                    raw_image1_capture_idx <= 10'd0;
                    raw_image1_sum_u8 <= 32'd0;
                end
            end

            if (batch_active && (batch_phase == BATCH_PHASE_LOADING) &&
                raw_image_compute_valid && !imgload_active && !imgload_start_pending &&
                !train_chunk_active && !infer_active && !response_ready) begin
                batch_phase <= BATCH_PHASE_RUNNING;
                if (batch_cfg_mode_train) begin
                    train_chunk_active       <= 1'b1;
                    train_chunk_mode         <= 3'd3;
                    train_chunk_state        <= TCK_INFER_START;
                    train_chunk_samples_left <= 16'd1;
                    train_chunk_steps_left   <= 16'd350;
                    train_chunk_seed_xin     <= 32'h13579BDF;
                    train_chunk_seed_xexc    <= 32'h2468ACE1;
                    train_chunk_winner       <= 7'd0;
                    train_chunk_pre_idx      <= 10'd0;
                    train_chunk_last_infer_spikes <= 32'd0;
                    train_chunk_last_blank_spikes <= 32'd0;
                    train_chunk_retry_curr_max_fr <= TRAIN_RETRY_MAX_FR_START;
                    train_chunk_retry_accepted_max_fr <= TRAIN_RETRY_MAX_FR_START;
                    train_chunk_retry_continue_infer <= 1'b0;
                    train_stdp_update_nt <= 32'd350;
                    infer_poisson_num_const_cfg <= POISSON_NUM_CONST;
                end else begin
                    infer_active       <= 1'b1;
                    infer_state        <= INFER_INIT_CLEAR;
                    infer_steps_target <= 32'd350;
                    infer_step_idx     <= 16'd0;
                    infer_neuron_idx   <= 7'd0;
                    infer_input_idx    <= 10'd0;
                    infer_prep_idx     <= 10'd0;
                    infer_accum        <= 32'sd0;
                    infer_accum_weight_phase <= 3'd0;
                    infer_apply_idx    <= 7'd0;
                    infer_trace_phase  <= 2'd0;
                    infer_total_spikes <= 32'd0;
                    infer_rng_state    <= batch_cfg_seed;
                    infer_skip_init_clear <= 1'b0;
                    infer_force_no_input  <= 1'b0;
                    infer_poisson_num_const_cfg <= POISSON_NUM_CONST;
                    infer_pre_active_count <= 10'd0;
                    raw_image0_rd_addr <= 10'd0;
                    infer_poisson_thresh_rd_addr <= 10'd0;
                    infer_model_state_valid <= 1'b0;
                end
            end

            if (batch_active && !batch_cfg_mode_train && !batch_use_cached_images && (batch_phase == BATCH_PHASE_RUNNING) &&
                infer_active && !batch_prefetch_active && !batch_prefetch_ready &&
                ((batch_processed_samples + 32'd1) < batch_cfg_num_samples) &&
                !sd_copy_active && !imgload_active && !imgload_start_pending &&
                !ddr_req_pending_core && !ddr_rsp_drain_active_core && !response_ready) begin
                logic [31:0] next_img_byte_off_tmp;
                logic [31:0] next_img_byte_in_sector_tmp;
                logic [31:0] next_img_byte_in_sector_norm_tmp;
                logic [31:0] next_img_sector_carry_tmp;
                logic [31:0] next_img_sector_off_tmp;
                logic [31:0] next_img_sectors_needed_tmp;
                next_img_byte_off_tmp = batch_img_byte_off + RAW1_BYTES_PER_IMAGE;
                next_img_byte_in_sector_tmp = batch_img_byte_in_sector + RAW1_BYTES_PER_IMAGE;
                if (next_img_byte_in_sector_tmp >= 32'd1024) begin
                    next_img_sector_carry_tmp = 32'd2;
                    next_img_byte_in_sector_norm_tmp = next_img_byte_in_sector_tmp - 32'd1024;
                end else if (next_img_byte_in_sector_tmp >= 32'd512) begin
                    next_img_sector_carry_tmp = 32'd1;
                    next_img_byte_in_sector_norm_tmp = next_img_byte_in_sector_tmp - 32'd512;
                end else begin
                    next_img_sector_carry_tmp = 32'd0;
                    next_img_byte_in_sector_norm_tmp = next_img_byte_in_sector_tmp;
                end
                next_img_sector_off_tmp = batch_img_sector_off + next_img_sector_carry_tmp;
                next_img_sectors_needed_tmp = (next_img_byte_in_sector_norm_tmp <= 32'd240) ? 32'd2 : 32'd3;
                batch_prefetch_active <= 1'b1;
                batch_prefetch_issue_pending <= 1'b1;
                batch_prefetch_ready <= 1'b0;
                batch_prefetch_sample_idx <= batch_current_sample_idx + 32'd1;
                batch_prefetch_img_byte_off <= next_img_byte_off_tmp;
                batch_prefetch_img_sector_off <= next_img_sector_off_tmp;
                batch_prefetch_img_byte_in_sector <= next_img_byte_in_sector_norm_tmp;
                batch_prefetch_img_sectors_needed <= next_img_sectors_needed_tmp;
                imgload_target_buf_sel <= batch_fill_buf_sel;
            end

            if (batch_prefetch_active && batch_prefetch_issue_pending &&
                !batch_prefetch_ready &&
                !sd_copy_active && !imgload_active && !imgload_start_pending &&
                !ddr_req_pending_core && !ddr_rsp_drain_active_core && !response_ready) begin
                batch_prefetch_issue_pending <= 1'b0;
                sd_copy_active        <= 1'b1;
                sd_in_read            <= 1'b0;
                sd_copy_lba           <= batch_cfg_start_lba + batch_prefetch_img_sector_off;
                sd_copy_sectors_left  <= batch_prefetch_img_sectors_needed;
                sd_byte_count         <= 9'd0;
                sd_pack_idx           <= 2'd0;
                sd_pack_word          <= 32'd0;
                sd_copy_words_written <= 32'd0;
                sd_sector_buf_ready   <= 2'b00;
                sd_header_done        <= 1'b0;
                sd_file_total_bytes   <= 32'd0;
                sd_file_bytes_seen    <= 32'd0;
                sd_copy_done_pending  <= 1'b0;
                sd_use_sector_limit   <= 1'b1;
                sd_copy_raw1_mode     <= 1'b0;
                sd_copy_dest_base_word <= IMG_STAGING_BASE_WORD;
            end

            if (batch_prefetch_active && !batch_prefetch_issue_pending && !batch_prefetch_ready &&
                !response_ready && !sd_copy_active && !imgload_active && !imgload_start_pending &&
                !raw_image_fill_valid && !ddr_req_pending_core && !ddr_rsp_drain_active_core) begin
                imgload_start_pending <= 1'b1;
                imgload_start_addr_word <= IMG_STAGING_BASE_WORD + {2'b00, batch_prefetch_img_byte_in_sector[31:2]};
                imgload_start_lane <= batch_prefetch_img_byte_in_sector[1:0];
                imgload_start_total_bytes <= RAW1_BYTES_PER_IMAGE[9:0];
                imgload_target_buf_sel <= batch_fill_buf_sel;
                imgload_word_valid <= 1'b0;
                if (!batch_fill_buf_sel) begin
                    raw_image0_valid <= 1'b0;
                    raw_image0_capture_idx <= 10'd0;
                    raw_image0_sum_u8 <= 32'd0;
                end else begin
                    raw_image1_valid <= 1'b0;
                    raw_image1_capture_idx <= 10'd0;
                    raw_image1_sum_u8 <= 32'd0;
                end
            end

            if (ddr_req_pending_core && (ddr_req_kind_core == DDR_REQ_IMGLOAD) &&
                !ddr_rsp_capture_pending_core && !response_ready) begin
                if (imgload_ddr_wait_counter >= (IMGLOAD_DDR_WAIT_TIMEOUT_CLKS - 1)) begin
                    ddr_req_pending_core <= 1'b0;
                    ddr_req_from_imgload_core <= 1'b0;
                    ddr_req_kind_core <= DDR_REQ_NONE;
                    imgload_active <= 1'b0;
                    imgload_word_valid <= 1'b0;
                    imgload_start_pending <= 1'b0;
                    imgload_ddr_wait_counter <= 32'd0;
                    resp_status    <= STATUS_BAD_PACKET;
                    resp_result    <= 32'sd0;
                    resp_checksum  <= 8'h00;
                    response_ready <= 1'b1;
                end else begin
                    imgload_ddr_wait_counter <= imgload_ddr_wait_counter + 32'd1;
                end
            end else begin
                imgload_ddr_wait_counter <= 32'd0;
            end
            if (ddr_req_pending_core && !response_ready &&
                !ddr_rsp_capture_pending_core &&
                !ddr_rsp_drain_active_core &&
                (ddr_rsp_toggle_core_sync2 != ddr_rsp_toggle_core_seen)) begin
                ddr_rsp_toggle_core_seen <= ddr_rsp_toggle_core_sync2;
                ddr_rsp_capture_pending_core <= 1'b1;
                ddr_rsp_payload_ready_core <= 1'b0;
                ddr_rsp_payload_settle_core <= 2'd3;
                ddr_rsp_kind_core <= ddr_req_kind_core;
            end
            if (ddr_req_pending_core && !response_ready && ddr_rsp_capture_pending_core) begin
                if (!ddr_rsp_payload_ready_core) begin
                    if (ddr_rsp_payload_settle_core != 2'd0) begin
                        ddr_rsp_payload_settle_core <= ddr_rsp_payload_settle_core - 2'd1;
                    end else if (ddr_rsp_payload_core_sync[47:32] == ddr_req_tag_expect_core) begin
                        ddr_resp_rdata_core <= ddr_rsp_payload_core_sync[31:0];
                        ddr_resp_status_core <= ddr_rsp_payload_core_sync[49] ? STATUS_OK : STATUS_BAD_PACKET;
                        ddr_resp_was_write_core <= ddr_rsp_payload_core_sync[48];
                        ddr_rsp_payload_ready_core <= 1'b1;
                    end
                end else begin
                    ddr_rsp_capture_pending_core <= 1'b0;
                    ddr_rsp_payload_ready_core <= 1'b0;
                    ddr_rsp_payload_settle_core <= 2'd0;
                    if ((ddr_rsp_kind_core == DDR_REQ_IMGLOAD) && ddr_resp_was_write_core) begin
                    // Ignore stale write ACK while waiting for an imgload read response.
                end else begin
                    ddr_req_pending_core <= 1'b0;
                    ddr_rsp_kind_core <= DDR_REQ_NONE;
                    ddr_req_kind_core <= DDR_REQ_NONE;
                if (ddr_rsp_kind_core == DDR_REQ_SD) begin
                    ddr_req_from_sd_core <= 1'b0;
                    if (ddr_resp_status_core != STATUS_OK) begin
                        sd_ddr_flush_active <= 1'b0;
                        sd_copy_active <= 1'b0;
                        sd_in_read <= 1'b0;
                        resp_status    <= STATUS_BAD_PACKET;
                        resp_result    <= 32'sd0;
                        resp_checksum  <= 8'h00;
                        response_ready <= 1'b1;
                    end else begin
                        if ((sd_ddr_flush_idx + {5'd0, ddr_req_word_count_core}) >= sd_sector_words_queued_bank[sd_flush_bank]) begin
                            sd_ddr_flush_active <= 1'b0;
                            sd_ddr_flush_idx <= 8'd0;
                            sd_sector_buf_ready[sd_flush_bank] <= 1'b0;
                            if (sd_copy_done_pending && !sd_in_read &&
                                ((sd_flush_bank == 1'b0 && !sd_sector_buf_ready[1]) ||
                                 (sd_flush_bank == 1'b1 && !sd_sector_buf_ready[0]))) begin
                                sd_copy_active <= 1'b0;
                                ddr_rsp_drain_active_core <= 1'b1;
                                ddr_rsp_drain_quiet_core <= 2'd0;
                                ddr_rsp_toggle_core_seen <= ddr_rsp_toggle_core_sync2;
                                sd_copy_resp_pending <= 1'b1;
                                sd_copy_resp_status <= STATUS_OK;
                                sd_copy_resp_result <= sd_copy_words_written;
                            end
                        end else begin
                            sd_ddr_flush_idx <= sd_ddr_flush_idx + {5'd0, ddr_req_word_count_core};
                        end
                    end
                end else if (ddr_rsp_kind_core == DDR_REQ_IMGLOAD) begin
                    ddr_req_from_imgload_core <= 1'b0;
                    if (ddr_resp_status_core != STATUS_OK) begin
                        imgload_active <= 1'b0;
                        imgload_word_valid <= 1'b0;
                        resp_status    <= STATUS_BAD_PACKET;
                        resp_result    <= 32'sd0;
                        resp_checksum  <= 8'h00;
                        response_ready <= 1'b1;
                    end else begin
                        imgload_word_valid <= 1'b1;
                        imgload_word_data  <= ddr_resp_rdata_core;
                        imgload_word_lane  <= imgload_lane;
                        imgload_addr_word <= imgload_addr_word + 32'd1;
                        imgload_lane <= 2'd0;
                    end
                end else if (TRAIN_ENABLE && (ddr_rsp_kind_core == DDR_REQ_TRAIN)) begin
                    ddr_req_from_train_core <= 1'b0;
                    if (ddr_resp_status_core != STATUS_OK) begin
                        train_trace_active <= 1'b0;
                        train_trace_state <= TRK_IDLE;
                        train_trace_use_infer_prelist <= 1'b0;
                        train_trace_skip_a <= 1'b0;
                        train_trace_skip_b <= 1'b0;
                        train_trace_multi_post_active <= 1'b0;
                        train_trace_post_scan_idx <= 7'd0;
                        train_stdp_active <= 1'b0;
                        train_stdp_state  <= TSK_IDLE;
                        train_stdp_batch_active <= 1'b0;
                        train_chunk_active <= 1'b0;
                        train_chunk_state <= TCK_IDLE;
                        train_gen_active  <= 1'b0;
                        train_gen_state   <= TGK_IDLE;
                        train_mem_init_active <= 1'b0;
                        train_mem_init_state <= TMI_IDLE;
                        resp_status    <= STATUS_BAD_PACKET;
                        resp_result    <= 32'sd0;
                        resp_checksum  <= 8'h00;
                        response_ready <= 1'b1;
                    end else if (train_mem_init_active) begin
                        case (train_mem_init_state)
                            TMI_A_WRITE_WAIT: begin
                                if (train_mem_init_idx == (N_WEIGHTS - 1)) begin
                                    train_mem_init_idx   <= '0;
                                    train_mem_init_state <= TMI_BT_WRITE_REQ;
                                end else begin
                                    train_mem_init_idx   <= train_mem_init_idx + {{(TRAIN_DENSE_ADDR_W-1){1'b0}}, 1'b1};
                                    train_mem_init_state <= TMI_A_WRITE_REQ;
                                end
                            end
                            TMI_BT_WRITE_WAIT: begin
                                if (train_mem_init_idx == (N_WEIGHTS - 1)) begin
                                    train_mem_init_state <= TMI_DONE;
                                end else begin
                                    train_mem_init_idx   <= train_mem_init_idx + {{(TRAIN_DENSE_ADDR_W-1){1'b0}}, 1'b1};
                                    train_mem_init_state <= TMI_BT_WRITE_REQ;
                                end
                            end
                            default: begin
                                train_mem_init_active <= 1'b0;
                                train_mem_init_state <= TMI_IDLE;
                                resp_status    <= STATUS_BAD_PACKET;
                                resp_result    <= 32'sd0;
                                resp_checksum  <= 8'h00;
                                response_ready <= 1'b1;
                            end
                        endcase
                    end else if (train_trace_active) begin
                        case (train_trace_state)
                            TRK_A_READ_X_WAIT: begin
                                train_tmp_x_val <= ddr_resp_rdata_core;
                                train_trace_state <= TRK_A_READ_A_REQ;
                            end
                            TRK_A_READ_A_WAIT: begin
                                train_tmp_mem_val <= ddr_resp_rdata_core;
                                train_trace_state <= TRK_A_WRITE_A_REQ;
                            end
                            TRK_A_WRITE_A_WAIT: begin
                                if (train_a_idx == (N_IN - 1)) begin
                                    if (train_trace_skip_b) begin
                                        train_trace_state <= TRK_DONE;
                                    end else begin
                                        train_pre_idx <= 10'd0;
                                        train_b_col_idx <= 7'd0;
                                        train_trace_state <= TRK_B_READ_PRE_REQ;
                                    end
                                end else begin
                                    train_a_idx <= train_a_idx + 10'd1;
                                    train_trace_state <= TRK_A_READ_X_REQ;
                                end
                            end
                            TRK_B_READ_PRE_WAIT: begin
                                train_curr_pre <= ddr_resp_rdata_core[9:0];
                                train_trace_bt_pre_base <= TRAIN_BASE_BT_Q16_WORDS + ({22'd0, ddr_resp_rdata_core[9:0]} * N_NEURONS);
                                train_b_col_idx <= 7'd0;
                                train_trace_state <= TRK_B_READ_X_REQ;
                            end
                            TRK_B_READ_X_WAIT: begin
                                train_tmp_x_val <= ddr_resp_rdata_core;
                                train_trace_state <= TRK_B_READ_BT_REQ;
                            end
                            TRK_B_READ_BT_WAIT: begin
                                train_tmp_mem_val <= ddr_resp_rdata_core;
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
                                train_trace_use_infer_prelist <= 1'b0;
                                train_trace_skip_a <= 1'b0;
                                train_trace_skip_b <= 1'b0;
                                train_trace_multi_post_active <= 1'b0;
                                resp_status    <= STATUS_BAD_PACKET;
                                resp_result    <= 32'sd0;
                                resp_checksum  <= 8'h00;
                                response_ready <= 1'b1;
                            end
                        endcase
                    end else if (train_stdp_active) begin
                        case (train_stdp_state)
                            TSK_READ_A_WAIT: begin
                                train_stdp_a_val <= $signed(ddr_resp_rdata_core);
                                train_stdp_state <= TSK_READ_BT_REQ;
                            end
                            TSK_READ_BT_WAIT: begin
                                train_stdp_bt_val <= $signed(ddr_resp_rdata_core);
                                train_stdp_state <= TSK_DIV_NORM_START;
                            end
                            TSK_CLR_A_WAIT: begin
                                train_stdp_state <= TSK_CLR_BT_REQ;
                            end
                            TSK_CLR_BT_WAIT: begin
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
                                        train_stdp_state   <= TSK_READ_W_REQ;
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
                                resp_checksum  <= 8'h00;
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
                                resp_checksum  <= 8'h00;
                                response_ready <= 1'b1;
                            end
                        endcase
                    end else begin
                        resp_status    <= STATUS_BAD_PACKET;
                        resp_result    <= 32'sd0;
                        resp_checksum  <= 8'h00;
                        response_ready <= 1'b1;
                    end
                end else begin
                    // Generic DDR read/write response path (non-SD, non-imgload, non-train-kernel).
                    resp_status    <= ddr_resp_status_core;
                    resp_result    <= ddr_resp_rdata_core;
                    resp_checksum  <= 8'h00;
                    response_ready <= 1'b1;
                end
            end
            end
            end
            if (response_ready || (rx_state == RX_WAIT_SYNC)) begin
                rx_timeout_counter <= '0;
            end else if (rx_dv) begin
                rx_timeout_counter <= '0;
            end else if (rx_timeout_counter >= RX_TIMEOUT_CLKS - 1) begin
                rx_state           <= RX_WAIT_SYNC;
                req_checksum_accum <= 8'h00;
                arg_byte_idx       <= 3'd0;
                args_seen          <= 3'd0;
                rx_timeout_counter <= '0;
                resp_status        <= STATUS_BAD_PACKET;
                resp_result        <= 32'sd0;
                resp_checksum  <= 8'h00;
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
                            resp_result    <= {16'd0, infer_spike_rd_data};
                            resp_checksum  <= 8'h00;
                            response_ready <= 1'b1;
                        end
                        MEMRD_TRAIN_LABEL_STAT_SUM: begin
                            resp_status    <= STATUS_OK;
                            resp_result    <= train_label_sum_rd_data;
                            resp_checksum  <= 8'h00;
                            response_ready <= 1'b1;
                        end
                        MEMRD_TRAIN_LABEL_STAT_COUNT: begin
                            resp_status    <= STATUS_OK;
                            resp_result    <= train_label_count_rd_data;
                            resp_checksum  <= 8'h00;
                            response_ready <= 1'b1;
                        end
                        default: begin
                            resp_status    <= STATUS_BAD_PACKET;
                            resp_result    <= 32'sd0;
                            resp_checksum  <= 8'h00;
                            response_ready <= 1'b1;
                        end
                    endcase
                    memrd_kind <= MEMRD_NONE;
                end
            end

            if (sd_ddr_flush_active && !ddr_req_pending_core && !response_ready && sd_copy_active) begin
                if (sd_ddr_flush_idx < sd_sector_words_queued_bank[sd_flush_bank]) begin
                    logic [2:0] words_this_req;
                    // Use single-word requests here to keep CDC/request datapath narrow and
                    // reduce routing pressure during implementation.
                    words_this_req = 3'd1;

                    ddr_req_pending_core   <= 1'b1;
                    ddr_req_we_core        <= 1'b1;
                    ddr_req_from_sd_core <= 1'b1;
                    ddr_req_kind_core <= DDR_REQ_SD;
                    ddr_req_from_imgload_core <= 1'b0;
                    ddr_req_from_train_core <= 1'b0;
                    ddr_req_addr_word_core <= sd_sector_ddr_base_word_bank[sd_flush_bank] + {24'd0, sd_ddr_flush_idx};
                    ddr_req_wdata_core     <= sd_sector_word_buf[sd_flush_bank][sd_ddr_flush_idx];
                    ddr_req_wide_core      <= 1'b0;
                    ddr_req_wdata128_core  <= 128'd0;
                    ddr_req_sel16_core     <= 16'd0;
                    ddr_req_word_count_core <= words_this_req;
                    ddr_req_toggle_core    <= ~ddr_req_toggle_core;
                end else begin
                    sd_ddr_flush_active <= 1'b0;
                end
            end

            if (imgload_active && imgload_word_valid && !response_ready) begin
                if (imgload_byte_idx < imgload_total_bytes) begin
                    logic [7:0] imgload_curr_byte;
                    imgload_curr_byte = lane_byte_sel(imgload_word_data, imgload_word_lane);
                    raw_image0_wr_en   <= 1'b1;
                    raw_image0_wr_addr <= imgload_byte_idx;
                    raw_image0_wr_data <= imgload_curr_byte;
                    imgload_sum_u8_accum <= imgload_sum_u8_accum + {24'd0, imgload_curr_byte};
                    if ((imgload_byte_idx + 10'd1) >= imgload_total_bytes) begin
                        imgload_active <= 1'b0;
                        imgload_word_valid <= 1'b0;
                        imgload_byte_idx <= imgload_byte_idx + 10'd1;
                        if (!imgload_target_buf_sel) begin
                            raw_image0_valid <= 1'b1;
                            raw_image0_capture_idx <= imgload_byte_idx + 10'd1;
                            raw_image0_sum_u8 <= imgload_sum_u8_accum + {24'd0, imgload_curr_byte};
                        end else begin
                            raw_image1_valid <= 1'b1;
                            raw_image1_capture_idx <= imgload_byte_idx + 10'd1;
                            raw_image1_sum_u8 <= imgload_sum_u8_accum + {24'd0, imgload_curr_byte};
                        end
                        if (batch_prefetch_active && (imgload_target_buf_sel == batch_fill_buf_sel)) begin
                            batch_prefetch_active <= 1'b0;
                            batch_prefetch_issue_pending <= 1'b0;
                            batch_prefetch_ready <= 1'b1;
                        end
                        if (!batch_active) begin
                            resp_status    <= STATUS_OK;
                            resp_result    <= imgload_byte_idx + 10'd1;
                            resp_checksum  <= 8'h00;
                            response_ready <= 1'b1;
                        end
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

            if (imgload_active && !imgload_word_valid && !ddr_req_pending_core && !response_ready &&
                !sd_ddr_flush_active) begin
                if (imgload_byte_idx < imgload_total_bytes) begin
                    if (imgload_addr_word < DDR_ADDR_WORD_LIMIT) begin
                        ddr_rsp_toggle_core_seen <= ddr_rsp_toggle_core_sync2;
                        ddr_rsp_capture_pending_core <= 1'b0;
                        ddr_rsp_payload_ready_core <= 1'b0;
                        ddr_req_pending_core    <= 1'b1;
                        ddr_req_we_core         <= 1'b0;
                        ddr_req_from_sd_core    <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b1;
                        ddr_req_kind_core <= DDR_REQ_IMGLOAD;
                        ddr_req_from_train_core <= 1'b0;
                        ddr_req_addr_word_core  <= imgload_addr_word;
                        ddr_req_wdata_core      <= 32'd0;
                        ddr_req_wide_core       <= 1'b0;
                        ddr_req_wdata128_core   <= 128'd0;
                        ddr_req_sel16_core      <= 16'd0;
                        ddr_req_word_count_core <= 3'd1;
                        ddr_req_toggle_core     <= ~ddr_req_toggle_core;
                    end else begin
                        imgload_active <= 1'b0;
                        imgload_word_valid <= 1'b0;
                        resp_status    <= STATUS_BAD_PACKET;
                        resp_result    <= 32'sd0;
                        resp_checksum  <= 8'h00;
                        response_ready <= 1'b1;
                    end
                end
            end

            if (TRAIN_ENABLE && train_trace_active && !ddr_req_pending_core && !response_ready && !sd_ddr_flush_active && !imgload_word_valid && !train_stdp_active) begin
                case (train_trace_state)
                    TRK_A_READ_X_REQ: begin
                        if (train_trace_skip_a) begin
                            train_pre_idx <= 10'd0;
                            train_b_col_idx <= 7'd0;
                            train_trace_state <= TRK_B_READ_PRE_REQ;
                        end else if (train_xin_cache_valid) begin
                            train_xin_rd_addr <= train_a_idx;
                            train_trace_state <= TRK_A_READ_X_WAIT;
                        end else begin
                            ddr_req_pending_core    <= 1'b1;
                            ddr_req_we_core         <= 1'b0;
                            ddr_req_from_sd_core    <= 1'b0;
                            ddr_req_from_imgload_core <= 1'b0;
                            ddr_req_from_train_core <= 1'b1;
                            ddr_req_kind_core <= DDR_REQ_TRAIN;
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
                        ddr_req_kind_core <= DDR_REQ_TRAIN;
                        ddr_req_addr_word_core  <= train_trace_a_row_base + {22'd0, train_a_idx};
                        ddr_req_wdata_core      <= 32'd0;
                        ddr_req_wide_core       <= 1'b0;
                        ddr_req_wdata128_core   <= 128'd0;
                        ddr_req_sel16_core      <= 16'd0;
                        ddr_req_word_count_core <= 3'd1;
                        ddr_req_toggle_core     <= ~ddr_req_toggle_core;
                        train_trace_state       <= TRK_A_READ_A_WAIT;
                    end
                    TRK_A_READ_X_WAIT: begin
                        train_tmp_x_val <= train_xin_rd_data;
                        train_trace_state <= TRK_A_READ_A_REQ;
                    end
                    TRK_A_WRITE_A_REQ: begin
                        ddr_req_pending_core    <= 1'b1;
                        ddr_req_we_core         <= 1'b1;
                        ddr_req_from_sd_core    <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core <= 1'b1;
                        ddr_req_kind_core <= DDR_REQ_TRAIN;
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
                        end else if (train_trace_use_infer_prelist) begin
                            infer_pre_rd_addr <= train_pre_idx;
                            train_trace_state <= TRK_B_READ_PRE_BRAM_WAIT;
                        end else begin
                            ddr_req_pending_core    <= 1'b1;
                            ddr_req_we_core         <= 1'b0;
                            ddr_req_from_sd_core    <= 1'b0;
                            ddr_req_from_imgload_core <= 1'b0;
                            ddr_req_from_train_core <= 1'b1;
                            ddr_req_kind_core <= DDR_REQ_TRAIN;
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
                    TRK_B_READ_PRE_BRAM_WAIT: begin
                        train_curr_pre <= infer_pre_rd_data;
                        train_trace_bt_pre_base <= TRAIN_BASE_BT_Q16_WORDS + ({22'd0, infer_pre_rd_data} * N_NEURONS);
                        train_b_col_idx <= 7'd0;
                        train_trace_state <= TRK_B_READ_X_REQ;
                    end
                    TRK_B_READ_X_REQ: begin
                        if (train_xexc_cache_valid) begin
                            train_xexc_rd_addr <= train_b_col_idx;
                            train_trace_state <= TRK_B_READ_X_WAIT;
                        end else begin
                            ddr_req_pending_core    <= 1'b1;
                            ddr_req_we_core         <= 1'b0;
                            ddr_req_from_sd_core    <= 1'b0;
                            ddr_req_from_imgload_core <= 1'b0;
                            ddr_req_from_train_core <= 1'b1;
                            ddr_req_kind_core <= DDR_REQ_TRAIN;
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
                        ddr_req_kind_core <= DDR_REQ_TRAIN;
                        ddr_req_addr_word_core  <= train_trace_bt_pre_base + {25'd0, train_b_col_idx};
                        ddr_req_wdata_core      <= 32'd0;
                        ddr_req_wide_core       <= 1'b0;
                        ddr_req_wdata128_core   <= 128'd0;
                        ddr_req_sel16_core      <= 16'd0;
                        ddr_req_word_count_core <= 3'd1;
                        ddr_req_toggle_core     <= ~ddr_req_toggle_core;
                        train_trace_state       <= TRK_B_READ_BT_WAIT;
                    end
                    TRK_B_READ_X_WAIT: begin
                        train_tmp_x_val <= train_xexc_rd_data;
                        train_trace_state <= TRK_B_READ_BT_REQ;
                    end
                    TRK_B_WRITE_BT_REQ: begin
                        ddr_req_pending_core    <= 1'b1;
                        ddr_req_we_core         <= 1'b1;
                        ddr_req_from_sd_core    <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core <= 1'b1;
                        ddr_req_kind_core <= DDR_REQ_TRAIN;
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
                        train_trace_use_infer_prelist <= 1'b0;
                        train_trace_skip_a <= 1'b0;
                        train_trace_skip_b <= 1'b0;
                        if (!train_chunk_active) begin
                            resp_status    <= STATUS_OK;
                            resp_result    <= {22'd0, train_pre_count};
                            resp_checksum  <= 8'h00;
                            response_ready <= 1'b1;
                        end
                    end
                    default: begin end
                endcase
            end

            if (TRAIN_ENABLE && train_stdp_active && !ddr_req_pending_core && !response_ready && !sd_ddr_flush_active && !imgload_word_valid && !train_trace_active) begin
                case (train_stdp_state)
                    TSK_SUM_READ_W_REQ: begin
                        // Legacy normalization pass removed for Brian2-aligned STDP.
                        train_stdp_state <= TSK_READ_W_REQ;
                    end
                    TSK_SUM_READ_W_WAIT: begin
                        train_stdp_state <= TSK_READ_W_REQ;
                    end
                    TSK_READ_W_REQ: begin
                        infer_w_rd_addr           <= train_stdp_w_row_base[W_ADDR_W-1:0] + {7'd0, train_stdp_col_idx};
                        train_stdp_state          <= TSK_READ_W_WAIT;
                    end
                    TSK_READ_W_WAIT: begin
                        train_stdp_w_val <= $signed({16'd0, infer_w_rd_data});
                        train_stdp_state <= TSK_READ_A_REQ;
                    end
                    TSK_READ_A_REQ: begin
                        ddr_req_pending_core      <= 1'b1;
                        ddr_req_we_core           <= 1'b0;
                        ddr_req_from_sd_core      <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core <= 1'b1;
                        ddr_req_kind_core <= DDR_REQ_TRAIN;
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
                        ddr_req_from_train_core <= 1'b1;
                        ddr_req_kind_core <= DDR_REQ_TRAIN;
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
                        train_stdp_w_norm_q16 <= train_stdp_w_val;
                        train_stdp_state <= TSK_DIV_DW_PREP;
                    end
                    TSK_DIV_NORM_WAIT: begin
                        train_stdp_state <= TSK_DIV_DW_PREP;
                    end
                    TSK_DIV_DW_PREP: begin
                        if ((train_stdp_a_val == 32'sd0) && (train_stdp_bt_val == 32'sd0)) begin
                            train_stdp_pot_mid_q16 <= 32'sd0;
                            train_stdp_dep_mid_q16 <= 32'sd0;
                            train_stdp_pot_term_q16 <= 32'sd0;
                            train_stdp_dep_term_q16 <= 32'sd0;
                            train_stdp_dW_q16 <= 32'sd0;
                            train_stdp_state <= TSK_DIV_DW_PIPE;
                        end else begin
                            // Brian2 minimal STDP:
                            //   on_pre : w -= 0.0001 * post1
                            //   on_post: w += 0.01   * pre * post2_before
                            // A and BT already hold the accumulated pre/post trace terms for this batch step.
                            train_stdp_pot_mid_prod_q32 <= $signed(TRAIN_LR_P_Q16) * 32'sd65536;
                            train_stdp_dep_mid_prod_q32 <= $signed(TRAIN_LR_M_Q16) * 32'sd65536;
                            train_stdp_state <= TSK_DIV_DW_PREP_MUL;
                        end
                    end
                    TSK_DIV_DW_PREP_MUL: begin
                        // keep a separate stage so the multiplier output does not feed
                        // directly into round/compare in one cycle.
                        train_stdp_state <= TSK_DIV_DW_PREP_ROUND;
                    end
                    TSK_DIV_DW_PREP_ROUND: begin
                        if (train_stdp_pot_mid_prod_q32 >= 0) begin
                            train_stdp_pot_mid_q16 <= $signed((train_stdp_pot_mid_prod_q32 + 64'sd32768) >>> 16);
                        end else begin
                            train_stdp_pot_mid_q16 <= $signed((train_stdp_pot_mid_prod_q32 - 64'sd32768) >>> 16);
                        end
                        if (train_stdp_dep_mid_prod_q32 >= 0) begin
                            train_stdp_dep_mid_q16 <= $signed((train_stdp_dep_mid_prod_q32 + 64'sd32768) >>> 16);
                        end else begin
                            train_stdp_dep_mid_q16 <= $signed((train_stdp_dep_mid_prod_q32 - 64'sd32768) >>> 16);
                        end
                        train_stdp_state <= TSK_DIV_DW_PIPE;
                    end
                    TSK_DIV_DW_PIPE: begin
                        train_stdp_pot_mid_pipe_q16 <= train_stdp_pot_mid_q16;
                        train_stdp_dep_mid_pipe_q16 <= train_stdp_dep_mid_q16;
                        train_stdp_a_val_pipe <= train_stdp_a_val;
                        train_stdp_bt_val_pipe <= train_stdp_bt_val;
                        train_stdp_state <= TSK_DIV_DW_TERM;
                    end
                    TSK_DIV_DW_TERM: begin
                        train_stdp_pot_prod_q32 <= $signed(train_stdp_pot_mid_pipe_q16) * $signed(train_stdp_a_val_pipe);
                        train_stdp_dep_prod_q32 <= $signed(train_stdp_dep_mid_pipe_q16) * $signed(train_stdp_bt_val_pipe);
                        train_stdp_state <= TSK_DIV_DW_TERM_ROUND;
                    end
                    TSK_DIV_DW_TERM_ROUND: begin
                        if (train_stdp_pot_prod_q32 >= 0) begin
                            train_stdp_pot_term_q16 <= $signed((train_stdp_pot_prod_q32 + 64'sd32768) >>> 16);
                        end else begin
                            train_stdp_pot_term_q16 <= $signed((train_stdp_pot_prod_q32 - 64'sd32768) >>> 16);
                        end
                        if (train_stdp_dep_prod_q32 >= 0) begin
                            train_stdp_dep_term_q16 <= $signed((train_stdp_dep_prod_q32 + 64'sd32768) >>> 16);
                        end else begin
                            train_stdp_dep_term_q16 <= $signed((train_stdp_dep_prod_q32 - 64'sd32768) >>> 16);
                        end
                        train_stdp_state <= TSK_DIV_DW_COMB;
                    end
                    TSK_DIV_DW_COMB: begin
                        train_stdp_dW_q16 <= train_stdp_pot_term_q16 - train_stdp_dep_term_q16;
                        train_stdp_state <= TSK_DIV_DW_ABS;
                    end
                    TSK_DIV_DW_ABS: begin
                        if (train_stdp_dW_q16 < 0)
                            train_stdp_dW_abs <= $unsigned(-train_stdp_dW_q16);
                        else
                            train_stdp_dW_abs <= $unsigned(train_stdp_dW_q16);
                        train_stdp_state <= TSK_DIV_DW_START;
                    end
                    TSK_DIV_DW_START: begin
                        if (train_stdp_dW_q16 == 32'sd0) begin
                            train_stdp_w_new <= train_stdp_w_norm_q16;
                            train_stdp_state <= TSK_WRITE_W_REQ;
                        end else begin
                            train_stdp_state <= TSK_DIV_DW_WNEXT;
                        end
                    end
                    TSK_DIV_DW_WAIT: begin
                        train_stdp_state <= TSK_DIV_DW_WNEXT;
                    end
                    TSK_DIV_DW_CLIP: begin
                        train_stdp_state <= TSK_DIV_DW_WNEXT;
                    end
                    TSK_DIV_DW_WNEXT: begin
                        logic signed [31:0] w_next_q16_tmp;
                        w_next_q16_tmp = train_stdp_w_norm_q16 + train_stdp_dW_q16;
                        if (w_next_q16_tmp > TRAIN_WMAX_Q16)
                            w_next_q16_tmp = TRAIN_WMAX_Q16;
                        else if (w_next_q16_tmp < TRAIN_WMIN_Q16)
                            w_next_q16_tmp = TRAIN_WMIN_Q16;
                        train_stdp_w_new <= w_next_q16_tmp;
                        train_stdp_state <= TSK_WRITE_W_REQ;
                    end
                    TSK_WRITE_W_REQ: begin
                        infer_w_wr_en             <= 1'b1;
                        infer_w_wr_addr           <= train_stdp_w_row_base[W_ADDR_W-1:0] + {7'd0, train_stdp_col_idx};
                        infer_w_wr_data           <= train_stdp_w_new[15:0];
                        train_stdp_state          <= TSK_WRITE_W_WAIT;
                    end
                    TSK_WRITE_W_WAIT: begin
                        train_stdp_state <= TSK_CLR_A_REQ;
                    end
                    TSK_CLR_A_REQ: begin
                        ddr_req_pending_core      <= 1'b1;
                        ddr_req_we_core           <= 1'b1;
                        ddr_req_from_sd_core      <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core <= 1'b1;
                        ddr_req_kind_core <= DDR_REQ_TRAIN;
                        ddr_req_addr_word_core    <= train_stdp_a_row_base + {22'd0, train_stdp_col_idx};
                        ddr_req_wdata_core        <= 32'd0;
                        ddr_req_wide_core         <= 1'b0;
                        ddr_req_wdata128_core     <= 128'd0;
                        ddr_req_sel16_core        <= 16'd0;
                        ddr_req_word_count_core   <= 3'd1;
                        ddr_req_toggle_core       <= ~ddr_req_toggle_core;
                        train_stdp_state          <= TSK_CLR_A_WAIT;
                    end
                    TSK_CLR_BT_REQ: begin
                        ddr_req_pending_core      <= 1'b1;
                        ddr_req_we_core           <= 1'b1;
                        ddr_req_from_sd_core      <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core <= 1'b1;
                        ddr_req_kind_core <= DDR_REQ_TRAIN;
                        ddr_req_addr_word_core    <= train_stdp_bt_col_base;
                        ddr_req_wdata_core        <= 32'd0;
                        ddr_req_wide_core         <= 1'b0;
                        ddr_req_wdata128_core     <= 128'd0;
                        ddr_req_sel16_core        <= 16'd0;
                        ddr_req_word_count_core   <= 3'd1;
                        ddr_req_toggle_core       <= ~ddr_req_toggle_core;
                        train_stdp_state          <= TSK_CLR_BT_WAIT;
                    end
                    TSK_DONE: begin
                        if (train_stdp_batch_active) begin
                            train_stdp_active       <= 1'b0;
                            train_stdp_state        <= TSK_IDLE;
                            train_stdp_batch_active <= 1'b0;
                            if (train_chunk_active) begin
                                if (train_chunk_samples_left <= 16'd1) begin
                                    // Phase3 continues with a blank-period inference after STDP.
                                    train_chunk_samples_left <= 16'd0;
                                    train_chunk_state        <= TCK_BLANK_INFER_START;
                                end else begin
                                    train_chunk_samples_left <= train_chunk_samples_left - 16'd1;
                                    train_stdp_batch_active  <= 1'b1;
                                    train_stdp_active        <= 1'b1;
                                    train_stdp_row0          <= 7'd0;
                                    train_stdp_row_end       <= N_NEURONS[6:0];
                                    train_stdp_row_idx       <= 7'd0;
                                    train_stdp_col_idx       <= 10'd0;
                                    train_stdp_w_val         <= 32'sd0;
                                    train_stdp_a_val         <= 32'sd0;
                                    train_stdp_bt_val        <= 32'sd0;
                                    train_stdp_w_new         <= 32'sd0;
                                    train_stdp_row_sum_abs   <= 32'd0;
                                    train_stdp_w_row_base    <= TRAIN_BASE_W_Q16_WORDS;
                                    train_stdp_a_row_base    <= TRAIN_BASE_A_Q16_WORDS;
                                    train_stdp_bt_col_base   <= TRAIN_BASE_BT_Q16_WORDS;
                                    train_stdp_state         <= TSK_READ_W_REQ;
                                end
                            end else begin
                                resp_status    <= STATUS_OK;
                                resp_result    <= N_NEURONS;
                                resp_checksum  <= 8'h00;
                                response_ready <= 1'b1;
                            end
                        end else begin
                            train_stdp_active <= 1'b0;
                            train_stdp_state  <= TSK_IDLE;
                            resp_status    <= STATUS_OK;
                            resp_result    <= {25'd0, train_stdp_row_end - train_stdp_row0};
                            resp_checksum  <= 8'h00;
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
                        if (!train_mem_init_done) begin
                            if (!train_mem_init_active) begin
                                train_mem_init_active <= 1'b1;
                                train_mem_init_state  <= TMI_A_WRITE_REQ;
                                train_mem_init_idx    <= '0;
                            end
                        end else if (raw_image_compute_valid && (raw_bytes_per_image == 32'd784) && (raw_image_compute_sum_u8 != 32'd0)) begin
                            infer_active       <= 1'b1;
                            if (train_chunk_retry_continue_infer) begin
                                // mine.py retry keeps neuron/synapse/RNG state and only reruns an inj window.
                                infer_state        <= INFER_CLEAR_SPIKE_COUNT;
                                infer_steps_target <= {16'd0, infer_step_idx} + {16'd0, train_chunk_steps_left};
                                infer_neuron_idx   <= 7'd0;
                                infer_input_idx    <= 10'd0;
                                infer_prep_idx     <= 10'd0;
                                infer_trace_phase <= 2'd0;
                                infer_accum        <= 32'sd0;
                                infer_accum_weight_phase <= 3'd0;
                                infer_apply_idx    <= 7'd0;
                                infer_sum_c_inh    <= 32'd0;
                                infer_total_spikes <= 32'd0;
                                                                        infer_pre_active_count <= 10'd0;
                                raw_image0_rd_addr <= 10'd0;
                                infer_poisson_thresh_rd_addr <= 10'd0;
                                infer_skip_init_clear <= 1'b1;
                                infer_force_no_input  <= 1'b0;
                                infer_trace_wait_last_step <= 1'b0;
                                train_xin_cache_valid  <= 1'b1;
                                train_xexc_cache_valid <= 1'b1;
                            end else begin
                                if (infer_model_state_valid) begin
                                    infer_state        <= INFER_CLEAR_SPIKE_COUNT;
                                end else begin
                                    infer_state        <= INFER_INIT_CLEAR;
                                    infer_model_state_valid <= 1'b1;
                                end
                                infer_steps_target <= {16'd0, infer_step_idx} + {16'd0, train_chunk_steps_left};
                                infer_neuron_idx   <= 7'd0;
                                infer_input_idx    <= 10'd0;
                                infer_prep_idx     <= 10'd0;
                                infer_trace_phase <= 2'd0;
                                infer_accum        <= 32'sd0;
                                infer_accum_weight_phase <= 3'd0;
                                infer_apply_idx    <= 7'd0;
                                infer_sum_c_inh    <= 32'd0;
                                infer_total_spikes <= 32'd0;
                                                                        if (!infer_model_state_valid) begin
                                    infer_rng_state <= 32'h12345678;
                                end
                                infer_pre_active_count <= 10'd0;
                                raw_image0_rd_addr <= 10'd0;
                                infer_skip_init_clear <= 1'b0;
                                infer_force_no_input  <= 1'b0;
                                infer_trace_wait_last_step <= 1'b0;
                                train_xin_cache_valid  <= 1'b1;
                                train_xexc_cache_valid <= 1'b1;
                            end
                            train_chunk_retry_continue_infer <= 1'b0;
                            train_chunk_state <= TCK_INFER_WAIT;
                        end else begin
                            resp_status       <= STATUS_BAD_PACKET;
                            resp_result       <= 32'h36E20001;
                            resp_checksum  <= 8'h00;
                            response_ready    <= 1'b1;
                            train_chunk_active <= 1'b0;
                            train_chunk_mode  <= 2'd0;
                            train_chunk_state <= TCK_IDLE;
                        end
                    end
                    TCK_INFER_WAIT: begin
                        if (!infer_active) begin
                            train_xin_cache_valid  <= 1'b1;
                            train_xexc_cache_valid <= 1'b1;
                            train_chunk_last_infer_spikes <= infer_total_spikes;
                            if (response_ready && (resp_status == STATUS_OK)) begin
                                response_ready <= 1'b0;
                            end
                            train_chunk_state <= TCK_SNAP_COPY_INIT;
                        end
                    end
                    TCK_SNAP_COPY_INIT: begin
                        train_chunk_snap_copy_idx <= 7'd0;
                        infer_spike_rd_addr <= 7'd0;
                        train_chunk_state <= TCK_SNAP_COPY_WAIT;
                    end
                    TCK_SNAP_COPY_WAIT: begin
                        train_chunk_state <= TCK_SNAP_COPY_WRITE;
                    end
                    TCK_SNAP_COPY_WRITE: begin
                        snap_count_we <= 1'b1;
                        snap_count_waddr <= train_chunk_snap_copy_idx;
                        snap_count_wdata <= infer_spike_rd_data;
                        if (train_chunk_snap_copy_idx == (N_NEURONS - 1)) begin
                            train_chunk_state <= TCK_REBASE_GIN_INIT;
                        end else begin
                            train_chunk_snap_copy_idx <= train_chunk_snap_copy_idx + 7'd1;
                            infer_spike_rd_addr <= train_chunk_snap_copy_idx + 7'd1;
                            train_chunk_state <= TCK_SNAP_COPY_WAIT;
                        end
                    end
                    TCK_BLANK_INFER_START: begin
                        if (raw_image_compute_valid && (raw_bytes_per_image == 32'd784) && (raw_image_compute_sum_u8 != 32'd0)) begin
                            infer_active       <= 1'b1;
                            infer_state        <= INFER_GEN_INPUT_SPIKES; // blank: continue existing state, skip clear/threshold prep
                            // Keep infer_step_idx continuous across inj->blank, so blank duration
                            // target must be relative to current step index.
                            infer_steps_target <= {16'd0, infer_step_idx} + TRAIN_MINE_NT_BLANK[31:0];
                            infer_neuron_idx   <= 7'd0;
                            infer_input_idx    <= 10'd0;
                            infer_prep_idx     <= 10'd0;
                            infer_accum        <= 32'sd0;
                            infer_accum_weight_phase <= 3'd0;
                            infer_apply_idx    <= 7'd0;
                            infer_sum_c_inh    <= 32'd0;
                            infer_total_spikes <= 32'd0;
                            infer_pre_active_count <= 10'd0;
                            infer_poisson_thresh_rd_addr <= 10'd0;
                            infer_skip_init_clear <= 1'b1;
                            infer_force_no_input  <= 1'b1;
                            train_chunk_state <= TCK_BLANK_INFER_WAIT;
                        end else begin
                            resp_status       <= STATUS_BAD_PACKET;
                            resp_result       <= 32'h37E30001;
                            resp_checksum  <= 8'h00;
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
                    TCK_REBASE_GIN_INIT: begin
                        train_rebase_neuron_idx <= 7'd0;
                        train_rebase_input_idx <= 10'd0;
                        train_rebase_edge_idx <= '0;
                        train_rebase_edge_end <= '0;
                        train_rebase_accum <= 32'sd0;
                        train_rebase_phase <= 3'd0;
                        train_chunk_state <= TCK_REBASE_GIN_RUN;
                    end
                    TCK_REBASE_GIN_RUN: begin
                        if (train_rebase_phase == 3'd0) begin
                            csr_row_ptr_rd_addr <= train_rebase_neuron_idx[ROW_IDX_W:0];
                            train_rebase_phase <= 3'd1;
                        end else if (train_rebase_phase == 3'd1) begin
                            train_rebase_edge_idx <= csr_row_ptr_rd_data[W_ADDR_W-1:0];
                            csr_row_ptr_rd_addr <= train_rebase_neuron_idx[ROW_IDX_W:0] + {{ROW_IDX_W{1'b0}}, 1'b1};
                            train_rebase_phase <= 3'd2;
                        end else if (train_rebase_phase == 3'd2) begin
                            train_rebase_edge_end <= csr_row_ptr_rd_data[W_ADDR_W-1:0];
                            if (train_rebase_edge_idx >= csr_row_ptr_rd_data[W_ADDR_W-1:0]) begin
                                infer_g_in_state[train_rebase_neuron_idx] <= fxp_mul_s16_16(train_rebase_accum, FXP_SCALE_1000);
                                if (train_rebase_neuron_idx == (N_NEURONS - 1)) begin
                                    train_chunk_state <= TCK_BLANK_INFER_START;
                                end else begin
                                    train_rebase_neuron_idx <= train_rebase_neuron_idx + 7'd1;
                                    train_rebase_accum <= 32'sd0;
                                    train_rebase_phase <= 3'd0;
                                end
                            end else begin
                                csr_col_idx_rd_addr <= train_rebase_edge_idx;
                                train_rebase_phase <= 3'd3;
                            end
                        end else if (train_rebase_phase == 3'd3) begin
                            infer_pre_spike_rd_addr <= {3'd0, csr_col_idx_rd_data};
                            train_rebase_phase <= 3'd4;
                        end else if (train_rebase_phase == 3'd4) begin
                            if (infer_pre_spike_rd_data) begin
                                infer_w_rd_addr <= train_rebase_edge_idx;
                                train_rebase_phase <= 3'd5;
                            end else begin
                                if ((train_rebase_edge_idx + {{(W_ADDR_W-1){1'b0}},1'b1}) >= train_rebase_edge_end) begin
                                    train_rebase_phase <= 3'd2;
                                end else begin
                                    train_rebase_edge_idx <= train_rebase_edge_idx + {{(W_ADDR_W-1){1'b0}},1'b1};
                                    csr_col_idx_rd_addr <= train_rebase_edge_idx + {{(W_ADDR_W-1){1'b0}},1'b1};
                                    train_rebase_phase <= 3'd3;
                                end
                            end
                        end else if (train_rebase_phase == 3'd5) begin
                            train_rebase_phase <= 3'd6;
                        end else begin
                            train_rebase_accum <= train_rebase_accum + $signed({16'd0, infer_w_rd_data_q});
                            if ((train_rebase_edge_idx + {{(W_ADDR_W-1){1'b0}},1'b1}) >= train_rebase_edge_end) begin
                                train_rebase_phase <= 3'd2;
                            end else begin
                                train_rebase_edge_idx <= train_rebase_edge_idx + {{(W_ADDR_W-1){1'b0}},1'b1};
                                csr_col_idx_rd_addr <= train_rebase_edge_idx + {{(W_ADDR_W-1){1'b0}},1'b1};
                                train_rebase_phase <= 3'd3;
                            end
                        end
                    end
                    TCK_DONE: begin
                        train_chunk_active       <= 1'b0;
                        train_chunk_state        <= TCK_IDLE;
                        train_chunk_mode         <= 2'd0;
                        train_chunk_samples_left <= 16'd0;
                        train_chunk_retry_continue_infer <= 1'b0;
                        if (batch_active) begin
                            if (batch_cfg_mode_train) begin
                                train_label_stats_active <= 1'b1;
                                train_label_stats_state <= TLS_ACCUM_READ;
                                train_label_stats_label <= batch_label_rd_data[3:0];
                                train_label_stats_idx <= 10'd0;
                                train_label_stats_base_idx <= {6'd0, batch_label_rd_data[3:0]} * N_NEURONS;
                            end else begin
                                batch_processed_samples <= batch_processed_samples + 32'd1;
                                batch_total_spikes <= batch_total_spikes + train_chunk_last_infer_spikes + train_chunk_last_blank_spikes;
                                if ((batch_processed_samples + 32'd1) >= batch_cfg_num_samples) begin
                                    batch_active <= 1'b0;
                                    batch_done <= 1'b1;
                                    batch_error <= 1'b0;
                                    batch_error_code <= BATCH_ERR_NONE;
                                    batch_phase <= BATCH_PHASE_DONE;
                                    batch_current_sample_idx <= batch_current_sample_idx + 32'd1;
                                end else begin
                                    logic [31:0] next_img_byte_off_tmp;
                                    logic [31:0] next_img_byte_in_sector_tmp;
                                    logic [31:0] next_img_byte_in_sector_norm_tmp;
                                    logic [31:0] next_img_sector_carry_tmp;
                                    logic [31:0] next_img_sector_off_tmp;
                                    logic [31:0] next_img_sectors_needed_tmp;
                                    next_img_byte_off_tmp = batch_img_byte_off + RAW1_BYTES_PER_IMAGE;
                                    next_img_byte_in_sector_tmp = batch_img_byte_in_sector + RAW1_BYTES_PER_IMAGE;
                                    if (next_img_byte_in_sector_tmp >= 32'd1024) begin
                                        next_img_sector_carry_tmp = 32'd2;
                                        next_img_byte_in_sector_norm_tmp = next_img_byte_in_sector_tmp - 32'd1024;
                                    end else if (next_img_byte_in_sector_tmp >= 32'd512) begin
                                        next_img_sector_carry_tmp = 32'd1;
                                        next_img_byte_in_sector_norm_tmp = next_img_byte_in_sector_tmp - 32'd512;
                                    end else begin
                                        next_img_sector_carry_tmp = 32'd0;
                                        next_img_byte_in_sector_norm_tmp = next_img_byte_in_sector_tmp;
                                    end
                                    next_img_sector_off_tmp = batch_img_sector_off + next_img_sector_carry_tmp;
                                    next_img_sectors_needed_tmp = (next_img_byte_in_sector_norm_tmp <= 32'd240) ? 32'd2 : 32'd3;
                                    batch_phase <= BATCH_PHASE_LOADING;
                                    batch_current_sample_idx <= batch_current_sample_idx + 32'd1;
                                    batch_img_byte_off <= next_img_byte_off_tmp;
                                    batch_img_sector_off <= next_img_sector_off_tmp;
                                    batch_img_byte_in_sector <= next_img_byte_in_sector_norm_tmp;
                                    batch_img_sectors_needed <= next_img_sectors_needed_tmp;
                                    batch_label_rd_addr <= batch_current_sample_idx + 32'd1;
                                    sd_copy_active        <= 1'b1;
                                    sd_in_read            <= 1'b0;
                                    sd_copy_lba           <= batch_cfg_start_lba + next_img_sector_off_tmp;
                                    sd_copy_sectors_left  <= next_img_sectors_needed_tmp;
                                    sd_byte_count         <= 9'd0;
                                    sd_pack_idx           <= 2'd0;
                                    sd_pack_word          <= 32'd0;
                                    sd_copy_words_written <= 32'd0;
                                    sd_sector_buf_ready   <= 2'b00;
                                    sd_header_done        <= 1'b0;
                                    sd_file_total_bytes   <= 32'd0;
                                    sd_file_bytes_seen    <= 32'd0;
                                    sd_copy_done_pending  <= 1'b0;
                                    sd_use_sector_limit   <= 1'b1;
                                    sd_copy_raw1_mode     <= 1'b0;
                                    sd_copy_dest_base_word <= IMG_STAGING_BASE_WORD;
                                    raw_image0_valid <= 1'b0;
                                    raw_image0_capture_idx <= 10'd0;
                                    raw_image0_sum_u8 <= 32'd0;
                                end
                            end
                        end else begin
                            // Return inj/blank totals packed as [31:16]=blank, [15:0]=inj (truncated)
                            resp_status    <= STATUS_OK;
                            resp_result    <= {train_chunk_last_blank_spikes[15:0], train_chunk_last_infer_spikes[15:0]};
                            resp_checksum  <= 8'h00;
                            response_ready <= 1'b1;
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
                            train_xin_wr_en   <= 1'b1;
                            train_xin_wr_addr <= train_gen_idx[9:0];
                            train_xin_wr_data <= train_gen_curr_word;
                        end else if (train_gen_cache_mode == 2'd2) begin
                            train_xexc_wr_en   <= 1'b1;
                            train_xexc_wr_addr <= train_gen_idx[6:0];
                            train_xexc_wr_data <= train_gen_curr_word;
                        end
                        ddr_req_pending_core      <= 1'b1;
                        ddr_req_we_core           <= 1'b1;
                        ddr_req_from_sd_core      <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core <= 1'b1;
                        ddr_req_kind_core <= DDR_REQ_TRAIN;
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
                            resp_checksum  <= 8'h00;
                            response_ready <= 1'b1;
                        end
                    end
                    default: begin end
                endcase
            end

            if (TRAIN_ENABLE && train_mem_init_active && !ddr_req_pending_core && !response_ready &&
                !sd_ddr_flush_active && !imgload_word_valid &&
                !train_trace_active && !train_stdp_active && !train_gen_active) begin
                case (train_mem_init_state)
                    TMI_W_READ_REQ: begin
                        train_mem_init_idx <= '0;
                        train_mem_init_state <= TMI_A_WRITE_REQ;
                    end
                    TMI_W_READ_WAIT: begin
                        train_mem_init_idx <= '0;
                        train_mem_init_state <= TMI_A_WRITE_REQ;
                    end
                    TMI_A_WRITE_REQ: begin
                        ddr_req_pending_core      <= 1'b1;
                        ddr_req_we_core           <= 1'b1;
                        ddr_req_from_sd_core      <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core <= 1'b1;
                        ddr_req_kind_core <= DDR_REQ_TRAIN;
                        ddr_req_addr_word_core    <= TRAIN_BASE_A_Q16_WORDS + {15'd0, train_mem_init_idx};
                        ddr_req_wdata_core        <= 32'd0;
                        ddr_req_wide_core         <= 1'b0;
                        ddr_req_wdata128_core     <= 128'd0;
                        ddr_req_sel16_core        <= 16'd0;
                        ddr_req_word_count_core   <= 3'd1;
                        ddr_req_toggle_core       <= ~ddr_req_toggle_core;
                        train_mem_init_state      <= TMI_A_WRITE_WAIT;
                    end
                    TMI_BT_WRITE_REQ: begin
                        ddr_req_pending_core      <= 1'b1;
                        ddr_req_we_core           <= 1'b1;
                        ddr_req_from_sd_core      <= 1'b0;
                        ddr_req_from_imgload_core <= 1'b0;
                        ddr_req_from_train_core <= 1'b1;
                        ddr_req_kind_core <= DDR_REQ_TRAIN;
                        ddr_req_addr_word_core    <= TRAIN_BASE_BT_Q16_WORDS + {15'd0, train_mem_init_idx};
                        ddr_req_wdata_core        <= 32'd0;
                        ddr_req_wide_core         <= 1'b0;
                        ddr_req_wdata128_core     <= 128'd0;
                        ddr_req_sel16_core        <= 16'd0;
                        ddr_req_word_count_core   <= 3'd1;
                        ddr_req_toggle_core       <= ~ddr_req_toggle_core;
                        train_mem_init_state      <= TMI_BT_WRITE_WAIT;
                    end
                    TMI_DONE: begin
                        train_mem_init_active <= 1'b0;
                        train_mem_init_done   <= 1'b1;
                        train_mem_init_state  <= TMI_IDLE;
                    end
                    default: begin end
                endcase
            end

            if (TRAIN_ENABLE && train_label_stats_active && !response_ready && !sd_ddr_flush_active && !imgload_word_valid &&
                !ddr_req_pending_core && !train_trace_active && !train_stdp_active && !train_gen_active && !train_chunk_active && !infer_active) begin
                case (train_label_stats_state)
                    TLS_RESET_SUM: begin
                        train_label_sum_wr_en   <= 1'b1;
                        train_label_sum_wr_addr <= train_label_stats_idx;
                        train_label_sum_wr_data <= 32'd0;
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
                        snap_count_raddr <= train_label_stats_idx[6:0];
                        train_label_stats_state <= TLS_ACCUM_WAIT;
                    end
                    TLS_ACCUM_WAIT: begin
                        train_label_stats_state <= TLS_ACCUM_SAMPLE;
                    end
                    TLS_ACCUM_SAMPLE: begin
                        train_label_stats_spike_q <= snap_count_rdata;
                        train_label_stats_state <= TLS_ACCUM_WRITE;
                    end
                    TLS_ACCUM_WRITE: begin
                        train_label_sum_wr_en   <= 1'b1;
                        train_label_sum_wr_addr <= train_label_stats_base_idx + train_label_stats_idx;
                        train_label_sum_wr_data <= train_label_sum_rd_data + {16'd0, train_label_stats_spike_q};
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
                        if (batch_active && batch_cfg_mode_train) begin
                            batch_processed_samples <= batch_processed_samples + 32'd1;
                            batch_total_spikes <= batch_total_spikes + train_chunk_last_infer_spikes + train_chunk_last_blank_spikes;
                            if ((batch_processed_samples + 32'd1) >= batch_cfg_num_samples) begin
                                batch_active <= 1'b0;
                                batch_done <= 1'b1;
                                batch_error <= 1'b0;
                                batch_error_code <= BATCH_ERR_NONE;
                                batch_phase <= BATCH_PHASE_DONE;
                                batch_current_sample_idx <= batch_current_sample_idx + 32'd1;
                            end else begin
                                batch_current_sample_idx <= batch_current_sample_idx + 32'd1;
                                batch_label_rd_addr <= batch_current_sample_idx + 32'd1;
                                if (batch_use_cached_images) begin
                                    batch_phase <= BATCH_PHASE_LOADING;
                                    batch_cache_img_byte_off <= batch_cache_img_byte_off + RAW1_BYTES_PER_IMAGE;
                                    raw_image0_valid <= 1'b0;
                                    raw_image0_capture_idx <= 10'd0;
                                    raw_image0_sum_u8 <= 32'd0;
                                end else begin
                                    logic [31:0] next_img_byte_off_tmp;
                                    logic [31:0] next_img_byte_in_sector_tmp;
                                    logic [31:0] next_img_byte_in_sector_norm_tmp;
                                    logic [31:0] next_img_sector_carry_tmp;
                                    logic [31:0] next_img_sector_off_tmp;
                                    logic [31:0] next_img_sectors_needed_tmp;
                                    next_img_byte_off_tmp = batch_img_byte_off + RAW1_BYTES_PER_IMAGE;
                                    next_img_byte_in_sector_tmp = batch_img_byte_in_sector + RAW1_BYTES_PER_IMAGE;
                                    if (next_img_byte_in_sector_tmp >= 32'd1024) begin
                                        next_img_sector_carry_tmp = 32'd2;
                                        next_img_byte_in_sector_norm_tmp = next_img_byte_in_sector_tmp - 32'd1024;
                                    end else if (next_img_byte_in_sector_tmp >= 32'd512) begin
                                        next_img_sector_carry_tmp = 32'd1;
                                        next_img_byte_in_sector_norm_tmp = next_img_byte_in_sector_tmp - 32'd512;
                                    end else begin
                                        next_img_sector_carry_tmp = 32'd0;
                                        next_img_byte_in_sector_norm_tmp = next_img_byte_in_sector_tmp;
                                    end
                                    next_img_sector_off_tmp = batch_img_sector_off + next_img_sector_carry_tmp;
                                    next_img_sectors_needed_tmp = (next_img_byte_in_sector_norm_tmp <= 32'd240) ? 32'd2 : 32'd3;
                                    batch_phase <= BATCH_PHASE_LOADING;
                                    batch_img_byte_off <= next_img_byte_off_tmp;
                                    batch_img_sector_off <= next_img_sector_off_tmp;
                                    batch_img_byte_in_sector <= next_img_byte_in_sector_norm_tmp;
                                    batch_img_sectors_needed <= next_img_sectors_needed_tmp;
                                    sd_copy_active        <= 1'b1;
                                    sd_in_read            <= 1'b0;
                                    sd_copy_lba           <= batch_cfg_start_lba + next_img_sector_off_tmp;
                                    sd_copy_sectors_left  <= next_img_sectors_needed_tmp;
                                    sd_byte_count         <= 9'd0;
                                    sd_pack_idx           <= 2'd0;
                                    sd_pack_word          <= 32'd0;
                                    sd_copy_words_written <= 32'd0;
                                    sd_sector_buf_ready   <= 2'b00;
                                    sd_header_done        <= 1'b0;
                                    sd_file_total_bytes   <= 32'd0;
                                    sd_file_bytes_seen    <= 32'd0;
                                    sd_copy_done_pending  <= 1'b0;
                                    sd_use_sector_limit   <= 1'b1;
                                    sd_copy_raw1_mode     <= 1'b0;
                                    sd_copy_dest_base_word <= IMG_STAGING_BASE_WORD;
                                    raw_image0_valid <= 1'b0;
                                    raw_image0_capture_idx <= 10'd0;
                                    raw_image0_sum_u8 <= 32'd0;
                                end
                            end
                        end else begin
                            resp_status    <= STATUS_OK;
                            resp_result    <= {28'd0, train_label_stats_label};
                            resp_checksum  <= 8'h00;
                            response_ready <= 1'b1;
                        end
                    end
                    default: begin
                        train_label_stats_active <= 1'b0;
                        train_label_stats_state <= TLS_IDLE;
                        resp_status    <= STATUS_BAD_PACKET;
                        resp_result    <= 32'sd0;
                        resp_checksum  <= 8'h00;
                        response_ready <= 1'b1;
                    end
                endcase
            end

            // During infer/imgload, drop incoming request bytes to avoid protocol desync
            // while long-running pipelines are active.
            if (rx_dv && !response_ready && !memrd_pending &&
                !sd_copy_active && !infer_active && !imgload_active && !imgload_start_pending) begin
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
                            resp_checksum  <= 8'h00;
                            response_ready  <= 1'b1;
                        end else if (
                            ((req_opcode == OP_SD_SECTORS_TO_DDR) || (req_opcode == OP_LOAD_IMAGE_FROM_DDR) ||
                             (req_opcode == OP_RUN_SAMPLE_INFER) || (req_opcode == OP_READ_SPIKE_COUNT) ||
                             (req_opcode == OP_TRAIN_QUERY_CAPS) ||
                             (req_opcode == OP_TRAIN_RUN_SAMPLE_PHASE4) ||
                             (req_opcode == OP_TRAIN_LABEL_STATS_RESET) ||
                             (req_opcode == OP_TRAIN_LABEL_STATS_ACCUM) ||
                             (req_opcode == OP_READ_TRAIN_LABEL_STAT_SUM) ||
                             (req_opcode == OP_READ_TRAIN_LABEL_STAT_COUNT) ||
                             (req_opcode == OP_BATCH_CONFIG0) ||
                             (req_opcode == OP_BATCH_CONFIG1) ||
                             (req_opcode == OP_BATCH_START) ||
                             (req_opcode == OP_BATCH_STATUS) ||
                             (req_opcode == OP_BATCH_READ_SUMMARY) ||
                             (req_opcode == OP_BATCH_CONFIG2) ||
                             (req_opcode == OP_BATCH_LABEL_WRITE) ||
                             (req_opcode == OP_BATCH_ASSIGN_WRITE))
                            && (rx_byte != 8'd2)
                        ) begin
                            rx_state        <= RX_WAIT_SYNC;
                            resp_status     <= STATUS_BAD_PACKET;
                            resp_result     <= 32'sd0;
                            resp_checksum  <= 8'h00;
                            response_ready  <= 1'b1;
                        end else if (
                            (req_opcode == OP_TRAIN_RUN_SAMPLE_PHASE3)
                            && (rx_byte != 8'd1)
                        ) begin
                            rx_state        <= RX_WAIT_SYNC;
                            resp_status     <= STATUS_BAD_PACKET;
                            resp_result     <= 32'sd0;
                            resp_checksum  <= 8'h00;
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
                            resp_checksum  <= 8'h00;
                            response_ready <= 1'b1;
                        end else if (train_busy_uart_blocked) begin
                            resp_status    <= STATUS_BAD_PACKET;
                            resp_result    <= 32'h31000000;
                            resp_checksum  <= 8'h00;
                            response_ready <= 1'b1;
                        end else if (!TRAIN_ENABLE &&
                                     (req_opcode == OP_TRAIN_RUN_SAMPLE_PHASE3)) begin
                            resp_status    <= STATUS_UNSUPPORTED_OP;
                            // Debug payload: [31:24]=0xF0, [23:16]=req_opcode, [15:8]=req_nargs, [7:0]=rx_state
                            resp_result    <= {8'hF0, req_opcode, req_nargs, {5'd0, rx_state}};
                            resp_checksum  <= 8'h00;
                            response_ready <= 1'b1;
                        end else begin
                            case (req_opcode)
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
                                        sd_copy_dest_base_word <= IMG_STAGING_BASE_WORD;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= {BADDBG_SD_REQ_ARG, req_opcode, arg0[15:0]};
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_LOAD_IMAGE_FROM_DDR: begin
                                    // Semantics:
                                    //   arg0 = DDR source byte offset (base_byte)
                                    //   arg1 = number of bytes to copy
                                    // Destination is always raw_image0[0..arg1-1] (dest base fixed to 0).
                                    if ((req_nargs == 8'd2) &&
                                        (arg0 >= 0) &&
                                        (arg1 > 0) && (arg1 <= N_IN) &&
                                        ((IMG_STAGING_BASE_WORD + {2'b00, arg0[31:2]}) < DDR_ADDR_WORD_LIMIT) &&
                                        ddr_calib_complete_core && !ddr_req_pending_core &&
                                        !imgload_active && !imgload_start_pending) begin
                                        // Drain residual DDR responses before starting imgload requests.
                                        sd_ddr_flush_active <= 1'b0;
                                        sd_sector_buf_ready <= 2'b00;
                                        ddr_rsp_toggle_core_seen <= ddr_rsp_toggle_core_sync2;
                                        ddr_rsp_capture_pending_core <= 1'b0;
                                        ddr_rsp_payload_ready_core <= 1'b0;
                                        ddr_rsp_drain_active_core <= 1'b1;
                                        ddr_rsp_drain_quiet_core <= 2'd0;
                                        imgload_target_buf_sel <= 1'b0;
                                        imgload_start_pending <= 1'b1;
                                        imgload_start_addr_word <= IMG_STAGING_BASE_WORD + {2'b00, arg0[31:2]};
                                        imgload_start_lane <= arg0[1:0];
                                        imgload_start_total_bytes <= arg1[9:0];
                                        imgload_word_valid <= 1'b0;
                                        raw_image0_valid <= 1'b0;
                                        raw_image0_capture_idx <= 10'd0;
                                        raw_image0_sum_u8 <= 32'd0;
                                        raw_bytes_per_image <= arg1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_RUN_SAMPLE_INFER: begin
                                    // arg0 = seed, arg1 = steps (>0); no STDP, no label accumulation.
                                    if ((req_nargs == 8'd2) &&
                                        (arg1 > 0) && (arg1 <= 32'sd65535) &&
                                        raw_image_compute_valid && (raw_bytes_per_image == 32'd784) && (raw_image_compute_sum_u8 != 32'd0) &&
                                        !ddr_req_pending_core &&
                                        !train_trace_active && !train_stdp_active && !train_gen_active &&
                                        !train_label_stats_active && !train_chunk_active && !infer_active) begin
                                        infer_active       <= 1'b1;
                                        infer_state        <= INFER_INIT_CLEAR;
                                        infer_steps_target <= {16'd0, arg1[15:0]};
                                        infer_step_idx     <= 16'd0;
                                        infer_neuron_idx   <= 7'd0;
                                        infer_input_idx    <= 10'd0;
                                        infer_prep_idx     <= 10'd0;
                                        infer_accum        <= 32'sd0;
                                        infer_accum_weight_phase <= 3'd0;
                                        infer_apply_idx    <= 7'd0;
                                        infer_trace_phase  <= 2'd0;
                                        infer_total_spikes <= 32'd0;
                                                                                                infer_rng_state    <= arg0[31:0];
                                        infer_skip_init_clear <= 1'b0;
                                        infer_force_no_input  <= 1'b0;
                                        infer_poisson_num_const_cfg <= POISSON_NUM_CONST;
                                        infer_pre_active_count <= 10'd0;
                                        raw_image0_rd_addr <= 10'd0;
                                        infer_poisson_thresh_rd_addr <= 10'd0;
                                        infer_model_state_valid <= 1'b0;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_READ_SPIKE_COUNT: begin
                                    if ((req_nargs == 8'd2) &&
                                        (arg0 >= 0) && (arg0 < N_NEURONS) &&
                                        !infer_active) begin
                                        infer_spike_rd_addr <= arg0[6:0];
                                        memrd_kind    <= MEMRD_SPIKE_COUNT;
                                        memrd_wait    <= 1'b1;
                                        memrd_pending <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRAIN_QUERY_CAPS: begin
                                    if (req_nargs == 8'd2) begin
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= TRAIN_ENABLE ? TRAIN_CAPS_VALUE : 32'd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_BATCH_CONFIG0: begin
                                    if ((req_nargs == 8'd2) && !batch_active) begin
                                        logic [31:0] cfg_start_sample_idx_tmp;
                                        logic [31:0] cfg_start_byte_off_tmp;
                                        logic [31:0] cfg_start_byte_in_sector_tmp;
                                        logic [31:0] cfg_num_samples_bytes_tmp;
                                        logic [31:0] cfg_cache_total_bytes_tmp;
                                        logic [31:0] cfg_cache_total_sectors_tmp;
                                        logic [31:0] cfg_cache_total_words_tmp;
                                        cfg_start_sample_idx_tmp = arg1;
                                        cfg_start_byte_off_tmp = RAW1_HEADER_BYTES + RAW1_NUM_IMAGES +
                                                                 (cfg_start_sample_idx_tmp * RAW1_BYTES_PER_IMAGE);
                                        cfg_start_byte_in_sector_tmp = cfg_start_byte_off_tmp % 32'd512;
                                        cfg_num_samples_bytes_tmp = batch_cfg_num_samples * RAW1_BYTES_PER_IMAGE;
                                        cfg_cache_total_bytes_tmp = cfg_start_byte_in_sector_tmp + cfg_num_samples_bytes_tmp;
                                        cfg_cache_total_sectors_tmp = (cfg_cache_total_bytes_tmp + 32'd511) >> 9;
                                        cfg_cache_total_words_tmp = cfg_cache_total_sectors_tmp << 7;
                                        batch_cfg_mode_train <= arg0[0];
                                        batch_cfg_start_sample_idx <= arg1;
                                        batch_cfg_start_byte_off <= cfg_start_byte_off_tmp;
                                        batch_cfg_start_sector_off <= cfg_start_byte_off_tmp >> 9;
                                        batch_cfg_start_byte_in_sector <= cfg_start_byte_in_sector_tmp;
                                        batch_cfg_start_sectors_needed <= (cfg_start_byte_in_sector_tmp + RAW1_BYTES_PER_IMAGE + 32'd511) >> 9;
                                        batch_cfg_cache_total_sectors <= cfg_cache_total_sectors_tmp;
                                        batch_cfg_cache_total_words <= cfg_cache_total_words_tmp;
                                        batch_cfg_cache_fits <= ((IMG_CACHE_BASE_WORD + cfg_cache_total_words_tmp) < DDR_ADDR_WORD_LIMIT);
                                        batch_cfg0_valid <= 1'b1;
                                        batch_done <= 1'b0;
                                        batch_error <= 1'b0;
                                        batch_error_code <= BATCH_ERR_NONE;
                                        batch_phase <= BATCH_PHASE_CONFIGURED;
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= 32'd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_BATCH_CONFIG1: begin
                                    if ((req_nargs == 8'd2) && !batch_active && (arg0 > 0)) begin
                                        logic [31:0] cfg_num_samples_tmp;
                                        logic [31:0] cfg_num_samples_bytes_tmp;
                                        logic [31:0] cfg_cache_total_bytes_tmp;
                                        logic [31:0] cfg_cache_total_sectors_tmp;
                                        logic [31:0] cfg_cache_total_words_tmp;
                                        cfg_num_samples_tmp = arg0;
                                        cfg_num_samples_bytes_tmp = cfg_num_samples_tmp * RAW1_BYTES_PER_IMAGE;
                                        cfg_cache_total_bytes_tmp = batch_cfg_start_byte_in_sector + cfg_num_samples_bytes_tmp;
                                        cfg_cache_total_sectors_tmp = (cfg_cache_total_bytes_tmp + 32'd511) >> 9;
                                        cfg_cache_total_words_tmp = cfg_cache_total_sectors_tmp << 7;
                                        batch_cfg_num_samples <= arg0;
                                        batch_cfg_seed <= arg1;
                                        batch_cfg_num_samples_bytes <= cfg_num_samples_bytes_tmp;
                                        batch_cfg_cache_total_sectors <= cfg_cache_total_sectors_tmp;
                                        batch_cfg_cache_total_words <= cfg_cache_total_words_tmp;
                                        batch_cfg_cache_fits <= ((IMG_CACHE_BASE_WORD + cfg_cache_total_words_tmp) < DDR_ADDR_WORD_LIMIT);
                                        batch_cfg1_valid <= 1'b1;
                                        batch_done <= 1'b0;
                                        batch_error <= 1'b0;
                                        batch_error_code <= BATCH_ERR_NONE;
                                        batch_phase <= BATCH_PHASE_CONFIGURED;
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= 32'd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_BATCH_CONFIG2: begin
                                    if ((req_nargs == 8'd2) && !batch_active) begin
                                        batch_cfg_start_lba <= arg0;
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= arg1;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_BATCH_LABEL_WRITE: begin
                                    if ((req_nargs == 8'd2) &&
                                        !batch_active &&
                                        (arg0 >= 0) && (arg0 < RAW1_NUM_IMAGES) &&
                                        (arg1 >= 0) && (arg1 < 10)) begin
                                        batch_label_wr_en <= 1'b1;
                                        batch_label_wr_addr <= arg0[13:0];
                                        batch_label_wr_data <= arg1[7:0];
                                        batch_label_rd_addr <= arg0[13:0];
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= arg0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_BATCH_ASSIGN_WRITE: begin
                                    if ((req_nargs == 8'd2) &&
                                        !batch_active &&
                                        (arg0 >= 0) && (arg0 < N_NEURONS) &&
                                        (arg1 >= 0) && (arg1 < 10)) begin
                                        batch_assign_wr_en <= 1'b1;
                                        batch_assign_wr_addr <= arg0[6:0];
                                        batch_assign_wr_data <= arg1[3:0];
                                        batch_assign_rd_addr <= arg0[6:0];
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= arg0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_BATCH_START: begin
                                    if ((req_nargs == 8'd2) && !batch_active) begin
                                        batch_processed_samples <= 32'd0;
                                        batch_current_sample_idx <= batch_cfg_start_sample_idx;
                                        batch_total_spikes <= 32'd0;
                                        batch_correct_count <= 32'd0;
                                        batch_elapsed_cycles <= 64'd0;
                                        batch_load_cycles <= 64'd0;
                                        batch_train_core_cycles <= 64'd0;
                                        batch_infer_core_cycles <= 64'd0;
                                        batch_label_stats_cycles <= 64'd0;
                                        batch_infer_eval_cycles <= 64'd0;
                                        batch_other_cycles <= 64'd0;
                                        batch_train_inject_infer_cycles <= 64'd0;
                                        batch_train_blank_infer_cycles <= 64'd0;
                                        batch_train_snap_cycles <= 64'd0;
                                        batch_train_rebase_cycles <= 64'd0;
                                        batch_train_evt_pre_cycles <= 64'd0;
                                        batch_train_evt_post_cycles <= 64'd0;
                                        batch_train_accum_cycles <= 64'd0;
                                        batch_img_byte_off <= batch_cfg_start_byte_off;
                                        batch_img_sector_off <= batch_cfg_start_sector_off;
                                        batch_img_byte_in_sector <= batch_cfg_start_byte_in_sector;
                                        batch_img_sectors_needed <= batch_cfg_start_sectors_needed;
                                        batch_use_cached_images <= 1'b0;
                                        batch_cache_img_byte_off <= batch_cfg_start_byte_in_sector;
                                        batch_label_rd_addr <= batch_cfg_start_sample_idx[13:0];
                                        batch_compute_buf_sel <= 1'b0;
                                        batch_fill_buf_sel <= 1'b1;
                                        imgload_target_buf_sel <= 1'b0;
                                        batch_prefetch_active <= 1'b0;
                                        batch_prefetch_issue_pending <= 1'b0;
                                        batch_prefetch_ready <= 1'b0;
                                        batch_prefetch_sample_idx <= 32'd0;
                                        if (!(batch_cfg0_valid && batch_cfg1_valid)) begin
                                            batch_done <= 1'b1;
                                            batch_error <= 1'b1;
                                            batch_phase <= BATCH_PHASE_DONE;
                                            batch_error_code <= BATCH_ERR_NOT_READY;
                                            resp_result <= 32'h42000001;
                                            resp_status <= STATUS_BAD_PACKET;
                                        end else if (ddr_calib_complete_core &&
                                                     !sd_copy_active && !imgload_active && !imgload_start_pending &&
                                                     !ddr_req_pending_core &&
                                                     !train_trace_active && !train_stdp_active && !train_stdp_batch_active &&
                                                     !train_chunk_active && !train_label_stats_active && !infer_active) begin
                                                batch_active <= 1'b1;
                                                batch_done <= 1'b0;
                                                batch_error <= 1'b0;
                                                batch_error_code <= BATCH_ERR_NONE;
                                                batch_phase <= BATCH_PHASE_LOADING;
                                                sd_copy_active        <= 1'b1;
                                                sd_in_read            <= 1'b0;
                                                if (batch_cfg_cache_fits) begin
                                                    batch_use_cached_images <= 1'b1;
                                                    batch_cache_img_byte_off <= batch_cfg_start_byte_in_sector;
                                                    sd_copy_lba           <= batch_cfg_start_lba + batch_cfg_start_sector_off;
                                                    sd_copy_sectors_left  <= batch_cfg_cache_total_sectors;
                                                    sd_copy_dest_base_word <= IMG_CACHE_BASE_WORD;
                                                end else begin
                                                    batch_use_cached_images <= 1'b0;
                                                    batch_cache_img_byte_off <= batch_cfg_start_byte_in_sector;
                                                    sd_copy_lba           <= batch_cfg_start_lba + batch_cfg_start_sector_off;
                                                    sd_copy_sectors_left  <= batch_cfg_start_sectors_needed;
                                                    sd_copy_dest_base_word <= IMG_STAGING_BASE_WORD;
                                                end
                                                sd_byte_count         <= 9'd0;
                                                sd_pack_idx           <= 2'd0;
                                                sd_pack_word          <= 32'd0;
                                                sd_copy_words_written <= 32'd0;
                                                sd_sector_buf_ready   <= 2'b00;
                                                sd_header_done        <= 1'b0;
                                                sd_file_total_bytes   <= 32'd0;
                                                sd_file_bytes_seen    <= 32'd0;
                                                sd_copy_done_pending  <= 1'b0;
                                                sd_use_sector_limit   <= 1'b1;
                                                sd_copy_raw1_mode     <= 1'b0;
                                                raw_image0_valid <= 1'b0;
                                                raw_image0_capture_idx <= 10'd0;
                                                raw_image0_sum_u8 <= 32'd0;
                                                resp_result <= 32'd0;
                                                resp_status <= STATUS_OK;
                                        end else begin
                                            batch_done <= 1'b1;
                                            batch_error <= 1'b1;
                                            batch_phase <= BATCH_PHASE_DONE;
                                            batch_error_code <= BATCH_ERR_NOT_READY;
                                            resp_result <= 32'h42000001;
                                            resp_status <= STATUS_BAD_PACKET;
                                        end
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_BATCH_STATUS: begin
                                    if (req_nargs == 8'd2) begin
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= batch_status_word;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_BATCH_READ_SUMMARY: begin
                                    if (req_nargs == 8'd2) begin
                                        resp_status <= STATUS_OK;
                                        resp_result <= batch_summary_word;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRAIN_RUN_SAMPLE_PHASE3: begin
                                    // arg0 = inj steps (>0); blank steps fixed to TRAIN_MINE_NT_BLANK
                                    if ((req_nargs == 8'd1) &&
                                        (arg0 > 0) && (arg0 <= 32'sd65535) &&
                                        ddr_calib_complete_core &&
                                        !ddr_req_pending_core &&
                                        !train_trace_active &&
                                        !train_stdp_active &&
                                        !train_stdp_batch_active &&
                                        !train_chunk_active) begin
                                        train_chunk_active       <= 1'b1;
                                        train_chunk_mode         <= 3'd3;
                                        train_chunk_state        <= TCK_INFER_START;
                                        train_chunk_samples_left <= 16'd1;
                                        train_chunk_steps_left   <= arg0[15:0];
                                        train_chunk_seed_xin     <= 32'h13579BDF;
                                        train_chunk_seed_xexc    <= 32'h2468ACE1;
                                        train_chunk_winner       <= 7'd0;
                                        train_chunk_pre_idx      <= 10'd0;
                                        train_chunk_last_infer_spikes <= 32'd0;
                                        train_chunk_last_blank_spikes <= 32'd0;
                                        train_chunk_retry_curr_max_fr <= TRAIN_RETRY_MAX_FR_START;
                                        train_chunk_retry_accepted_max_fr <= TRAIN_RETRY_MAX_FR_START;
                                        train_chunk_retry_continue_infer <= 1'b0;
                                        train_stdp_update_nt <= {16'd0, arg0[15:0]};
                                        infer_poisson_num_const_cfg <= POISSON_NUM_CONST;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= TRAIN_CAPS_VALUE;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRAIN_RUN_SAMPLE_PHASE4: begin
                                    if ((req_nargs == 8'd2) &&
                                        (arg0 > 0) && (arg0 <= 32'sd65535) &&
                                        ddr_calib_complete_core &&
                                        !ddr_req_pending_core &&
                                        !train_trace_active &&
                                        !train_stdp_active &&
                                        !train_stdp_batch_active &&
                                        !train_chunk_active &&
                                        !train_label_stats_active) begin
                                        train_chunk_active       <= 1'b1;
                                        train_chunk_mode         <= 3'd3;
                                        train_chunk_state        <= TCK_INFER_START;
                                        train_chunk_samples_left <= 16'd1;
                                        train_chunk_steps_left   <= arg0[15:0];
                                        train_chunk_seed_xin     <= 32'h13579BDF;
                                        train_chunk_seed_xexc    <= 32'h2468ACE1;
                                        train_chunk_winner       <= 7'd0;
                                        train_chunk_pre_idx      <= 10'd0;
                                        train_chunk_last_infer_spikes <= 32'd0;
                                        train_chunk_last_blank_spikes <= 32'd0;
                                        train_chunk_retry_curr_max_fr <= TRAIN_RETRY_MAX_FR_START;
                                        train_chunk_retry_accepted_max_fr <= TRAIN_RETRY_MAX_FR_START;
                                        train_chunk_retry_continue_infer <= 1'b0;
                                        train_stdp_update_nt <= {16'd0, arg0[15:0]};
                                        infer_poisson_num_const_cfg <= POISSON_NUM_CONST;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= TRAIN_CAPS_VALUE;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRAIN_LABEL_STATS_RESET: begin
                                    if ((req_nargs == 8'd2) &&
                                        !train_label_stats_active &&
                                        !train_chunk_active &&
                                        !train_trace_active &&
                                        !train_stdp_active &&
                                        !train_stdp_batch_active &&
                                        !train_gen_active &&
                                        !infer_active) begin
                                        train_label_stats_active <= 1'b1;
                                        train_label_stats_state <= TLS_RESET_SUM;
                                        train_label_stats_idx <= 10'd0;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_TRAIN_LABEL_STATS_ACCUM: begin
                                    if ((req_nargs == 8'd2) &&
                                        (arg0 >= 0) && (arg0 < 10) &&
                                        !train_label_stats_active &&
                                        !train_chunk_active &&
                                        !train_trace_active &&
                                        !train_stdp_active &&
                                        !train_stdp_batch_active &&
                                        !train_gen_active &&
                                        !infer_active) begin
                                        train_label_stats_active <= 1'b1;
                                        train_label_stats_state <= TLS_ACCUM_READ;
                                        train_label_stats_label <= arg0[3:0];
                                        train_label_stats_idx <= 10'd0;
                                        train_label_stats_base_idx <= {6'd0, arg0[3:0]} * N_NEURONS;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_READ_TRAIN_LABEL_STAT_SUM: begin
                                    if ((req_nargs == 8'd2) &&
                                        (arg0 >= 0) && (arg0 < 10) &&
                                        (arg1 >= 0) && (arg1 < N_NEURONS) &&
                                        !train_label_stats_active) begin
                                        train_label_sum_rd_addr <= ({6'd0, arg0[3:0]} * N_NEURONS) + arg1[9:0];
                                        memrd_kind    <= MEMRD_TRAIN_LABEL_STAT_SUM;
                                        memrd_wait    <= 1'b1;
                                        memrd_pending <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_READ_TRAIN_LABEL_STAT_COUNT: begin
                                    if ((req_nargs == 8'd2) &&
                                        (arg0 >= 0) && (arg0 < 10) &&
                                        !train_label_stats_active) begin
                                        train_label_count_rd_addr <= arg0[3:0];
                                        memrd_kind    <= MEMRD_TRAIN_LABEL_STAT_COUNT;
                                        memrd_wait    <= 1'b1;
                                        memrd_pending <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= 8'h00;
                                        response_ready <= 1'b1;
                                    end
                                end
                                default: begin
                                    resp_status    <= STATUS_UNSUPPORTED_OP;
                                    // Debug payload: [31:24]=0xF1, [23:16]=req_opcode, [15:8]=req_nargs, [7:0]=rx_state
                                    resp_result    <= {8'hF1, req_opcode, req_nargs, {5'd0, rx_state}};
                                    resp_checksum  <= 8'h00;
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
                        resp_result    <= {BADDBG_SD_CD_N, OP_SD_SECTORS_TO_DDR, SD_CD_N, 10'd0, sd_status};
                        resp_checksum  <= 8'h00;
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
                            resp_result    <= {BADDBG_SD_WAIT_TO, OP_SD_SECTORS_TO_DDR, sd_wait_counter[15:0]};
                            resp_checksum  <= 8'h00;
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
                            ddr_rsp_drain_active_core <= 1'b1;
                            ddr_rsp_drain_quiet_core <= 2'd0;
                            ddr_rsp_toggle_core_seen <= ddr_rsp_toggle_core_sync2;
                            if (!batch_active) begin
                                sd_copy_resp_pending <= 1'b1;
                                sd_copy_resp_status <= STATUS_OK;
                                sd_copy_resp_result <= sd_copy_words_written;
                            end
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
                        infer_c_exc_state[infer_apply_idx] <= 32'sd0;
                        infer_c_inh_state[infer_apply_idx] <= 32'sd0;
                        infer_g_inh_state[infer_apply_idx] <= 32'sd0;
                        infer_g_exc_delay0[infer_apply_idx] <= 32'sd0;
                        infer_g_exc_delay1[infer_apply_idx] <= 32'sd0;
                        infer_s_exc[infer_apply_idx] <= 1'b0;
                        infer_exc_last_spike_step[infer_apply_idx] <= 16'd0;
                        infer_inh_last_spike_step[infer_apply_idx] <= 16'd0;
                        if (infer_apply_idx == (N_NEURONS - 1)) begin
                            infer_apply_idx <= 7'd0;
                            infer_state <= INFER_CLEAR_SPIKE_COUNT;
                        end else begin
                            infer_apply_idx <= infer_apply_idx + 7'd1;
                        end
                    end

                    INFER_CLEAR_SPIKE_COUNT: begin
                        spike_count_we <= 1'b1;
                        spike_count_waddr <= infer_apply_idx;
                        spike_count_wdata <= 16'd0;
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
                                // Brian2 minimal: p_i = clip((raw_u8 / 8) * 2.0 * 1e-3, 0, 1).
                                // Convert probability to a threshold against rand11 in [0, 2047]:
                                // threshold_i = floor(raw_u8 * 2048 / 4000).
                                infer_prep_div_prod_q32 <= $unsigned(infer_poisson_num_const_cfg) * $unsigned({24'd0, raw_image_compute_rd_data});
                                infer_divisor  <= POISSON_DEN_CONST;
                                infer_state <= INFER_PREP_DIV_MUL;
                            end
                        end else begin
                            infer_state <= INFER_GEN_INPUT_SPIKES;
                            infer_input_idx <= 10'd0;
                            infer_trace_phase <= 2'd0;
                            infer_neuron_idx <= 7'd0;
                            infer_accum <= 32'sd0;
                            infer_poisson_thresh_rd_addr <= 10'd0;
                                infer_accum_weight_phase <= 3'd0;
                        end
                    end

                    INFER_PREP_DIV_MUL: begin
                        infer_dividend  <= infer_prep_div_prod_q32[31:0];
                        infer_div_valid <= 1'b1;
                        infer_state     <= INFER_PREP_DIV_WAIT;
                    end

                    INFER_PREP_DIV_WAIT: begin
                        if (infer_div_out_valid) begin
                            if (infer_div_q[31:12] != 0) begin
                                infer_poisson_thresh_wr_data <= RNG_MAX;
                            end else if (infer_div_q[11:0] > RNG_MAX) begin
                                infer_poisson_thresh_wr_data <= RNG_MAX;
                            end else begin
                                infer_poisson_thresh_wr_data <= infer_div_q[11:0];
                            end
                            infer_poisson_thresh_wr_en <= 1'b1;
                            infer_poisson_thresh_wr_addr <= infer_prep_idx;
                            if (infer_prep_idx == (N_IN - 1)) begin
                                infer_state <= INFER_GEN_INPUT_SPIKES;
                                infer_input_idx <= 10'd0;
                                infer_trace_phase <= 2'd0;
                                infer_neuron_idx <= 7'd0;
                                infer_accum <= 32'sd0;
                                infer_poisson_thresh_rd_addr <= 10'd0;
                                infer_accum_weight_phase <= 3'd0;
                            end else begin
                                infer_prep_idx <= infer_prep_idx + 10'd1;
                                raw_image0_rd_addr <= infer_prep_idx + 10'd1;
                                infer_state <= INFER_PREP_DIV_START;
                            end
                        end else if (infer_div_err) begin
                            infer_active <= 1'b0;
                            infer_state <= INFER_IDLE;
                            resp_status <= STATUS_BAD_PACKET;
                            resp_result <= 32'sd0;
                            resp_checksum  <= 8'h00;
                            response_ready <= 1'b1;
                        end
                    end

                    INFER_GEN_INPUT_SPIKES: begin
                        logic spike_in_now;
                        if (infer_trace_phase == 2'd0) begin
                            // Keep prelist write address/data updates unconditional in this phase
                            // to avoid deep CE gating on infer_pre_wr_* registers.
                            infer_pre_wr_addr <= infer_pre_active_count;
                            infer_pre_wr_data <= infer_input_idx;
                            infer_rng_mul_prod_q32 <= $unsigned(infer_rng_state) * LCG_A;
                            if (infer_force_no_input) begin
                                spike_in_now = 1'b0;
                            end else begin
                                // Compare against current state to keep LCG multiply off the
                                // same-cycle spike decision critical path.
                                spike_in_now = (infer_rng_state[31:21] < infer_poisson_thresh_rd_data);
                            end
                            infer_trace_spike_latched <= spike_in_now;
                            if (infer_input_idx == 10'd0) begin
                                infer_pre_active_count <= 10'd0;
                            end
                            train_xin_rd_addr <= infer_input_idx;
                            infer_trace_phase <= 3'd1;
                        end else if (infer_trace_phase == 2'd1) begin
                            if (!infer_force_no_input) begin
                                infer_rng_state <= infer_rng_mul_prod_q32[31:0] + LCG_C;
                            end
                            infer_pre_spike_wr_en <= 1'b1;
                            infer_pre_spike_wr_addr <= infer_input_idx;
                            infer_pre_spike_wr_data <= infer_trace_spike_latched;
                            if (infer_trace_spike_latched) begin
                                infer_pre_wr_en <= 1'b1;
                                infer_pre_active_count <= infer_pre_active_count + 10'd1;
                            end
                            infer_trace_phase <= 3'd2;
                        end else if (infer_trace_phase == 3'd2) begin
                            infer_apply_xin_prod <= $signed(train_xin_rd_data) * $signed(FXP_TRACE_PRE_DECAY);
                            infer_trace_phase <= 3'd3;
                        end else if (infer_trace_phase == 3'd3) begin
                            if (infer_apply_xin_prod >= 0) begin
                                infer_apply_xin_decay <= $signed((infer_apply_xin_prod + 64'sd32768) >>> 16);
                            end else begin
                                infer_apply_xin_decay <= $signed((infer_apply_xin_prod - 64'sd32768) >>> 16);
                            end
                            infer_trace_phase <= 3'd4;
                        end else if (infer_trace_phase == 3'd4) begin
                            infer_apply_xin_next <= infer_trace_spike_latched
                                                  ? FXP_TRACE_EVENT_SET
                                                  : infer_apply_xin_decay;
                            infer_trace_phase <= 3'd5;
                        end else begin
                            train_xin_wr_en <= 1'b1;
                            train_xin_wr_addr <= infer_input_idx;
                            train_xin_wr_data <= infer_apply_xin_next;
                            infer_trace_phase <= 3'd0;
                            if (infer_input_idx == (N_IN - 1)) begin
                                if (infer_step_idx == 16'd0) begin
                                end
                                infer_input_idx <= 10'd0;
                                infer_state <= INFER_ACCUM_NEURON;
                            end else begin
                                infer_input_idx <= infer_input_idx + 10'd1;
                                infer_poisson_thresh_rd_addr <= infer_input_idx + 10'd1;
                            end
                        end
                    end

                    INFER_ACCUM_NEURON: begin
                        // Sparse O(N_EDGES) accumulation: walk CSR edges for current neuron.
                        if (infer_accum_weight_phase == 3'd0) begin
                            csr_row_ptr_rd_addr <= infer_neuron_idx[ROW_IDX_W:0];
                            infer_accum_weight_phase <= 3'd1;
                        end else if (infer_accum_weight_phase == 3'd1) begin
                            infer_evt_edge_idx <= csr_row_ptr_rd_data[W_ADDR_W-1:0];
                            csr_row_ptr_rd_addr <= infer_neuron_idx[ROW_IDX_W:0] + {{ROW_IDX_W{1'b0}}, 1'b1};
                            infer_accum_weight_phase <= 3'd2;
                        end else if (infer_accum_weight_phase == 3'd2) begin
                            infer_evt_edge_end <= csr_row_ptr_rd_data[W_ADDR_W-1:0];
                            if (infer_evt_edge_idx >= csr_row_ptr_rd_data[W_ADDR_W-1:0]) begin
                                infer_delay_pipe_valid <= 1'b1;
                                infer_delay_pipe_idx <= infer_neuron_idx;
                                infer_accum_weight_phase <= 3'd0;
                                infer_state <= INFER_ACCUM_NEURON_GIN_MUL;
                            end else begin
                                csr_col_idx_rd_addr <= infer_evt_edge_idx;
                                if ((infer_evt_edge_idx + {{(W_ADDR_W-1){1'b0}}, 1'b1}) < csr_row_ptr_rd_data[W_ADDR_W-1:0]) begin
                                    csr_col_idx_rd_addr_lane1 <= infer_evt_edge_idx + {{(W_ADDR_W-1){1'b0}}, 1'b1};
                                    infer_accum_pair_count <= 2'd2;
                                end else begin
                                    csr_col_idx_rd_addr_lane1 <= infer_evt_edge_idx;
                                    infer_accum_pair_count <= 2'd1;
                                end
                                infer_accum_weight_phase <= 3'd3;
                            end
                        end else if (infer_accum_weight_phase == 3'd3) begin
                            infer_pre_spike_rd_addr <= {3'd0, csr_col_idx_rd_data};
                            infer_pre_spike_rd_addr_lane1 <= {3'd0, csr_col_idx_rd_data_lane1};
                            infer_accum_weight_phase <= 3'd4;
                        end else if (infer_accum_weight_phase == 3'd4) begin
                            infer_accum_lane0_fire <= infer_pre_spike_rd_data;
                            infer_accum_lane1_fire <= (infer_accum_pair_count == 2'd2) && infer_pre_spike_rd_data_lane1;
                            if (infer_pre_spike_rd_data) begin
                                infer_w_rd_addr <= infer_evt_edge_idx;
                            end
                            if ((infer_accum_pair_count == 2'd2) && infer_pre_spike_rd_data_lane1) begin
                                infer_w_rd_addr_lane1 <= infer_evt_edge_idx + {{(W_ADDR_W-1){1'b0}}, 1'b1};
                            end
                            infer_accum_weight_phase <= 3'd5;
                        end else if (infer_accum_weight_phase == 3'd5) begin
                            // weight BRAM read latency fill cycle.
                            infer_accum_weight_phase <= 3'd6;
                        end else begin
                            logic signed [31:0] accum_delta_q16;
                            logic [EDGE_ADDR_W-1:0] infer_evt_edge_next;
                            accum_delta_q16 = 32'sd0;
                            if (infer_accum_lane0_fire) begin
                                accum_delta_q16 = accum_delta_q16 + $signed({16'd0, infer_w_rd_data_q});
                            end
                            if (infer_accum_lane1_fire) begin
                                accum_delta_q16 = accum_delta_q16 + $signed({16'd0, infer_w_rd_data_q_lane1});
                            end
                            infer_accum <= infer_accum + accum_delta_q16;
                            infer_evt_edge_next = infer_evt_edge_idx + {{(W_ADDR_W-1){1'b0}}, infer_accum_pair_count};
                            infer_accum_lane0_fire <= 1'b0;
                            infer_accum_lane1_fire <= 1'b0;
                            if (infer_evt_edge_next >= infer_evt_edge_end) begin
                                infer_delay_pipe_valid <= 1'b1;
                                infer_delay_pipe_idx <= infer_neuron_idx;
                                infer_accum_weight_phase <= 3'd0;
                                infer_state <= INFER_ACCUM_NEURON_GIN_MUL;
                            end else begin
                                infer_evt_edge_idx <= infer_evt_edge_next;
                                csr_col_idx_rd_addr <= infer_evt_edge_next;
                                if ((infer_evt_edge_next + {{(W_ADDR_W-1){1'b0}}, 1'b1}) < infer_evt_edge_end) begin
                                    csr_col_idx_rd_addr_lane1 <= infer_evt_edge_next + {{(W_ADDR_W-1){1'b0}}, 1'b1};
                                    infer_accum_pair_count <= 2'd2;
                                end else begin
                                    csr_col_idx_rd_addr_lane1 <= infer_evt_edge_next;
                                    infer_accum_pair_count <= 2'd1;
                                end
                                infer_accum_weight_phase <= 3'd3;
                            end
                        end
                    end

                    INFER_ACCUM_NEURON_GIN_MUL: begin
                        // Register DSP output first to shorten the critical path into g_in pipeline regs.
                        infer_delay_pipe_mul_prod_q32 <= $signed(infer_accum) * $signed(FXP_SCALE_1000);
                        infer_state <= INFER_ACCUM_NEURON_GIN_MUL_ROUND;
                    end

                    INFER_ACCUM_NEURON_GIN_MUL_ROUND: begin
                        if (infer_delay_pipe_mul_prod_q32 >= 0) begin
                            infer_delay_pipe_mul_term <= $signed((infer_delay_pipe_mul_prod_q32 + 64'sd32768) >>> 16);
                        end else begin
                            infer_delay_pipe_mul_term <= $signed((infer_delay_pipe_mul_prod_q32 - 64'sd32768) >>> 16);
                        end
                        infer_state <= INFER_ACCUM_NEURON_GIN_COMB;
                    end

                    INFER_ACCUM_NEURON_GIN_COMB: begin
                        logic signed [31:0] g_in_state_next;
                        g_in_state_next = $signed(($signed(infer_g_in_state[infer_delay_pipe_idx]) * $signed(FXP_INPUT_G_DECAY)) >>> 16)
                                       + $signed(infer_delay_pipe_mul_term);
                        infer_g_in_state[infer_delay_pipe_idx] <= g_in_state_next;
                        infer_delay_pipe_g_in_curr <= g_in_state_next;
                        infer_delay_pipe_d0 <= infer_g_in_delay0[infer_delay_pipe_idx];
                        infer_delay_pipe_d1 <= infer_g_in_delay1[infer_delay_pipe_idx];
                        infer_delay_pipe_d2 <= infer_g_in_delay2[infer_delay_pipe_idx];
                        infer_delay_pipe_d3 <= infer_g_in_delay3[infer_delay_pipe_idx];
                        infer_delay_pipe_delayed_g_in <= infer_g_in_delay4[infer_delay_pipe_idx];
                        infer_accum <= 32'sd0;
                        infer_state <= INFER_ACCUM_NEURON_PIPE;
                    end

                    INFER_ACCUM_NEURON_PIPE: begin
                        if (infer_delay_pipe_valid) begin
                            infer_g_in_delay4[infer_delay_pipe_idx] <= infer_delay_pipe_d3;
                            infer_g_in_delay3[infer_delay_pipe_idx] <= infer_delay_pipe_d2;
                            infer_g_in_delay2[infer_delay_pipe_idx] <= infer_delay_pipe_d1;
                            infer_g_in_delay1[infer_delay_pipe_idx] <= infer_delay_pipe_d0;
                            infer_g_in_delay0[infer_delay_pipe_idx] <= infer_delay_pipe_g_in_curr;
                            infer_eval_idx <= infer_delay_pipe_idx;
                            infer_eval_v_cur <= infer_v_state[infer_delay_pipe_idx];
                            infer_eval_theta_cur <= infer_exc_theta[infer_delay_pipe_idx];
                            infer_eval_g_inh_cur <= infer_g_inh_state[infer_delay_pipe_idx];
                            infer_eval_delayed_g_in <= infer_delay_pipe_delayed_g_in;
                            infer_eval_last_spike_step <= infer_exc_last_spike_step[infer_delay_pipe_idx];
                            infer_delay_pipe_valid <= 1'b0;
                            infer_state <= INFER_NEURON_DV_PRE;
                        end else begin
                            infer_state <= INFER_ACCUM_NEURON;
                        end
                    end

                    INFER_NEURON_DV_PRE: begin
                        infer_eval_eexc_minus_v <= (FXP_EXC_EEXC - infer_eval_v_cur);
                        infer_eval_einh_minus_v <= (FXP_EXC_EINH - infer_eval_v_cur);
                        infer_eval_vrest_minus_v <= (FXP_EXC_VREST - infer_eval_v_cur);
                        infer_state <= INFER_NEURON_DV_DRIVE;
                    end

                    INFER_NEURON_DV_DRIVE: begin
                        logic signed [31:0] exc_drive_dt;
                        logic signed [31:0] inh_drive_dt;
                        logic signed [31:0] leak_dt;
                        logic exc_refractory_ok;

                        exc_refractory_ok = ((infer_step_idx - infer_eval_last_spike_step) > EXC_TREF_STEPS);
                        exc_drive_dt = fxp_mul_s16_16(infer_eval_eexc_minus_v, FXP_EXC_DT_OVER_TCM);
                        inh_drive_dt = fxp_mul_s16_16(infer_eval_einh_minus_v, FXP_EXC_DT_OVER_TCM);
                        leak_dt = fxp_mul_s16_16(infer_eval_vrest_minus_v, FXP_EXC_DT_OVER_TCM);
                        infer_eval_exc_refractory_ok <= exc_refractory_ok;
                        infer_eval_exc_drive_dt <= exc_drive_dt;
                        infer_eval_inh_drive_dt <= inh_drive_dt;
                        infer_eval_leak_dt <= leak_dt;
                        infer_state <= INFER_NEURON_DV_SYN;
                    end

                    INFER_NEURON_DV_SYN: begin
                        infer_eval_i_syn_exc_prod_q32 <= $signed(infer_eval_delayed_g_in) * $signed(infer_eval_exc_drive_dt);
                        infer_eval_i_syn_inh_prod_q32 <= $signed(infer_eval_g_inh_cur) * $signed(infer_eval_inh_drive_dt);
                        infer_state <= INFER_NEURON_DV_SYN_ROUND;
                    end

                    INFER_NEURON_DV_SYN_ROUND: begin
                        if (infer_eval_i_syn_exc_prod_q32 >= 0) begin
                            infer_eval_i_syn_exc_step <= $signed((infer_eval_i_syn_exc_prod_q32 + 64'sd32768) >>> 16);
                        end else begin
                            infer_eval_i_syn_exc_step <= $signed((infer_eval_i_syn_exc_prod_q32 - 64'sd32768) >>> 16);
                        end
                        if (infer_eval_i_syn_inh_prod_q32 >= 0) begin
                            infer_eval_i_syn_inh_step <= $signed((infer_eval_i_syn_inh_prod_q32 + 64'sd32768) >>> 16);
                        end else begin
                            infer_eval_i_syn_inh_step <= $signed((infer_eval_i_syn_inh_prod_q32 - 64'sd32768) >>> 16);
                        end
                        infer_state <= INFER_NEURON_VNEXT;
                    end

                    INFER_NEURON_VNEXT: begin
                        logic signed [31:0] dv_exc_step;
                        logic signed [31:0] v_next;
                        logic signed [31:0] v_prop;
                        dv_exc_step = infer_eval_leak_dt + infer_eval_i_syn_exc_step + infer_eval_i_syn_inh_step;
                        infer_eval_dv_exc_step <= dv_exc_step;
                        v_prop = infer_eval_v_cur + dv_exc_step;
                        v_next = infer_eval_exc_refractory_ok ? v_prop : infer_eval_v_cur;
                        infer_commit_idx <= infer_eval_idx;
                        infer_commit_v_next <= v_next;
                        infer_commit_thresh <= FXP_THRESH_BASE + infer_eval_theta_cur;
                        infer_state <= INFER_NEURON_SPIKE;
                    end

                    INFER_NEURON_SPIKE: begin
                        infer_commit_spike_now <= (infer_commit_v_next >= infer_commit_thresh);
                        infer_state <= INFER_NEURON_THETA_PRE;
                    end

                    INFER_NEURON_THETA_PRE: begin
                        infer_commit_theta_prod <= $signed(infer_eval_theta_cur) * $signed(FXP_THETA_DECAY);
                        infer_state <= INFER_NEURON_THETA_ROUND;
                    end

                    INFER_NEURON_THETA_ROUND: begin
                        if (infer_commit_theta_prod >= 0) begin
                            infer_commit_theta_decay <= $signed((infer_commit_theta_prod + 64'sd32768) >>> 16);
                        end else begin
                            infer_commit_theta_decay <= $signed((infer_commit_theta_prod - 64'sd32768) >>> 16);
                        end
                        infer_state <= INFER_NEURON_COMMIT;
                    end

                    INFER_NEURON_COMMIT: begin
                        logic signed [31:0] theta_next;
                        theta_next = infer_commit_theta_decay;
                        if (infer_commit_spike_now) begin
                            theta_next = theta_next + FXP_THETA_PLUS;
                        end
                        if (theta_next < 32'sd0) begin
                            theta_next = 32'sd0;
                        end
                        if (theta_next > FXP_THETA_MAX) begin
                            theta_next = FXP_THETA_MAX;
                        end
                        infer_spike_rd_addr <= infer_commit_idx;
                        infer_commit_theta_next <= theta_next;
                        infer_state <= INFER_NEURON_WRITE;
                    end

                    INFER_NEURON_WRITE: begin
                        infer_exc_theta[infer_commit_idx] <= infer_commit_theta_next;
                        if (infer_commit_spike_now) begin
                            // mine.py sets the membrane to vreset after spike (no residual carry).
                            infer_v_state[infer_commit_idx] <= FXP_EXC_VRESET;
                            spike_count_we <= 1'b1;
                            spike_count_waddr <= infer_commit_idx;
                            spike_count_wdata <= infer_spike_rd_data + 16'd1;
                            infer_total_spikes <= infer_total_spikes + 32'd1;
                            infer_exc_last_spike_step[infer_commit_idx] <= infer_step_idx;
                            infer_s_exc[infer_commit_idx] <= 1'b1;
                        end else begin
                            infer_v_state[infer_commit_idx] <= infer_commit_v_next;
                            infer_s_exc[infer_commit_idx] <= 1'b0;
                        end

                        if (infer_commit_idx == (N_NEURONS - 1)) begin
                            infer_neuron_idx <= 7'd0;
                            infer_apply_idx <= 7'd0;
                            infer_trace_phase <= 2'd0;
                            infer_sum_c_inh <= 32'sd0;
                            infer_state <= INFER_APPLY_WTA;
                        end else begin
                            infer_neuron_idx <= infer_commit_idx + 7'd1;
                            infer_state <= INFER_ACCUM_NEURON;
                        end
                    end

	                    INFER_APPLY_WTA: begin
	                        logic signed [31:0] g_exc_new;
	                        logic signed [31:0] delayed_g_exc;
                            logic signed [31:0] c_exc_next;
                            if (infer_trace_phase == 2'd0) begin
                                train_xexc_rd_addr <= infer_apply_idx;
                                train_xpost2_rd_addr <= infer_apply_idx;
                                infer_trace_spike_latched <= infer_s_exc[infer_apply_idx];
                                infer_trace_phase <= 2'd1;
                            end else if (infer_trace_phase == 2'd1) begin
                                infer_trace_phase <= 2'd2;
                            end else if (infer_trace_phase == 2'd2) begin
                                infer_apply_xexc_trace_q <= $signed(train_xexc_rd_data);
                                infer_apply_xpost2_trace_q <= $signed(train_xpost2_rd_data);
                                infer_post1_before[infer_apply_idx] <= $signed(train_xexc_rd_data);
                                infer_post2_before[infer_apply_idx] <= $signed(train_xpost2_rd_data);
                                infer_trace_phase <= 3'd3;
                            end else if (infer_trace_phase == 3'd3) begin
                                infer_apply_xexc_prod <= $signed(infer_apply_xexc_trace_q) * $signed(FXP_TRACE_POST1_DECAY);
                                infer_apply_xpost2_prod <= $signed(infer_apply_xpost2_trace_q) * $signed(FXP_TRACE_POST2_DECAY);
                                infer_trace_phase <= 3'd4;
                            end else if (infer_trace_phase == 3'd4) begin
                                if (infer_apply_xexc_prod >= 0) begin
                                    infer_apply_xexc_decay <= $signed((infer_apply_xexc_prod + 64'sd32768) >>> 16);
                                end else begin
                                    infer_apply_xexc_decay <= $signed((infer_apply_xexc_prod - 64'sd32768) >>> 16);
                                end
                                if (infer_apply_xpost2_prod >= 0) begin
                                    infer_apply_xpost2_decay <= $signed((infer_apply_xpost2_prod + 64'sd32768) >>> 16);
                                end else begin
                                    infer_apply_xpost2_decay <= $signed((infer_apply_xpost2_prod - 64'sd32768) >>> 16);
                                end
                                infer_trace_phase <= 3'd5;
                            end else if (infer_trace_phase == 3'd5) begin
                                infer_apply_xexc_next <= infer_trace_spike_latched
                                                       ? FXP_TRACE_EVENT_SET
                                                       : infer_apply_xexc_decay;
                                infer_apply_xpost2_next <= infer_trace_spike_latched
                                                         ? FXP_TRACE_EVENT_SET
                                                         : infer_apply_xpost2_decay;
                                infer_trace_phase <= 3'd6;
                            end else begin
                                train_xexc_wr_en <= 1'b1;
                                train_xexc_wr_addr <= infer_apply_idx;
                                train_xexc_wr_data <= infer_apply_xexc_next;
                                train_xpost2_wr_en <= 1'b1;
                                train_xpost2_wr_addr <= infer_apply_idx;
                                train_xpost2_wr_data <= infer_apply_xpost2_next;

                                // Match mine.py exc_synapse(td=1ms, dt=1ms): c_exc = 1000*s_exc (no decay carry).
                                c_exc_next = infer_trace_spike_latched ? FXP_SCALE_1000 : 32'sd0;
                                infer_c_exc_state[infer_apply_idx] <= c_exc_next;
	                            g_exc_new = fxp_mul_s16_16(c_exc_next, FXP_WEXC);
	                            delayed_g_exc = infer_g_exc_delay1[infer_apply_idx];
	                            infer_g_exc_delay1[infer_apply_idx] <= infer_g_exc_delay0[infer_apply_idx];
	                            infer_g_exc_delay0[infer_apply_idx] <= g_exc_new;
                                infer_apply_delayed_g_exc <= delayed_g_exc;
                                infer_apply_v_inh_cur <= infer_v_inh_state[infer_apply_idx];
                                infer_apply_c_inh_cur <= infer_c_inh_state[infer_apply_idx];
                                infer_apply_inh_last_spike <= infer_inh_last_spike_step[infer_apply_idx];
                                infer_trace_phase <= 3'd0;
                                infer_state <= INFER_APPLY_WTA_PRE;
                            end
	                    end

	                    INFER_APPLY_WTA_PRE: begin
                            infer_apply_inh_refractory_ok <= ((infer_step_idx - infer_apply_inh_last_spike) > INH_TREF_STEPS);
                            infer_apply_inh_eexc_minus_v <= (FXP_INH_EEXC - infer_apply_v_inh_cur);
                            infer_apply_inh_vrest_minus_v <= (FXP_INH_VREST - infer_apply_v_inh_cur);
                            infer_state <= INFER_APPLY_WTA_PRE_MUL;
                        end

	                    INFER_APPLY_WTA_PRE_MUL: begin
                            infer_apply_exc_drive_prod_q32 <= $signed(infer_apply_inh_eexc_minus_v) * $signed(FXP_INH_DT_OVER_TCM);
                            infer_apply_leak_prod_q32 <= $signed(infer_apply_inh_vrest_minus_v) * $signed(FXP_INH_DT_OVER_TCM);
                            infer_state <= INFER_APPLY_WTA_PRE_ROUND;
                        end

	                    INFER_APPLY_WTA_PRE_ROUND: begin
                            if (infer_apply_exc_drive_prod_q32 >= 0) begin
                                infer_apply_exc_drive_dt_inh <= $signed((infer_apply_exc_drive_prod_q32 + 64'sd32768) >>> 16);
                            end else begin
                                infer_apply_exc_drive_dt_inh <= $signed((infer_apply_exc_drive_prod_q32 - 64'sd32768) >>> 16);
                            end
                            if (infer_apply_leak_prod_q32 >= 0) begin
                                infer_apply_leak_dt_inh <= $signed((infer_apply_leak_prod_q32 + 64'sd32768) >>> 16);
                            end else begin
                                infer_apply_leak_dt_inh <= $signed((infer_apply_leak_prod_q32 - 64'sd32768) >>> 16);
                            end
                            infer_state <= INFER_APPLY_WTA_INH;
                        end

	                    INFER_APPLY_WTA_INH: begin
                            infer_apply_i_syn_exc_mul_a <= infer_apply_delayed_g_exc;
                            infer_apply_i_syn_exc_mul_b <= infer_apply_exc_drive_dt_inh;
                            infer_state <= INFER_APPLY_WTA_INH_PROD;
                        end

	                    INFER_APPLY_WTA_INH_PROD: begin
                            infer_apply_i_syn_exc_prod_q32 <= $signed(infer_apply_i_syn_exc_mul_a) * $signed(infer_apply_i_syn_exc_mul_b);
                            infer_state <= INFER_APPLY_WTA_INH_MUL;
                        end

	                    INFER_APPLY_WTA_INH_MUL: begin
                            if (infer_apply_i_syn_exc_prod_q32 >= 0) begin
                                infer_apply_i_syn_exc_step_inh <= $signed((infer_apply_i_syn_exc_prod_q32 + 64'sd32768) >>> 16);
                            end else begin
                                infer_apply_i_syn_exc_step_inh <= $signed((infer_apply_i_syn_exc_prod_q32 - 64'sd32768) >>> 16);
                            end
                            infer_state <= INFER_APPLY_WTA_INH_DV;
                        end

	                    INFER_APPLY_WTA_INH_DV: begin
                            infer_apply_dv_inh_step <= infer_apply_leak_dt_inh + infer_apply_i_syn_exc_step_inh;
                            infer_state <= INFER_APPLY_WTA_INH_VPROP;
                        end

	                    INFER_APPLY_WTA_INH_VPROP: begin
                            infer_apply_v_inh_prop <= $signed(infer_apply_v_inh_cur) + $signed(infer_apply_dv_inh_step);
                            infer_state <= INFER_APPLY_WTA_INH_POST;
                        end

	                    INFER_APPLY_WTA_INH_POST: begin
                            logic signed [31:0] v_inh_next;
	                        logic s_inh_now;
	                        logic signed [31:0] c_inh_next;
                            v_inh_next = infer_apply_inh_refractory_ok ? infer_apply_v_inh_prop : infer_apply_v_inh_cur;
	                        s_inh_now = (v_inh_next >= FXP_INH_THRESH);
                            infer_apply_s_inh_now <= s_inh_now;
                            infer_apply_v_inh_write <= s_inh_now ? FXP_INH_VRESET : v_inh_next;
	                        c_inh_next = $signed(infer_apply_c_inh_cur) >>> 1;
	                        if (s_inh_now) begin
	                            c_inh_next = c_inh_next + FXP_SCALE_500;
	                        end
                            infer_apply_c_inh_next <= c_inh_next;
                            infer_state <= INFER_APPLY_WTA_ACCUM;
                        end

	                    INFER_APPLY_WTA_ACCUM: begin
                            infer_v_inh_state[infer_apply_idx] <= infer_apply_v_inh_write;
                            infer_c_inh_state[infer_apply_idx] <= infer_apply_c_inh_next;
                            if (infer_apply_s_inh_now) begin
                                infer_inh_last_spike_step[infer_apply_idx] <= infer_step_idx;
                            end
	                        infer_sum_c_inh <= infer_sum_c_inh + infer_apply_c_inh_next;
	                        if (infer_apply_idx == (N_NEURONS - 1)) begin
	                            infer_apply_idx <= 7'd0;
                                infer_step_winner_valid <= 1'b0;
                                infer_step_winner_idx <= 7'd0;
	                            infer_state <= INFER_WTA_PASS2_PRE;
	                        end else begin
	                            infer_apply_idx <= infer_apply_idx + 7'd1;
                                infer_state <= INFER_APPLY_WTA;
	                        end
	                    end

	                    INFER_WTA_PASS2_PRE: begin
	                        logic signed [31:0] diff_c_inh;
                            if (!infer_step_winner_valid && infer_s_exc[infer_apply_idx]) begin
                                infer_step_winner_valid <= 1'b1;
                                infer_step_winner_idx   <= infer_apply_idx;
                            end
	                        diff_c_inh = infer_sum_c_inh - infer_c_inh_state[infer_apply_idx];
	                        if (diff_c_inh < 0) begin
	                            diff_c_inh = 32'sd0;
	                        end
                            infer_pass2_diff_c_inh <= diff_c_inh;
                            infer_state <= INFER_WTA_PASS2;
	                    end

	                    INFER_WTA_PASS2: begin
                            logic run_online_trace_now;
                            logic step_last_now;
                            run_online_trace_now = (TRAIN_ENABLE && train_chunk_active &&
                                                    (train_chunk_mode == 3'd3) &&
                                                    !infer_force_no_input);
                            step_last_now = ((infer_step_idx + 16'd1) >= infer_steps_target[15:0]);
	                        infer_pass2_g_inh_next <= fxp_mul_s16_16(infer_pass2_diff_c_inh, FXP_INH_COEFF);
                            infer_state <= INFER_WTA_PASS2_WRITE;
	                    end

	                    INFER_WTA_PASS2_WRITE: begin
                            logic run_online_trace_now;
                            logic step_last_now;
                            run_online_trace_now = (TRAIN_ENABLE && train_chunk_active &&
                                                    (train_chunk_mode == 3'd3) &&
                                                    !infer_force_no_input);
                            step_last_now = ((infer_step_idx + 16'd1) >= infer_steps_target[15:0]);
	                        infer_g_inh_state[infer_apply_idx] <= infer_pass2_g_inh_next;

	                        if (infer_apply_idx == (N_NEURONS - 1)) begin
                                if (run_online_trace_now) begin
                                    infer_evt_has_winner <= infer_step_winner_valid;
                                    infer_evt_winner_idx <= infer_step_winner_idx;
                                    infer_evt_prelist_idx <= 10'd0;
                                    infer_evt_pre_idx <= 10'd0;
                                    infer_evt_post_idx <= 7'd0;
                                    infer_evt_post_input_idx <= 10'd0;
                                    infer_evt_trace_val <= 32'sd0;
                                    infer_evt_w_cur <= 32'sd0;
                                    infer_evt_edge_idx <= '0;
                                    infer_evt_edge_end <= '0;
                                    infer_evt_edge_ptr <= '0;
                                    infer_trace_wait_last_step <= step_last_now;
                                    if (infer_pre_active_count != 10'd0) begin
                                        infer_state <= INFER_EVT_PRE_PRELIST_REQ;
                                    end else if (infer_step_winner_valid) begin
                                        infer_state <= INFER_EVT_POST_PTR0_REQ;
                                    end else begin
                                        infer_state <= INFER_EVT_DONE;
                                    end
                                end else if (step_last_now) begin
                                    // Match mine.py tcount semantics: increment at end of each processed step,
                                    // including the terminal step.
                                    infer_step_idx <= infer_step_idx + 16'd1;
	                                infer_active <= 1'b0;
	                                infer_state <= INFER_IDLE;
                                    infer_skip_init_clear <= 1'b0;
                                    infer_force_no_input  <= 1'b0;
                                    if (TRAIN_ENABLE && train_chunk_active &&
                                        ((train_chunk_state == TCK_INFER_WAIT) || (train_chunk_state == TCK_BLANK_INFER_WAIT))) begin
                                        // Sub-step completion for TRAIN_RUN_CHUNK phase2: do not emit host response here.
                                    end else if (batch_active) begin
                                        batch_infer_eval_active <= 1'b1;
                                        batch_infer_eval_state <= BIE_RESET;
                                        batch_infer_eval_idx <= 7'd0;
                                        batch_infer_eval_label_idx <= 4'd0;
                                        batch_infer_eval_best_label <= 4'd0;
                                        batch_infer_eval_best_sum <= 32'd0;
                                        batch_infer_eval_best_count <= 16'd0;
                                        batch_infer_eval_pred_label <= 4'd0;
                                    end else begin
	                                    resp_status <= STATUS_OK;
	                                    resp_result <= infer_total_spikes;
	                                    resp_checksum  <= 8'h00;
	                                    response_ready <= 1'b1;
                                    end
	                            end else begin
	                                    infer_step_idx <= infer_step_idx + 16'd1;
	                                    infer_state <= INFER_GEN_INPUT_SPIKES;
                                        infer_trace_phase <= 2'd0;
	                            end
	                            infer_apply_idx <= 7'd0;
	                            infer_neuron_idx <= 7'd0;
	                            infer_input_idx <= 10'd0;
	                            infer_accum <= 32'sd0;
                                infer_accum_weight_phase <= 3'd0;
	                        end else begin
	                            infer_apply_idx <= infer_apply_idx + 7'd1;
                                infer_state <= INFER_WTA_PASS2_PRE;
	                        end
	                    end

                    INFER_EVT_PRE_PRELIST_REQ: begin
                        infer_pre_rd_addr <= infer_evt_prelist_idx;
                        infer_state <= INFER_EVT_PRE_PRELIST_WAIT;
                    end

                    INFER_EVT_PRE_PRELIST_WAIT: begin
                        infer_evt_pre_idx <= infer_pre_rd_data;
                        infer_state <= INFER_EVT_PRE_PTR0_REQ;
                    end

                    INFER_EVT_PRE_PTR0_REQ: begin
                        csc_col_ptr_rd_addr <= {1'b0, infer_evt_pre_idx};
                        infer_state <= INFER_EVT_PRE_PTR0_WAIT;
                    end

                    INFER_EVT_PRE_PTR0_WAIT: begin
                        infer_evt_edge_idx <= csc_col_ptr_rd_data[EDGE_ADDR_W-1:0];
                        infer_state <= INFER_EVT_PRE_PTR1_REQ;
                    end

                    INFER_EVT_PRE_PTR1_REQ: begin
                        csc_col_ptr_rd_addr <= {1'b0, infer_evt_pre_idx} + {{COL_IDX_W{1'b0}}, 1'b1};
                        infer_state <= INFER_EVT_PRE_PTR1_WAIT;
                    end

                    INFER_EVT_PRE_PTR1_WAIT: begin
                        infer_evt_edge_end <= csc_col_ptr_rd_data[EDGE_ADDR_W-1:0];
                        if (infer_evt_edge_idx >= csc_col_ptr_rd_data[EDGE_ADDR_W-1:0]) begin
                            if ((infer_evt_prelist_idx + 10'd1) >= infer_pre_active_count) begin
                                if (infer_evt_has_winner) begin
                                    infer_state <= INFER_EVT_POST_PTR0_REQ;
                                end else begin
                                    infer_state <= INFER_EVT_DONE;
                                end
                            end else begin
                                infer_evt_prelist_idx <= infer_evt_prelist_idx + 10'd1;
                                infer_state <= INFER_EVT_PRE_PRELIST_REQ;
                            end
                        end else begin
                            infer_state <= INFER_EVT_PRE_EDGE_REQ;
                        end
                    end

                    INFER_EVT_PRE_EDGE_REQ: begin
                        csc_row_idx_rd_addr <= infer_evt_edge_idx;
                        csc_edge_idx_rd_addr <= infer_evt_edge_idx;
                        infer_state <= INFER_EVT_PRE_EDGE_WAIT;
                    end

                    INFER_EVT_PRE_EDGE_WAIT: begin
                        infer_evt_post_idx <= csc_row_idx_rd_data;
                        infer_evt_edge_ptr <= csc_edge_idx_rd_data;
                        infer_state <= INFER_EVT_PRE_TRACE_WAIT;
                    end

                    INFER_EVT_PRE_TRACE_REQ: begin
                        infer_state <= INFER_EVT_PRE_TRACE_WAIT;
                    end

                    INFER_EVT_PRE_TRACE_WAIT: begin
                        infer_evt_trace_val <= infer_post1_before[infer_evt_post_idx];
                        infer_w_rd_addr <= infer_evt_edge_ptr;
                        infer_state <= INFER_EVT_PRE_W_WAIT;
                    end

                    INFER_EVT_PRE_W_WAIT: begin
                        infer_evt_w_cur <= $signed({16'd0, infer_w_rd_data});
                        infer_state <= INFER_EVT_PRE_APPLY;
                    end

                    INFER_EVT_PRE_APPLY: begin
                        infer_evt_mid_q16 <= TRAIN_LR_M_Q16;
                        infer_state <= INFER_EVT_PRE_APPLY_MUL1;
                    end

                    INFER_EVT_PRE_APPLY_MUL1: begin
                        infer_evt_term_prod_q32 <= $signed(infer_evt_mid_q16) * $signed(infer_evt_trace_val);
                        infer_state <= INFER_EVT_PRE_APPLY_MUL2;
                    end

                    INFER_EVT_PRE_APPLY_MUL2: begin
                        if (infer_evt_term_prod_q32 >= 0)
                            infer_evt_term_q16 <= $signed((infer_evt_term_prod_q32 + 64'sd32768) >>> 16);
                        else
                            infer_evt_term_q16 <= $signed((infer_evt_term_prod_q32 - 64'sd32768) >>> 16);
                        infer_state <= INFER_EVT_PRE_APPLY_CLIP;
                    end

                    INFER_EVT_PRE_APPLY_CLIP: begin
                        logic signed [31:0] dW_q16;
                        dW_q16 = -infer_evt_term_q16;
                        infer_evt_dw_q16 <= dW_q16;
                        infer_state <= INFER_EVT_PRE_APPLY_WNEXT;
                    end

                    INFER_EVT_PRE_APPLY_WNEXT: begin
                        logic signed [31:0] w_next_q16;
                        w_next_q16 = infer_evt_w_cur + infer_evt_dw_q16;
                        if (w_next_q16 > TRAIN_WMAX_Q16)
                            w_next_q16 = TRAIN_WMAX_Q16;
                        else if (w_next_q16 < TRAIN_WMIN_Q16)
                            w_next_q16 = TRAIN_WMIN_Q16;
                        infer_evt_w_next_q16 <= w_next_q16;
                        if (w_next_q16[15:0] != infer_evt_w_cur[15:0]) begin
                            infer_w_wr_en   <= 1'b1;
                            infer_w_wr_addr <= infer_evt_edge_ptr;
                            infer_w_wr_data <= w_next_q16[15:0];
                        end
                        if ((infer_evt_edge_idx + {{(EDGE_ADDR_W-1){1'b0}},1'b1}) >= infer_evt_edge_end) begin
                            if ((infer_evt_prelist_idx + 10'd1) >= infer_pre_active_count) begin
                                if (infer_evt_has_winner) begin
                                    infer_state <= INFER_EVT_POST_PTR0_REQ;
                                end else begin
                                    infer_state <= INFER_EVT_DONE;
                                end
                            end else begin
                                infer_evt_prelist_idx <= infer_evt_prelist_idx + 10'd1;
                                infer_state <= INFER_EVT_PRE_PRELIST_REQ;
                            end
                        end else begin
                            infer_evt_edge_idx <= infer_evt_edge_idx + {{(EDGE_ADDR_W-1){1'b0}},1'b1};
                            infer_state <= INFER_EVT_PRE_EDGE_REQ;
                        end
                    end

                    INFER_EVT_POST_PTR0_REQ: begin
                        csr_row_ptr_rd_addr <= infer_evt_winner_idx[ROW_IDX_W:0];
                        infer_state <= INFER_EVT_POST_PTR0_WAIT;
                    end

                    INFER_EVT_POST_PTR0_WAIT: begin
                        infer_evt_edge_idx <= csr_row_ptr_rd_data[EDGE_ADDR_W-1:0];
                        infer_state <= INFER_EVT_POST_PTR1_REQ;
                    end

                    INFER_EVT_POST_PTR1_REQ: begin
                        csr_row_ptr_rd_addr <= infer_evt_winner_idx[ROW_IDX_W:0] + {{ROW_IDX_W{1'b0}}, 1'b1};
                        infer_state <= INFER_EVT_POST_PTR1_WAIT;
                    end

                    INFER_EVT_POST_PTR1_WAIT: begin
                        infer_evt_edge_end <= csr_row_ptr_rd_data[EDGE_ADDR_W-1:0];
                        if (infer_evt_edge_idx >= csr_row_ptr_rd_data[EDGE_ADDR_W-1:0]) begin
                            infer_state <= INFER_EVT_DONE;
                        end else begin
                            infer_state <= INFER_EVT_POST_EDGE_REQ;
                        end
                    end

                    INFER_EVT_POST_EDGE_REQ: begin
                        csr_col_idx_rd_addr <= infer_evt_edge_idx;
                        infer_state <= INFER_EVT_POST_EDGE_WAIT;
                    end

                    INFER_EVT_POST_EDGE_WAIT: begin
                        infer_evt_post_input_idx <= {3'd0, csr_col_idx_rd_data};
                        infer_evt_edge_ptr <= infer_evt_edge_idx;
                        infer_state <= INFER_EVT_POST_TRACE_REQ;
                    end

                    INFER_EVT_POST_TRACE_REQ: begin
                        train_xin_rd_addr <= infer_evt_post_input_idx;
                        infer_state <= INFER_EVT_POST_TRACE_WAIT;
                    end

                    INFER_EVT_POST_TRACE_WAIT: begin
                        infer_evt_trace_val <= $signed(train_xin_rd_data);
                        infer_evt_post2_before_q <= infer_post2_before[infer_evt_winner_idx];
                        infer_w_rd_addr <= infer_evt_edge_ptr;
                        infer_state <= INFER_EVT_POST_W_WAIT;
                    end

                    INFER_EVT_POST_W_WAIT: begin
                        infer_evt_w_cur <= $signed({16'd0, infer_w_rd_data});
                        infer_state <= INFER_EVT_POST_APPLY;
                    end

                    INFER_EVT_POST_APPLY: begin
                        infer_evt_mid_q16 <= fxp_mul_s16_16(TRAIN_LR_P_Q16, infer_evt_trace_val);
                        infer_state <= INFER_EVT_POST_APPLY_MUL1;
                    end

                    INFER_EVT_POST_APPLY_MUL1: begin
                        infer_evt_term_prod_q32 <= $signed(infer_evt_mid_q16) * $signed(infer_evt_post2_before_q);
                        infer_state <= INFER_EVT_POST_APPLY_MUL2;
                    end

                    INFER_EVT_POST_APPLY_MUL2: begin
                        if (infer_evt_term_prod_q32 >= 0)
                            infer_evt_term_q16 <= $signed((infer_evt_term_prod_q32 + 64'sd32768) >>> 16);
                        else
                            infer_evt_term_q16 <= $signed((infer_evt_term_prod_q32 - 64'sd32768) >>> 16);
                        infer_state <= INFER_EVT_POST_APPLY_CLIP;
                    end

                    INFER_EVT_POST_APPLY_CLIP: begin
                        logic signed [31:0] dW_q16;
                        dW_q16 = infer_evt_term_q16;
                        infer_evt_dw_q16 <= dW_q16;
                        infer_state <= INFER_EVT_POST_APPLY_WNEXT;
                    end

                    INFER_EVT_POST_APPLY_WNEXT: begin
                        logic signed [31:0] w_next_q16;
                        w_next_q16 = infer_evt_w_cur + infer_evt_dw_q16;
                        if (w_next_q16 > TRAIN_WMAX_Q16)
                            w_next_q16 = TRAIN_WMAX_Q16;
                        else if (w_next_q16 < TRAIN_WMIN_Q16)
                            w_next_q16 = TRAIN_WMIN_Q16;
                        infer_evt_w_next_q16 <= w_next_q16;
                        if (w_next_q16[15:0] != infer_evt_w_cur[15:0]) begin
                            infer_w_wr_en   <= 1'b1;
                            infer_w_wr_addr <= infer_evt_edge_ptr;
                            infer_w_wr_data <= w_next_q16[15:0];
                        end
                        if ((infer_evt_edge_idx + {{(EDGE_ADDR_W-1){1'b0}},1'b1}) >= infer_evt_edge_end) begin
                            infer_state <= INFER_EVT_DONE;
                        end else begin
                            infer_evt_edge_idx <= infer_evt_edge_idx + {{(EDGE_ADDR_W-1){1'b0}},1'b1};
                            infer_state <= INFER_EVT_POST_EDGE_REQ;
                        end
                    end

                    INFER_EVT_DONE: begin
                        if (infer_trace_wait_last_step) begin
                            infer_trace_wait_last_step <= 1'b0;
                            infer_step_idx <= infer_step_idx + 16'd1;
                            infer_active <= 1'b0;
                            infer_state <= INFER_IDLE;
                            infer_skip_init_clear <= 1'b0;
                            infer_force_no_input  <= 1'b0;
                            if (TRAIN_ENABLE && train_chunk_active &&
                                ((train_chunk_state == TCK_INFER_WAIT) || (train_chunk_state == TCK_BLANK_INFER_WAIT))) begin
                                // Sub-step completion for TRAIN_RUN_CHUNK phase flows.
                            end else if (batch_active) begin
                                batch_infer_eval_active <= 1'b1;
                                batch_infer_eval_state <= BIE_RESET;
                                batch_infer_eval_idx <= 7'd0;
                                batch_infer_eval_label_idx <= 4'd0;
                                batch_infer_eval_best_label <= 4'd0;
                                batch_infer_eval_best_sum <= 32'd0;
                                batch_infer_eval_best_count <= 16'd0;
                                batch_infer_eval_pred_label <= 4'd0;
                            end else begin
                                resp_status <= STATUS_OK;
                                resp_result <= infer_total_spikes;
                                resp_checksum  <= 8'h00;
                                response_ready <= 1'b1;
                            end
                        end else begin
                            infer_step_idx <= infer_step_idx + 16'd1;
                            infer_state <= INFER_GEN_INPUT_SPIKES;
                            infer_trace_phase <= 3'd0;
                        end
                    end

                    default: begin
                        infer_state <= INFER_IDLE;
                    end
                endcase
            end

            if (batch_infer_eval_active && !response_ready && !infer_active && !train_chunk_active && !train_label_stats_active) begin
                case (batch_infer_eval_state)
                    BIE_RESET: begin
                        batch_infer_eval_sum[batch_infer_eval_label_idx] <= 32'd0;
                        batch_infer_eval_count[batch_infer_eval_label_idx] <= 16'd0;
                        if (batch_infer_eval_label_idx == 4'd9) begin
                            batch_infer_eval_idx <= 7'd0;
                            batch_infer_eval_state <= BIE_READ;
                        end else begin
                            batch_infer_eval_label_idx <= batch_infer_eval_label_idx + 4'd1;
                        end
                    end
                    BIE_READ: begin
                        infer_spike_rd_addr <= batch_infer_eval_idx;
                        batch_assign_rd_addr <= batch_infer_eval_idx;
                        batch_infer_eval_state <= BIE_WAIT;
                    end
                    BIE_WAIT: begin
                        batch_infer_eval_state <= BIE_ACCUM;
                    end
                    BIE_ACCUM: begin
                        batch_infer_eval_spike_q <= infer_spike_rd_data;
                        batch_infer_eval_assign_q <= batch_assign_rd_data;
                        batch_infer_eval_sum[batch_assign_rd_data] <= batch_infer_eval_sum[batch_assign_rd_data] + {16'd0, infer_spike_rd_data};
                        batch_infer_eval_count[batch_assign_rd_data] <= batch_infer_eval_count[batch_assign_rd_data] + 16'd1;
                        if (batch_infer_eval_idx == (N_NEURONS - 1)) begin
                            batch_infer_eval_label_idx <= 4'd0;
                            batch_infer_eval_best_label <= 4'd0;
                            batch_infer_eval_best_sum <= 32'd0;
                            batch_infer_eval_best_count <= 16'd0;
                            batch_infer_eval_cmp_lhs <= 64'd0;
                            batch_infer_eval_cmp_rhs <= 64'd0;
                            batch_infer_eval_state <= BIE_SELECT_PREP;
                        end else begin
                            batch_infer_eval_idx <= batch_infer_eval_idx + 7'd1;
                            batch_infer_eval_state <= BIE_READ;
                        end
                    end
                    BIE_SELECT_PREP: begin
                        if (batch_infer_eval_count[batch_infer_eval_label_idx] != 16'd0) begin
                            batch_infer_eval_curr_sum_q <= batch_infer_eval_sum[batch_infer_eval_label_idx];
                            batch_infer_eval_curr_count_q <= batch_infer_eval_count[batch_infer_eval_label_idx];
                            batch_infer_eval_best_count_eff_q <= (batch_infer_eval_best_count == 16'd0) ? 16'd1
                                                                                                           : batch_infer_eval_best_count;
                        end else begin
                            batch_infer_eval_curr_sum_q <= 32'd0;
                            batch_infer_eval_curr_count_q <= 16'd0;
                            batch_infer_eval_best_count_eff_q <= 16'd0;
                        end
                        batch_infer_eval_state <= BIE_SELECT_MUL;
                    end
                    BIE_SELECT_MUL: begin
                        batch_infer_eval_cmp_lhs <= batch_infer_eval_curr_sum_q * batch_infer_eval_best_count_eff_q;
                        batch_infer_eval_cmp_rhs <= batch_infer_eval_best_sum * batch_infer_eval_curr_count_q;
                        batch_infer_eval_state <= BIE_SELECT;
                    end
                    BIE_SELECT: begin
                        if (batch_infer_eval_curr_count_q != 16'd0) begin
                            if ((batch_infer_eval_best_count == 16'd0) ||
                                (batch_infer_eval_cmp_lhs > batch_infer_eval_cmp_rhs)) begin
                                batch_infer_eval_best_label <= batch_infer_eval_label_idx;
                                batch_infer_eval_best_sum <= batch_infer_eval_curr_sum_q;
                                batch_infer_eval_best_count <= batch_infer_eval_curr_count_q;
                            end
                        end
                        if (batch_infer_eval_label_idx == 4'd9) begin
                            batch_infer_eval_pred_label <= batch_infer_eval_best_label;
                            batch_infer_eval_state <= BIE_DONE;
                        end else begin
                            batch_infer_eval_label_idx <= batch_infer_eval_label_idx + 4'd1;
                            batch_infer_eval_state <= BIE_SELECT_PREP;
                        end
                    end
                    BIE_DONE: begin
                        batch_infer_eval_active <= 1'b0;
                        batch_infer_eval_state <= BIE_IDLE;
                        batch_processed_samples <= batch_processed_samples + 32'd1;
                        batch_total_spikes <= batch_total_spikes + infer_total_spikes;
                        if (batch_infer_eval_best_label == batch_label_rd_data[3:0]) begin
                            batch_correct_count <= batch_correct_count + 32'd1;
                        end
                        if ((batch_processed_samples + 32'd1) >= batch_cfg_num_samples) begin
                            batch_active <= 1'b0;
                            batch_done <= 1'b1;
                            batch_error <= 1'b0;
                            batch_error_code <= BATCH_ERR_NONE;
                            batch_phase <= BATCH_PHASE_DONE;
                            batch_current_sample_idx <= batch_current_sample_idx + 32'd1;
                        end else begin
                            batch_current_sample_idx <= batch_current_sample_idx + 32'd1;
                            batch_label_rd_addr <= batch_current_sample_idx + 32'd1;
                            if (batch_use_cached_images) begin
                                batch_phase <= BATCH_PHASE_LOADING;
                                batch_cache_img_byte_off <= batch_cache_img_byte_off + RAW1_BYTES_PER_IMAGE;
                                if (!batch_compute_buf_sel) begin
                                    raw_image0_valid <= 1'b0;
                                    raw_image0_capture_idx <= 10'd0;
                                    raw_image0_sum_u8 <= 32'd0;
                                end else begin
                                    raw_image1_valid <= 1'b0;
                                    raw_image1_capture_idx <= 10'd0;
                                    raw_image1_sum_u8 <= 32'd0;
                                end
                            end else if (batch_prefetch_ready && (batch_prefetch_sample_idx == (batch_current_sample_idx + 32'd1))) begin
                                batch_phase <= BATCH_PHASE_RUNNING;
                                batch_img_byte_off <= batch_prefetch_img_byte_off;
                                batch_img_sector_off <= batch_prefetch_img_sector_off;
                                batch_img_byte_in_sector <= batch_prefetch_img_byte_in_sector;
                                batch_img_sectors_needed <= batch_prefetch_img_sectors_needed;
                                batch_compute_buf_sel <= batch_fill_buf_sel;
                                batch_fill_buf_sel <= batch_compute_buf_sel;
                                batch_prefetch_ready <= 1'b0;
                                if (!batch_compute_buf_sel) begin
                                    raw_image0_valid <= 1'b0;
                                    raw_image0_capture_idx <= 10'd0;
                                    raw_image0_sum_u8 <= 32'd0;
                                end else begin
                                    raw_image1_valid <= 1'b0;
                                    raw_image1_capture_idx <= 10'd0;
                                    raw_image1_sum_u8 <= 32'd0;
                                end
                                infer_active       <= 1'b1;
                                infer_state        <= INFER_INIT_CLEAR;
                                infer_steps_target <= 32'd350;
                                infer_step_idx     <= 16'd0;
                                infer_neuron_idx   <= 7'd0;
                                infer_input_idx    <= 10'd0;
                                infer_prep_idx     <= 10'd0;
                                infer_accum        <= 32'sd0;
                                infer_accum_weight_phase <= 3'd0;
                                infer_apply_idx    <= 7'd0;
                                infer_trace_phase  <= 2'd0;
                                infer_total_spikes <= 32'd0;
                                infer_rng_state    <= batch_cfg_seed;
                                infer_skip_init_clear <= 1'b0;
                                infer_force_no_input  <= 1'b0;
                                infer_poisson_num_const_cfg <= POISSON_NUM_CONST;
                                infer_pre_active_count <= 10'd0;
                                raw_image0_rd_addr <= 10'd0;
                                infer_poisson_thresh_rd_addr <= 10'd0;
                                infer_model_state_valid <= 1'b0;
                            end else begin
                                logic [31:0] next_img_byte_off_tmp;
                                logic [31:0] next_img_byte_in_sector_tmp;
                                logic [31:0] next_img_byte_in_sector_norm_tmp;
                                logic [31:0] next_img_sector_carry_tmp;
                                logic [31:0] next_img_sector_off_tmp;
                                logic [31:0] next_img_sectors_needed_tmp;
                                next_img_byte_off_tmp = batch_img_byte_off + RAW1_BYTES_PER_IMAGE;
                                next_img_byte_in_sector_tmp = batch_img_byte_in_sector + RAW1_BYTES_PER_IMAGE;
                                if (next_img_byte_in_sector_tmp >= 32'd1024) begin
                                    next_img_sector_carry_tmp = 32'd2;
                                    next_img_byte_in_sector_norm_tmp = next_img_byte_in_sector_tmp - 32'd1024;
                                end else if (next_img_byte_in_sector_tmp >= 32'd512) begin
                                    next_img_sector_carry_tmp = 32'd1;
                                    next_img_byte_in_sector_norm_tmp = next_img_byte_in_sector_tmp - 32'd512;
                                end else begin
                                    next_img_sector_carry_tmp = 32'd0;
                                    next_img_byte_in_sector_norm_tmp = next_img_byte_in_sector_tmp;
                                end
                                next_img_sector_off_tmp = batch_img_sector_off + next_img_sector_carry_tmp;
                                next_img_sectors_needed_tmp = (next_img_byte_in_sector_norm_tmp <= 32'd240) ? 32'd2 : 32'd3;
                                batch_phase <= BATCH_PHASE_LOADING;
                                batch_img_byte_off <= next_img_byte_off_tmp;
                                batch_img_sector_off <= next_img_sector_off_tmp;
                                batch_img_byte_in_sector <= next_img_byte_in_sector_norm_tmp;
                                batch_img_sectors_needed <= next_img_sectors_needed_tmp;
                                sd_copy_active        <= 1'b1;
                                sd_in_read            <= 1'b0;
                                sd_copy_lba           <= batch_cfg_start_lba + next_img_sector_off_tmp;
                                sd_copy_sectors_left  <= next_img_sectors_needed_tmp;
                                sd_byte_count         <= 9'd0;
                                sd_pack_idx           <= 2'd0;
                                sd_pack_word          <= 32'd0;
                                sd_copy_words_written <= 32'd0;
                                sd_sector_buf_ready   <= 2'b00;
                                sd_header_done        <= 1'b0;
                                sd_file_total_bytes   <= 32'd0;
                                sd_file_bytes_seen    <= 32'd0;
                                sd_copy_done_pending  <= 1'b0;
                                sd_use_sector_limit   <= 1'b1;
                                sd_copy_raw1_mode     <= 1'b0;
                                sd_copy_dest_base_word <= IMG_STAGING_BASE_WORD;
                                if (!batch_compute_buf_sel) begin
                                    raw_image0_valid <= 1'b0;
                                    raw_image0_capture_idx <= 10'd0;
                                    raw_image0_sum_u8 <= 32'd0;
                                end else begin
                                    raw_image1_valid <= 1'b0;
                                    raw_image1_capture_idx <= 10'd0;
                                    raw_image1_sum_u8 <= 32'd0;
                                end
                            end
                        end
                    end
                    default: begin
                        batch_infer_eval_active <= 1'b0;
                        batch_infer_eval_state <= BIE_IDLE;
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
                        3'd6: tx_byte <= calc_resp_checksum(resp_status, resp_result);
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
 
// reset the default net type to wire, sometimes other code expects this.
`default_nettype wire




