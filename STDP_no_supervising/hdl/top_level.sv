`default_nettype none

module top_level(
        input  wire        clk_100mhz,
        input  wire [15:0] sw,  // unused
        input  wire [3:0]  btn,
        output logic [15:0] led,
        output logic [2:0] rgb0,
        output logic [2:0] rgb1,
        output logic [3:0] ss0_an,
        output logic [3:0] ss1_an,
        output logic [6:0] ss0_c,
        output logic [6:0] ss1_c,

        // SD card socket signals (SPI mode)
        input  wire SD_DQ0,   // MISO
        output wire SD_DQ1,   // hold HIGH
        output wire SD_DQ2,   // hold HIGH
        output wire SD_DQ3,   // CS
        output wire SD_CMD,   // MOSI
        output wire SD_CLK,   // SCLK
        input  wire SD_CD_N   // card detect, active low
    );

    // -----------------------------
    // Clocks and reset
    // -----------------------------
    // Power-on reset in 100 MHz domain (no external reset pin)
    localparam int POR_CYCLES = 16'd1024;
    logic [15:0] por_cnt_100;
    logic por_reset_100;
    always_ff @(posedge clk_100mhz) begin
        if (por_cnt_100 != POR_CYCLES) begin
            por_cnt_100 <= por_cnt_100 + 1'b1;
            por_reset_100 <= 1'b1;
        end else begin
            por_reset_100 <= 1'b0;
        end
    end

    // 25 MHz clock from 100 MHz input (divide by 4)
    logic [1:0] clk_div;
    always_ff @(posedge clk_100mhz) begin
        if (por_reset_100) begin
            clk_div <= 2'b0;
        end else begin
            clk_div <= clk_div + 2'b1;
        end
    end
    wire clk_25mhz = clk_div[1];

    // Synchronize reset into 25 MHz domain
    logic [1:0] por_sync_25;
    always_ff @(posedge clk_25mhz) begin
        por_sync_25 <= {por_sync_25[0], por_reset_100};
    end
    wire reset_por = por_sync_25[1];

    // Soft reset from btn[0] (synchronous, stretched)
    logic [7:0] soft_reset_cnt;
    logic soft_reset;

    wire reset = reset_por | soft_reset;

    localparam int N_IN = 784;
    localparam int N_NEURONS = 100;
    localparam int TSTEP_W = 16;
    localparam int BLANK_STEPS = 150;
    localparam int N_LABELS = 10;
    localparam int LABEL_BITS = 4;
    localparam int N_SAMPLES = 10000;
    localparam int TRAIN_SAMPLES = 9000;
    localparam int EVAL_SAMPLES = 1000;
    localparam int POS_LABEL = 0;

    // -----------------------------
    // SD controller instance
    // -----------------------------
    logic        rd;
    logic        wr;
    logic [31:0] address;
    logic [7:0]  dout;
    logic        byte_available;
    logic        ready;
    logic [4:0]  status;

    sd_controller u_sd(
        .cs                 (SD_DQ3),
        .mosi               (SD_CMD),
        .miso               (SD_DQ0),
        .sclk               (SD_CLK),
        .rd                 (rd),
        .dout               (dout),
        .byte_available     (byte_available),
        .wr                 (wr),
        .din                (8'h00),
        .ready_for_next_byte(),
        .reset              (reset),
        .ready              (ready),
        .address            (address),
        .clk                (clk_25mhz),
        .status             (status)
    );

    assign SD_DQ1 = 1'b1; // SPI mode: DAT1/2 should be HIGH
    assign SD_DQ2 = 1'b1;

    // -----------------------------
    // Streamed read of SPK1 file from SD (starting at LBA 2048)
    // -----------------------------
    localparam int SECTOR_BYTES = 512;
    localparam int LBA_START    = 32'd2048;
    localparam int HEADER_BYTES = 20;

    logic [8:0]  byte_count;
    logic        in_read;
    logic [31:0] sector_addr;
    logic [31:0] file_byte_index;

    // Header parsing
    logic [7:0]  header_bytes [0:19];
    logic [31:0] num_images_u32;
    logic [31:0] n_time_u32;
    logic [31:0] n_neurons_u32;
    logic        header_done;

    // Labels
    logic [31:0] label_index;

    logic        streaming;

    // Buttons (sync + edge) in clk_25mhz domain
    logic [3:0] btn_sync;
    logic [3:0] btn_prev;
    wire btn0_rise = btn_sync[0] & ~btn_prev[0];
    wire btn1_rise = btn_sync[1] & ~btn_prev[1];
    wire btn2_rise = btn_sync[2] & ~btn_prev[2];
    wire btn3_rise = btn_sync[3] & ~btn_prev[3];

    always_ff @(posedge clk_25mhz) begin
        if (reset_por) begin
            btn_sync <= 4'b0;
            btn_prev <= 4'b0;
            soft_reset_cnt <= 8'd0;
            soft_reset <= 1'b0;
        end else begin
            btn_sync <= {btn[3], btn[2], btn[1], btn[0]};
            btn_prev <= btn_sync;
            if (btn0_rise) begin
                soft_reset_cnt <= 8'd64;
            end else if (soft_reset_cnt != 0) begin
                soft_reset_cnt <= soft_reset_cnt - 1'b1;
            end
            soft_reset <= (soft_reset_cnt != 0);
        end
    end

    typedef enum logic [2:0] {
        RUN_IDLE,
        RUN_TRAIN,
        RUN_TRAIN_DONE,
        RUN_EVAL,
        RUN_EVAL_DONE
    } run_state_e;
    run_state_e run_state;

    logic [1:0] stats_sel;
    logic train_start_pulse;
    logic eval_start_pulse;

    typedef enum logic [2:0] {
        S_IDLE,
        S_START_READ,
        S_READ_BYTES,
        S_DONE
    } stream_state_e;
    stream_state_e stream_state;

    // -----------------------------
    // FIFO for spike bytes (from SD)
    // -----------------------------
    localparam int FIFO_DEPTH = 2048;
    localparam int FIFO_W = $clog2(FIFO_DEPTH);
    logic [7:0] fifo_mem [0:FIFO_DEPTH-1];
    logic [FIFO_W-1:0] fifo_wptr;
    logic [FIFO_W-1:0] fifo_rptr;
    logic [FIFO_W:0] fifo_count;
    logic fifo_wr_en;
    logic [7:0] fifo_wr_data;
    logic fifo_rd_en;
    wire [7:0] fifo_rd_data = fifo_mem[fifo_rptr];

    // Blank-flag FIFO (tracks which inputs are blank steps)
    localparam int BFIFO_DEPTH = 2048;
    localparam int BFIFO_W = $clog2(BFIFO_DEPTH);
    logic bfifo_mem [0:BFIFO_DEPTH-1];
    logic [BFIFO_W-1:0] bfifo_wptr;
    logic [BFIFO_W-1:0] bfifo_rptr;
    logic [BFIFO_W:0] bfifo_count;
    logic bfifo_wr_en;
    logic bfifo_wr_data;
    logic bfifo_rd_en;
    wire bfifo_rd_data = bfifo_mem[bfifo_rptr];

    // -----------------------------
    // Pipeline small instance
    // -----------------------------
    logic                        ps_s_tvalid;
    logic                        ps_s_tready;
    logic [TSTEP_W+N_IN-1:0]      ps_s_tdata;
    logic                        ps_s_stdp_en;
    logic                        ps_m_tvalid;
    logic                        ps_m_tready;
    logic [TSTEP_W+N_NEURONS-1:0] ps_m_tdata;

    // Training/eval memories and control (streaming)
    logic assign_start;
    logic assign_done;
    logic assign_sample_commit;
    logic [LABEL_BITS-1:0] assign_sample_label;
    logic assign_count_we;
    logic [$clog2(N_NEURONS)-1:0] assign_count_neuron;
    logic [7:0] assign_count_value;
    logic [N_NEURONS*LABEL_BITS-1:0] assignments_bus;

    logic pred_start;
    logic pred_ready;
    logic pred_sample_begin;
    logic pred_sample_done;
    logic [LABEL_BITS-1:0] pred_sample_label;
    logic pred_count_we;
    logic [$clog2(N_NEURONS)-1:0] pred_count_neuron;
    logic [7:0] pred_count_value;
    logic pred_valid;
    logic [LABEL_BITS-1:0] pred_label;
    logic [LABEL_BITS-1:0] true_label;

    logic [15:0] exc_counts [0:N_NEURONS-1];
    logic [$clog2(N_NEURONS):0] count_idx;
    logic write_counts_active;
    logic [$clog2(N_SAMPLES)-1:0] sample_idx_out;
    logic [31:0] time_idx_out;
    logic samples_done;
    logic assign_running;
    logic pred_running;
    logic [LABEL_BITS-1:0] labels_mem_q;
    logic [$clog2(N_SAMPLES)-1:0] labels_rd_addr;
    logic [$clog2(N_SAMPLES)-1:0] labels_wr_addr;
    logic [LABEL_BITS-1:0] labels_wr_data;
    logic labels_wr_en;
    logic [31:0] tp_count;
    logic [31:0] tn_count;
    logic [31:0] fp_count;
    logic [31:0] fn_count;
    logic [31:0] display_value;
    logic [3:0] disp_digits [0:3];
    logic [1:0] disp_sel;
    logic [15:0] disp_div;
    typedef enum logic [2:0] {DISP_IDLE, DISP_DIV0, DISP_DIV1, DISP_DIV2, DISP_DIV3} disp_state_e;
    disp_state_e disp_state;
    logic [31:0] disp_value_reg;
    logic [31:0] disp_dividend;
    logic [31:0] disp_divisor;
    logic        disp_div_valid_in;
    logic [31:0] disp_quotient;
    logic [31:0] disp_remainder;
    logic        disp_div_valid_out;
    logic        disp_div_error;
    logic        disp_div_busy;
    logic train_done;
    logic eval_done;
    logic hold_input;

    pipeline_small #(
        .TSTEP_W(TSTEP_W),
        .N_IN(N_IN),
        .N_NEURONS(N_NEURONS),
        .UPDATE_NT(350),
        .W_INIT_FROM_FILE(1)
    ) u_pipeline_small (
        .clk(clk_25mhz),
        .rst(reset),
        .s_tvalid(ps_s_tvalid),
        .s_tready(ps_s_tready),
        .s_tdata(ps_s_tdata),
        .s_stdp_en(ps_s_stdp_en),
        .m_tvalid(ps_m_tvalid),
        .m_tready(ps_m_tready),
        .m_tdata(ps_m_tdata),
        .dbg_en(1'b0),
        .dbg_neuron('0),
        .dbg_in('0),
        .dbg_valid(),
        .dbg_data()
    );

    assign ps_m_tready = 1'b1;

    assign_labels_stream #(
        .N_NEURONS(N_NEURONS),
        .N_LABELS(N_LABELS),
        .LABEL_BITS(LABEL_BITS)
    ) u_assign_labels_stream (
        .clk(clk_25mhz),
        .rst(reset),
        .sample_commit(assign_sample_commit),
        .sample_label(assign_sample_label),
        .count_we(assign_count_we),
        .count_neuron(assign_count_neuron),
        .count_value(assign_count_value),
        .start(assign_start),
        .done(assign_done),
        .assignments(assignments_bus)
    );

    prediction_stream #(
        .N_NEURONS(N_NEURONS),
        .N_LABELS(N_LABELS),
        .LABEL_BITS(LABEL_BITS)
    ) u_prediction_stream (
        .clk(clk_25mhz),
        .rst(reset),
        .start(pred_start),
        .ready(pred_ready),
        .sample_begin(pred_sample_begin),
        .sample_label_in(pred_sample_label),
        .count_neuron(pred_count_neuron),
        .count_value(pred_count_value),
        .count_we(pred_count_we),
        .sample_done(pred_sample_done),
        .assignments(assignments_bus),
        .pred_valid(pred_valid),
        .pred_label(pred_label),
        .true_label(true_label)
    );

    // Labels RAM (XPM, read latency = 1)
    xpm_memory_sdpram #(
        .ADDR_WIDTH_A($clog2(N_SAMPLES)),
        .ADDR_WIDTH_B($clog2(N_SAMPLES)),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(LABEL_BITS),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(N_SAMPLES * LABEL_BITS),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(LABEL_BITS),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("0"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(0),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(LABEL_BITS),
        .WRITE_MODE_B("read_first")
    ) u_labels_mem (
        .clka(clk_25mhz),
        .ena(labels_wr_en),
        .wea(labels_wr_en),
        .addra(labels_wr_addr),
        .dina(labels_wr_data),
        .clkb(clk_25mhz),
        .enb(1'b1),
        .addrb(labels_rd_addr),
        .doutb(labels_mem_q),
        .rstb(reset),
        .regceb(1'b1),
        .sleep(1'b0),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0)
    );

    // read control + parsing
    always_ff @(posedge clk_25mhz) begin
        if (reset) begin
            rd              <= 1'b0;
            wr              <= 1'b0;
            address         <= LBA_START;
            sector_addr     <= LBA_START;
            byte_count      <= 9'd0;
            in_read         <= 1'b0;
            file_byte_index <= 32'd0;

            header_done     <= 1'b0;
            num_images_u32  <= 32'd0;
            n_time_u32      <= 32'd0;
            n_neurons_u32   <= 32'd0;
            label_index     <= 32'd0;

            streaming       <= 1'b0;

            stream_state    <= S_IDLE;
            fifo_wr_en      <= 1'b0;
            fifo_wr_data    <= 8'd0;
            labels_wr_en    <= 1'b0;
            labels_wr_addr  <= '0;
            labels_wr_data  <= '0;
            run_state <= RUN_IDLE;
            stats_sel <= 2'b0;
            train_start_pulse <= 1'b0;
            eval_start_pulse <= 1'b0;
        end else begin
            rd          <= 1'b0;
            fifo_wr_en  <= 1'b0;
            labels_wr_en <= 1'b0;

            train_start_pulse <= 1'b0;
            eval_start_pulse <= 1'b0;

            if (run_state == RUN_IDLE) begin
                if (btn3_rise) begin
                    run_state <= RUN_TRAIN;
                    train_start_pulse <= 1'b1;
                    stats_sel <= 2'b0;
                end
            end else if (run_state == RUN_TRAIN_DONE) begin
                if (btn1_rise) begin
                    run_state <= RUN_EVAL;
                    eval_start_pulse <= 1'b1;
                    stats_sel <= 2'b0;
                end
            end else if (run_state == RUN_EVAL_DONE) begin
                if (btn2_rise) begin
                    stats_sel <= stats_sel + 1'b1;
                end
            end

            if (train_done) begin
                run_state <= RUN_TRAIN_DONE;
            end
            if (eval_done) begin
                run_state <= RUN_EVAL_DONE;
            end

            case (stream_state)
                S_IDLE: begin
                    if (ready && (SD_CD_N == 1'b0)) begin
                        stream_state <= S_START_READ;
                    end
                end

                S_START_READ: begin
                    if (!in_read && ready && !hold_input && (fifo_count <= (FIFO_DEPTH-512))) begin
                        address   <= sector_addr;
                        rd        <= 1'b1;
                        in_read   <= 1'b1;
                        byte_count <= 9'd0;
                        stream_state <= S_READ_BYTES;
                    end
                end

                S_READ_BYTES: begin
                    if (byte_available) begin
                        // Capture header
                        if (!header_done) begin
                            if ((file_byte_index >= 32'd8) && (file_byte_index <= 32'd18)) begin
                                header_bytes[file_byte_index] <= dout;
                            end
                            if (file_byte_index == 32'd19) begin
                                // parse header (little-endian), use current dout for byte[19]
                                num_images_u32 <= {header_bytes[11], header_bytes[10], header_bytes[9],  header_bytes[8]};
                                n_time_u32     <= {header_bytes[15], header_bytes[14], header_bytes[13], header_bytes[12]};
                                n_neurons_u32  <= {dout, header_bytes[18], header_bytes[17], header_bytes[16]};
                                header_done    <= 1'b1;
                            end
                        end else if (file_byte_index < (HEADER_BYTES + num_images_u32)) begin
                            // Labels area
                            labels_wr_en <= 1'b1;
                            labels_wr_addr <= label_index[$clog2(N_SAMPLES)-1:0];
                            labels_wr_data <= dout[LABEL_BITS-1:0];
                            label_index <= label_index + 1'b1;
                        end else if (streaming) begin
                            // Spikes area (one byte per neuron per time step)
                            if (fifo_count < FIFO_DEPTH) begin
                                fifo_wr_en   <= 1'b1;
                                fifo_wr_data <= dout;
                            end
                        end

                        // advance file byte index
                        file_byte_index <= file_byte_index + 1'b1;

                        // enter streaming after header+labels
                        if (header_done && !streaming &&
                            (file_byte_index + 1 >= (HEADER_BYTES + num_images_u32))) begin
                            streaming  <= 1'b1;
                        end

                        if (byte_count == SECTOR_BYTES - 1) begin
                            in_read    <= 1'b0;
                            sector_addr <= sector_addr + 1'b1;
                            stream_state <= (streaming && (file_byte_index >= (HEADER_BYTES + num_images_u32)) &&
                                             (file_byte_index - (HEADER_BYTES + num_images_u32) >=
                                              (num_images_u32 * n_time_u32 * n_neurons_u32)))
                                            ? S_DONE : S_START_READ;
                        end else begin
                            byte_count <= byte_count + 1'b1;
                        end
                    end
                end

                S_DONE: begin
                    streaming <= 1'b0;
                end
            endcase
        end
    end

    // FIFO update (single writer for pointers/count)
    always_ff @(posedge clk_25mhz) begin
        if (reset) begin
            fifo_wptr  <= '0;
            fifo_rptr  <= '0;
            fifo_count <= '0;
            bfifo_wptr <= '0;
            bfifo_rptr <= '0;
            bfifo_count <= '0;
        end else begin
            if (fifo_wr_en) begin
                fifo_mem[fifo_wptr] <= fifo_wr_data;
            end
            if (bfifo_wr_en) begin
                bfifo_mem[bfifo_wptr] <= bfifo_wr_data;
            end

            case ({fifo_wr_en, fifo_rd_en})
                2'b10: begin
                    fifo_wptr  <= fifo_wptr + 1'b1;
                    fifo_count <= fifo_count + 1'b1;
                end
                2'b01: begin
                    fifo_rptr  <= fifo_rptr + 1'b1;
                    fifo_count <= fifo_count - 1'b1;
                end
                2'b11: begin
                    fifo_wptr <= fifo_wptr + 1'b1;
                    fifo_rptr <= fifo_rptr + 1'b1;
                end
                default: begin
                end
            endcase

            case ({bfifo_wr_en, bfifo_rd_en})
                2'b10: begin
                    bfifo_wptr  <= bfifo_wptr + 1'b1;
                    bfifo_count <= bfifo_count + 1'b1;
                end
                2'b01: begin
                    bfifo_rptr  <= bfifo_rptr + 1'b1;
                    bfifo_count <= bfifo_count - 1'b1;
                end
                2'b11: begin
                    bfifo_wptr <= bfifo_wptr + 1'b1;
                    bfifo_rptr <= bfifo_rptr + 1'b1;
                end
                default: begin
                end
            endcase
        end
    end

    // -----------------------------
    // Spike packer -> pipeline_small
    // -----------------------------
    typedef enum logic [1:0] {
        P_IDLE,
        P_PACK,
        P_WAIT_SEND,
        P_BLANK
    } pack_state_e;
    pack_state_e pack_state;

    logic [N_IN-1:0] spike_vec;
    logic [$clog2(N_IN):0] spike_idx;
    logic [31:0] sample_idx_p;
    logic [31:0] time_idx_p;
    logic [31:0] blank_left;
    logic [TSTEP_W-1:0] tstep_id;
    wire train_mode = (run_state == RUN_TRAIN);

    always_ff @(posedge clk_25mhz) begin
        if (reset) begin
            pack_state  <= P_IDLE;
            spike_vec   <= '0;
            spike_idx   <= '0;
            sample_idx_p<= 32'd0;
            time_idx_p  <= 32'd0;
            blank_left  <= 32'd0;
            tstep_id    <= '0;
            ps_s_tvalid <= 1'b0;
            ps_s_tdata  <= '0;
            ps_s_stdp_en<= 1'b0;
            fifo_rd_en  <= 1'b0;
            bfifo_wr_en <= 1'b0;
            bfifo_wr_data <= 1'b0;
        end else begin
            ps_s_tvalid <= 1'b0;
            fifo_rd_en  <= 1'b0;
            ps_s_stdp_en <= 1'b0;
            bfifo_wr_en <= 1'b0;

            case (pack_state)
                P_IDLE: begin
                    if (streaming && !hold_input) begin
                        pack_state <= P_PACK;
                        spike_vec  <= '0;
                        spike_idx  <= '0;
                    end
                end

                P_PACK: begin
                    fifo_rd_en <= 1'b0;
                    if (!hold_input && (fifo_count != 0)) begin
                        fifo_rd_en <= 1'b1;
                        spike_vec[spike_idx] <= (fifo_rd_data != 8'd0);

                        if (spike_idx == N_IN-1) begin
                            spike_idx <= '0;
                            pack_state <= P_WAIT_SEND;
                        end else begin
                            spike_idx <= spike_idx + 1'b1;
                        end
                    end
                end

                P_WAIT_SEND: begin
                    if (!hold_input && ps_s_tready) begin
                        ps_s_tdata <= {tstep_id, spike_vec};
                        ps_s_tvalid <= 1'b1;
                        ps_s_stdp_en <= (run_state == RUN_TRAIN);
                        bfifo_wr_en <= 1'b1;
                        bfifo_wr_data <= 1'b0;
                        tstep_id <= tstep_id + 1'b1;

                        if (time_idx_p + 1 >= n_time_u32) begin
                            time_idx_p <= 32'd0;
                            sample_idx_p <= sample_idx_p + 1'b1;
                            if (sample_idx_p + 1 >= num_images_u32) begin
                                pack_state <= P_IDLE;
                            end else if (BLANK_STEPS != 0) begin
                                blank_left <= BLANK_STEPS;
                                pack_state <= P_BLANK;
                            end else begin
                                pack_state <= P_PACK;
                            end
                        end else begin
                            time_idx_p <= time_idx_p + 1'b1;
                            pack_state <= P_PACK;
                        end
                    end
                end

                P_BLANK: begin
                    if (!hold_input && ps_s_tready) begin
                        ps_s_tdata <= {tstep_id, {N_IN{1'b0}}};
                        ps_s_tvalid <= 1'b1;
                        ps_s_stdp_en <= 1'b0;
                        bfifo_wr_en <= 1'b1;
                        bfifo_wr_data <= 1'b1;
                        tstep_id <= tstep_id + 1'b1;
                        if (blank_left <= 1) begin
                            blank_left <= 0;
                            pack_state <= P_PACK;
                        end else begin
                            blank_left <= blank_left - 1'b1;
                        end
                    end
                end
            endcase
        end
    end

    // -----------------------------
    // Output accumulation + training/eval control
    // -----------------------------
    always_ff @(posedge clk_25mhz) begin
        if (reset) begin
            labels_rd_addr <= '0;
        end else begin
            labels_rd_addr <= sample_idx_out[$clog2(N_SAMPLES)-1:0];
        end
    end

    integer nn;
    always_ff @(posedge clk_25mhz) begin
        if (reset) begin
            for (nn = 0; nn < N_NEURONS; nn = nn + 1) begin
                exc_counts[nn] <= '0;
            end
            time_idx_out <= 32'd0;
            sample_idx_out <= '0;
            samples_done <= 1'b0;
            write_counts_active <= 1'b0;
            count_idx <= '0;

            assign_start <= 1'b0;
            assign_running <= 1'b0;
            pred_start <= 1'b0;
            pred_running <= 1'b0;
            train_done <= 1'b0;
            eval_done <= 1'b0;
            hold_input <= 1'b1;

            assign_sample_commit <= 1'b0;
            assign_sample_label <= '0;
            assign_count_we <= 1'b0;
            assign_count_neuron <= '0;
            assign_count_value <= '0;
            pred_sample_begin <= 1'b0;
            pred_sample_done <= 1'b0;
            pred_sample_label <= '0;
            pred_count_we <= 1'b0;
            pred_count_neuron <= '0;
            pred_count_value <= '0;
            tp_count <= 32'd0;
            tn_count <= 32'd0;
            fp_count <= 32'd0;
            fn_count <= 32'd0;
        end else begin
            assign_start <= 1'b0;
            pred_start <= 1'b0;
            assign_sample_commit <= 1'b0;
            assign_count_we <= 1'b0;
            pred_sample_begin <= 1'b0;
            pred_sample_done <= 1'b0;
            pred_count_we <= 1'b0;
            bfifo_rd_en <= 1'b0;

            hold_input <= write_counts_active | assign_running | pred_running |
                          (run_state == RUN_IDLE) | (run_state == RUN_TRAIN_DONE) | (run_state == RUN_EVAL_DONE);

            if (train_start_pulse) begin
                for (nn = 0; nn < N_NEURONS; nn = nn + 1) begin
                    exc_counts[nn] <= '0;
                end
                time_idx_out <= 32'd0;
                sample_idx_out <= '0;
                samples_done <= 1'b0;
                write_counts_active <= 1'b0;
                count_idx <= '0;
                train_done <= 1'b0;
                eval_done <= 1'b0;
                tp_count <= 32'd0;
                tn_count <= 32'd0;
                fp_count <= 32'd0;
                fn_count <= 32'd0;
            end else if (eval_start_pulse) begin
                for (nn = 0; nn < N_NEURONS; nn = nn + 1) begin
                    exc_counts[nn] <= '0;
                end
                time_idx_out <= 32'd0;
                samples_done <= 1'b0;
                write_counts_active <= 1'b0;
                count_idx <= '0;
                eval_done <= 1'b0;
                tp_count <= 32'd0;
                tn_count <= 32'd0;
                fp_count <= 32'd0;
                fn_count <= 32'd0;
                pred_start <= 1'b1;
                pred_running <= 1'b1;
            end

            // Pop blank-flag FIFO on each output
            if (ps_m_tvalid && ps_m_tready && (bfifo_count != 0)) begin
                bfifo_rd_en <= 1'b1;
                if (!bfifo_rd_data && !write_counts_active && !assign_running && !pred_running) begin
                    for (nn = 0; nn < N_NEURONS; nn = nn + 1) begin
                        if (ps_m_tdata[nn]) begin
                            exc_counts[nn] <= exc_counts[nn] + 1'b1;
                        end
                    end

                    if (time_idx_out + 1 >= n_time_u32) begin
                        time_idx_out <= 32'd0;
                        sample_idx_out <= sample_idx_out + 1'b1;
                        if (run_state == RUN_TRAIN) begin
                            samples_done <= (sample_idx_out + 1 >= TRAIN_SAMPLES);
                        end else if (run_state == RUN_EVAL) begin
                            samples_done <= (sample_idx_out + 1 >= (TRAIN_SAMPLES + EVAL_SAMPLES));
                        end else begin
                            samples_done <= 1'b0;
                        end
                        write_counts_active <= 1'b1;
                        count_idx <= '0;
                        assign_sample_label <= labels_mem_q;
                        pred_sample_label <= labels_mem_q;
                    end else begin
                        time_idx_out <= time_idx_out + 1'b1;
                    end
                end
            end

            // Stream per-sample spike counts into assign/pred modules
            if (write_counts_active) begin
                if (count_idx < N_NEURONS) begin
                    if (run_state == RUN_TRAIN) begin
                        if (count_idx == 0) begin
                            assign_sample_commit <= 1'b1;
                        end
                        assign_count_we <= 1'b1;
                        assign_count_neuron <= count_idx[$clog2(N_NEURONS)-1:0];
                        assign_count_value <= (exc_counts[count_idx] > 16'd255) ? 8'hFF : exc_counts[count_idx][7:0];
                    end else begin
                        if (count_idx == 0) begin
                            pred_sample_begin <= 1'b1;
                        end
                        pred_count_we <= 1'b1;
                        pred_count_neuron <= count_idx[$clog2(N_NEURONS)-1:0];
                        pred_count_value <= (exc_counts[count_idx] > 16'd255) ? 8'hFF : exc_counts[count_idx][7:0];
                    end
                    count_idx <= count_idx + 1'b1;
                end else begin
                    for (nn = 0; nn < N_NEURONS; nn = nn + 1) begin
                        exc_counts[nn] <= '0;
                    end
                    write_counts_active <= 1'b0;
                    if (run_state == RUN_EVAL) begin
                        pred_sample_done <= 1'b1;
                    end
                end
            end

            // After all samples captured, run assign_labels (train)
            if (samples_done && !assign_running && !pred_running && !write_counts_active) begin
                if ((run_state == RUN_TRAIN) && !train_done) begin
                    assign_start <= 1'b1;
                    assign_running <= 1'b1;
                end
            end

            if (assign_running && assign_done) begin
                assign_running <= 1'b0;
                train_done <= 1'b1;
            end

            if (pred_running && pred_ready) begin
                pred_running <= 1'b0;
            end
            if (samples_done && pred_ready && !write_counts_active) begin
                eval_done <= 1'b1;
            end

            if (pred_valid) begin
                if ((true_label == POS_LABEL[LABEL_BITS-1:0]) && (pred_label == POS_LABEL[LABEL_BITS-1:0])) begin
                    tp_count <= tp_count + 1'b1;
                end else if ((true_label != POS_LABEL[LABEL_BITS-1:0]) && (pred_label != POS_LABEL[LABEL_BITS-1:0])) begin
                    tn_count <= tn_count + 1'b1;
                end else if ((true_label != POS_LABEL[LABEL_BITS-1:0]) && (pred_label == POS_LABEL[LABEL_BITS-1:0])) begin
                    fp_count <= fp_count + 1'b1;
                end else begin
                    fn_count <= fn_count + 1'b1;
                end
            end
        end
    end

    // -----------------------------
    // LED + 7seg status display
    // -----------------------------
    // Debug LED bank select
    // sw[1:0] = bank
    // sw[5:2] = header byte select (bank 3)
    logic [1:0] led_bank;
    logic [3:0] hdr_sel;
    logic [7:0] hdr_byte;
    assign led_bank = sw[1:0];
    assign hdr_sel = sw[5:2];

    always_comb begin
        if (hdr_sel <= 4'd11) begin
            hdr_byte = header_bytes[hdr_sel + 4'd8];
        end else begin
            hdr_byte = 8'h00;
        end
    end

    always_comb begin
        led = 16'b0;
        case (led_bank)
            2'd0: begin
                led[0]  = train_done;               // training finished
                led[1]  = eval_done;                // eval finished
                led[2]  = ready;                    // SD controller ready
                led[3]  = (SD_CD_N == 1'b0);         // SD card inserted
                led[4]  = header_done;              // header parsed
                led[5]  = streaming;                // streaming active
                led[6]  = (run_state != RUN_IDLE);  // run active
                led[7]  = ps_s_tvalid;              // input to pipeline valid
                led[8]  = ps_s_tready;              // pipeline ready
                led[9]  = ps_m_tvalid;              // pipeline output valid
                led[10] = (fifo_count != 0);        // spike FIFO not empty
                led[11] = in_read;                  // SD read in progress
                led[12] = byte_available;           // SD byte available
                led[13] = assign_running;           // assign_labels running
                led[14] = pred_running;             // prediction running
                led[15] = samples_done;             // samples done flag
            end
            2'd1: begin
                led[2:0]  = stream_state;           // stream FSM
                led[5:3]  = run_state;              // run FSM
                led[6]    = reset;                  // global reset
                led[7]    = hold_input;             // input hold
                led[8]    = in_read;                // SD read in progress
                led[9]    = byte_available;         // SD byte available
                led[10]   = fifo_wr_en;             // spike FIFO write
                led[11]   = fifo_rd_en;             // spike FIFO read
                led[12]   = bfifo_wr_en;            // blank FIFO write
                led[13]   = bfifo_rd_en;            // blank FIFO read
                led[14]   = ps_s_tvalid;            // input valid
                led[15]   = ps_m_tvalid;            // output valid
            end
            2'd2: begin
                led[7:0]  = fifo_count[7:0];        // spike FIFO fill (LSB)
                led[15:8] = bfifo_count[7:0];       // blank FIFO fill (LSB)
            end
            default: begin
                led[7:0]  = hdr_byte;               // header byte (sw[5:2])
                led[8]    = header_done;            // header parsed
                led[9]    = streaming;              // streaming active
                led[10]   = ready;                  // SD ready
                led[11]   = (SD_CD_N == 1'b0);       // SD present
                led[15:12]= status[4:1];            // SD status (upper 4)
            end
        endcase
        rgb0 = 3'b000;
        rgb1 = 3'b000;
    end

    always_comb begin
        if (run_state == RUN_TRAIN) begin
            display_value = sample_idx_out + 1;
        end else if (run_state == RUN_TRAIN_DONE) begin
            display_value = TRAIN_SAMPLES;
        end else if (run_state == RUN_EVAL) begin
            if (sample_idx_out >= TRAIN_SAMPLES)
                display_value = (sample_idx_out - TRAIN_SAMPLES) + 1;
            else
                display_value = 0;
        end else if (run_state == RUN_EVAL_DONE) begin
            case (stats_sel)
                2'd0: display_value = tp_count;
                2'd1: display_value = tn_count;
                2'd2: display_value = fp_count;
                default: display_value = fn_count;
            endcase
        end else begin
            display_value = 0;
        end
    end

    divider2b #(.WIDTH(32)) u_disp_divider(
        .clk_in        (clk_25mhz),
        .rst_in        (reset),
        .dividend_in   (disp_dividend),
        .divisor_in    (disp_divisor),
        .data_valid_in (disp_div_valid_in),
        .quotient_out  (disp_quotient),
        .remainder_out (disp_remainder),
        .data_valid_out(disp_div_valid_out),
        .error_out     (disp_div_error),
        .busy_out      (disp_div_busy)
    );

    always_ff @(posedge clk_25mhz) begin
        if (reset) begin
            disp_value_reg <= '0;
            disp_dividend <= '0;
            disp_divisor <= '0;
            disp_div_valid_in <= 1'b0;
            disp_state <= DISP_IDLE;
            disp_digits[0] <= '0;
            disp_digits[1] <= '0;
            disp_digits[2] <= '0;
            disp_digits[3] <= '0;
        end else begin
            disp_div_valid_in <= 1'b0;
            case (disp_state)
                DISP_IDLE: begin
                    if (display_value != disp_value_reg) begin
                        disp_value_reg <= display_value;
                        disp_dividend <= display_value;
                        disp_divisor <= 32'd10;
                        disp_div_valid_in <= 1'b1;
                        disp_state <= DISP_DIV0;
                    end
                end
                DISP_DIV0: begin
                    if (disp_div_valid_out) begin
                        disp_digits[0] <= disp_remainder[3:0];
                        disp_dividend <= disp_quotient;
                        disp_divisor <= 32'd10;
                        disp_div_valid_in <= 1'b1;
                        disp_state <= DISP_DIV1;
                    end
                end
                DISP_DIV1: begin
                    if (disp_div_valid_out) begin
                        disp_digits[1] <= disp_remainder[3:0];
                        disp_dividend <= disp_quotient;
                        disp_divisor <= 32'd10;
                        disp_div_valid_in <= 1'b1;
                        disp_state <= DISP_DIV2;
                    end
                end
                DISP_DIV2: begin
                    if (disp_div_valid_out) begin
                        disp_digits[2] <= disp_remainder[3:0];
                        disp_dividend <= disp_quotient;
                        disp_divisor <= 32'd10;
                        disp_div_valid_in <= 1'b1;
                        disp_state <= DISP_DIV3;
                    end
                end
                DISP_DIV3: begin
                    if (disp_div_valid_out) begin
                        disp_digits[3] <= disp_remainder[3:0];
                        disp_state <= DISP_IDLE;
                    end
                end
                default: disp_state <= DISP_IDLE;
            endcase
        end
    end

    always_ff @(posedge clk_25mhz) begin
        if (reset) begin
            disp_div <= 16'd0;
            disp_sel <= 2'd0;
        end else begin
            disp_div <= disp_div + 1'b1;
            disp_sel <= disp_div[15:14];
        end
    end

    logic [6:0] seg_raw;
    bto7s u_bto7s(.x(disp_digits[disp_sel]), .s(seg_raw));

    always_comb begin
        ss0_an = 4'b1111;
        ss0_an[disp_sel] = 1'b0;
        ss0_c = ~seg_raw;
        ss1_an = 4'hF;
        ss1_c = 7'h7F;
    end

endmodule

`default_nettype wire
