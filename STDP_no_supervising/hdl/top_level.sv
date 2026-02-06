`default_nettype none

module top_level(
        input  wire        clk_100mhz,
        input  wire [15:0] sw,  // unused
        input  wire [3:0]  btn, // btn[0] used as reset
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
    wire reset = btn[0];

    // 25 MHz clock from 100 MHz input (divide by 4)
    logic [1:0] clk_div;
    always_ff @(posedge clk_100mhz) begin
        if (reset) begin
            clk_div <= 2'b0;
        end else begin
            clk_div <= clk_div + 2'b1;
        end
    end
    wire clk_25mhz = clk_div[1];

    localparam int N_IN = 784;
    localparam int N_NEURONS = 100;
    localparam int TSTEP_W = 16;
    localparam int BLANK_STEPS = 150;

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
    logic [31:0] version_u32;
    logic [31:0] num_images_u32;
    logic [31:0] n_time_u32;
    logic [31:0] n_neurons_u32;
    logic        header_done;
    logic        header_ok;

    // Labels
    logic [31:0] label_index;
    logic [7:0]  last_label;

    logic        streaming;

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
    logic fifo_overflow;
    logic fifo_wr_en;
    logic [7:0] fifo_wr_data;
    logic fifo_rd_en;
    wire [7:0] fifo_rd_data = fifo_mem[fifo_rptr];

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
            header_ok       <= 1'b0;
            version_u32     <= 32'd0;
            num_images_u32  <= 32'd0;
            n_time_u32      <= 32'd0;
            n_neurons_u32   <= 32'd0;
            label_index     <= 32'd0;
            last_label      <= 8'd0;

            streaming       <= 1'b0;

            stream_state    <= S_IDLE;
            fifo_overflow   <= 1'b0;
            fifo_wr_en      <= 1'b0;
            fifo_wr_data    <= 8'd0;
        end else begin
            rd          <= 1'b0;
            fifo_wr_en  <= 1'b0;

            case (stream_state)
                S_IDLE: begin
                    if (ready && (SD_CD_N == 1'b0)) begin
                        stream_state <= S_START_READ;
                    end
                end

                S_START_READ: begin
                    if (!in_read && ready && (fifo_count <= (FIFO_DEPTH-512))) begin
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
                            header_bytes[file_byte_index] <= dout;
                            if (file_byte_index == 32'd19) begin
                                // parse header (little-endian), use current dout for byte[19]
                                version_u32    <= {header_bytes[7],  header_bytes[6],  header_bytes[5],  header_bytes[4]};
                                num_images_u32 <= {header_bytes[11], header_bytes[10], header_bytes[9],  header_bytes[8]};
                                n_time_u32     <= {header_bytes[15], header_bytes[14], header_bytes[13], header_bytes[12]};
                                n_neurons_u32  <= {dout, header_bytes[18], header_bytes[17], header_bytes[16]};
                                header_ok      <= (header_bytes[0] == 8'h53) && (header_bytes[1] == 8'h50) &&
                                                  (header_bytes[2] == 8'h4B) && (header_bytes[3] == 8'h31);
                                header_done    <= 1'b1;
                            end
                        end else if (file_byte_index < (HEADER_BYTES + num_images_u32)) begin
                            // Labels area
                            last_label  <= dout;
                            label_index <= label_index + 1'b1;
                        end else if (streaming) begin
                            // Spikes area (one byte per neuron per time step)
                            if (fifo_count < FIFO_DEPTH) begin
                                fifo_wr_en   <= 1'b1;
                                fifo_wr_data <= dout;
                            end else begin
                                fifo_overflow <= 1'b1;
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
        end else begin
            if (fifo_wr_en) begin
                fifo_mem[fifo_wptr] <= fifo_wr_data;
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
    wire train_mode = sw[0];

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
        end else begin
            ps_s_tvalid <= 1'b0;
            fifo_rd_en  <= 1'b0;
            ps_s_stdp_en <= 1'b0;

            case (pack_state)
                P_IDLE: begin
                    if (streaming) begin
                        pack_state <= P_PACK;
                        spike_vec  <= '0;
                        spike_idx  <= '0;
                    end
                end

                P_PACK: begin
                    fifo_rd_en <= 1'b0;
                    if (fifo_count != 0) begin
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
                    if (ps_s_tready) begin
                        ps_s_tdata <= {tstep_id, spike_vec};
                        ps_s_tvalid <= 1'b1;
                        ps_s_stdp_en <= train_mode;
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
                    if (ps_s_tready) begin
                        ps_s_tdata <= {tstep_id, {N_IN{1'b0}}};
                        ps_s_tvalid <= 1'b1;
                        ps_s_stdp_en <= 1'b0;
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
    // Debug outputs
    // -----------------------------
    // led[0]  = fifo_overflow
    // led[1]  = train_mode
    // led[2]  = header_ok
    // led[3]  = streaming
    // led[7:4]= stream_state
    // led[15:8]= byte_count (within sector)
    always_comb begin
        led = 16'b0;
        led[0]    = fifo_overflow;
        led[1]    = train_mode;
        led[2]    = header_ok;
        led[3]    = streaming;
        led[7:4]  = stream_state;
        led[15:8] = byte_count;
    end

    // Debug color: show label low bits once labels start
    assign rgb0  = last_label[2:0];
    assign rgb1  = {2'b0, header_ok};
    assign ss0_an = 4'hF;
    assign ss1_an = 4'hF;
    assign ss0_c  = 7'h7F;
    assign ss1_c  = 7'h7F;

endmodule

`default_nettype wire
