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
    localparam logic [31:0] DDR_ADDR_WORD_LIMIT = 32'd16777216; // 64MiB / 4
    localparam logic [7:0] MAX_SUPPORTED_NARGS = 8'd2;
    localparam int RX_TIMEOUT_CLKS = CLKS_PER_BIT * 20; // timeout while waiting for remaining bytes
    localparam int N_IN = 784;
    localparam int N_NEURONS = 100;
    localparam logic signed [31:0] FXP_ALPHA = 32'sd62259; // 0.95 in S16.16
    localparam logic signed [31:0] FXP_INPUT_W = 32'sd8192; // 0.125 in S16.16
    localparam logic signed [31:0] FXP_THRESH = 32'sd65536; // 1.0 in S16.16
    localparam logic signed [31:0] FXP_BIAS_LSB = 32'sd512; // 0.0078125 in S16.16

    localparam logic [7:0] STATUS_OK             = 8'h00;
    localparam logic [7:0] STATUS_BAD_PACKET     = 8'hE1;
    localparam logic [7:0] STATUS_UNSUPPORTED_OP = 8'hE2;

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
    logic [7:0]  raw_image0_bits [0:97];
    logic        raw_image0_valid;
    logic [31:0] raw_num_images;
    logic [31:0] raw_bytes_per_image;
    logic [6:0]  raw_image0_capture_idx;

    logic        infer_active;
    logic [31:0] infer_steps_target;
    logic [15:0] infer_step_idx;
    logic [6:0]  infer_neuron_idx;
    logic [9:0]  infer_input_idx;
    logic signed [31:0] infer_accum;
    logic signed [31:0] infer_v_state [0:N_NEURONS-1];
    logic [15:0] infer_spike_count [0:N_NEURONS-1];
    logic [31:0] infer_total_spikes;
    integer rr;

    wire [7:0] r_in = {sw[15:11], 3'b000};
    wire [7:0] g_in = {sw[10:5],  2'b00};
    wire [7:0] b_in = {sw[4:0],   3'b000};

    assign SD_DQ1 = 1'b1;
    assign SD_DQ2 = 1'b1;

    assign rgb0[2] = tx_active;  // blue LED: UART TX active
    assign rgb0[1] = ddr_write_count[0]; // green LED: DDR write activity bit
    assign rgb0[0] = (resp_status == STATUS_OK); // red LED: OK result

    assign rgb1 = 3'b000;
    assign pmoda = {rgb0[0], rgb0[1], rgb0[2]};

    always_ff @(posedge clk_100mhz) begin
        if (btn[0]) begin
            clk_div <= 2'b00;
        end else begin
            clk_div <= clk_div + 2'b01;
        end
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

    function automatic logic get_image0_bit(input logic [9:0] bit_idx);
        logic [6:0] byte_idx;
        logic [2:0] bit_in_byte;
        begin
            byte_idx = bit_idx[9:3];
            bit_in_byte = bit_idx[2:0];
            if (byte_idx < 7'd98) begin
                get_image0_bit = raw_image0_bits[byte_idx][bit_in_byte];
            end else begin
                get_image0_bit = 1'b0;
            end
        end
    endfunction

    function automatic signed [31:0] neuron_bias(input logic [6:0] neuron_idx);
        begin
            neuron_bias = ($signed({28'd0, neuron_idx[2:0]}) + 32'sd1) * FXP_BIAS_LSB;
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
            raw_image0_capture_idx <= 7'd0;
            infer_active        <= 1'b0;
            infer_steps_target  <= 32'd0;
            infer_step_idx      <= 16'd0;
            infer_neuron_idx    <= 7'd0;
            infer_input_idx     <= 10'd0;
            infer_accum         <= 32'sd0;
            infer_total_spikes  <= 32'd0;
            for (rr = 0; rr < 98; rr = rr + 1) begin
                raw_image0_bits[rr] <= 8'h00;
            end
            for (rr = 0; rr < N_NEURONS; rr = rr + 1) begin
                infer_v_state[rr] <= 32'sd0;
                infer_spike_count[rr] <= 16'd0;
            end
        end else begin
            tx_dv <= 1'b0;
            sd_rd <= 1'b0;
            sd_wr <= 1'b0;
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

            if (rx_dv && !response_ready && !sd_copy_active && !infer_active) begin
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
                             (req_opcode == OP_READ_SPIKE_COUNT))
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
                                        raw_image0_capture_idx <= 7'd0;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_RUN_SAMPLE_INFER: begin
                                    if (
                                        (req_nargs == 8'd2) &&
                                        (arg0 == 32'sd0) &&
                                        (arg1 > 0) &&
                                        raw_image0_valid &&
                                        !infer_active
                                    ) begin
                                        infer_active       <= 1'b1;
                                        infer_steps_target <= arg1;
                                        infer_step_idx     <= 16'd0;
                                        infer_neuron_idx   <= 7'd0;
                                        infer_input_idx    <= 10'd0;
                                        infer_accum        <= neuron_bias(7'd0);
                                        infer_total_spikes <= 32'd0;
                                        for (rr = 0; rr < N_NEURONS; rr = rr + 1) begin
                                            infer_v_state[rr] <= 32'sd0;
                                            infer_spike_count[rr] <= 16'd0;
                                        end
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
                                end
                                OP_READ_SPIKE_COUNT: begin
                                    if ((req_nargs == 8'd2) && (arg0 >= 0) && (arg0 < N_NEURONS)) begin
                                        resp_status    <= STATUS_OK;
                                        resp_result    <= {16'd0, infer_spike_count[arg0[6:0]]};
                                        resp_checksum  <= calc_resp_checksum(STATUS_OK, {16'd0, infer_spike_count[arg0[6:0]]});
                                        response_ready <= 1'b1;
                                    end else begin
                                        resp_status    <= STATUS_BAD_PACKET;
                                        resp_result    <= 32'sd0;
                                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                        response_ready <= 1'b1;
                                    end
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
                        resp_result    <= 32'sd0;
                        resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
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
                            resp_result    <= 32'sd0;
                            resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
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
                                resp_result    <= 32'sd0;
                                resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
                                response_ready <= 1'b1;
                            end
                        end

                        if (
                            sd_header_done &&
                            (sd_file_bytes_seen >= (32'd20 + raw_num_images)) &&
                            (raw_image0_capture_idx < 7'd98)
                        ) begin
                            raw_image0_bits[raw_image0_capture_idx] <= sd_dout;
                            if (raw_image0_capture_idx == 7'd97) begin
                                raw_image0_valid <= 1'b1;
                            end
                            raw_image0_capture_idx <= raw_image0_capture_idx + 7'd1;
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
                            resp_result    <= 32'sd0;
                            resp_checksum  <= calc_resp_checksum(STATUS_BAD_PACKET, 32'sd0);
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
                if (infer_input_idx < N_IN) begin
                    if (
                        get_image0_bit(infer_input_idx) &&
                        (((infer_input_idx + infer_neuron_idx) & 10'd3) == 10'd0)
                    ) begin
                        infer_accum <= infer_accum + FXP_INPUT_W;
                    end
                    infer_input_idx <= infer_input_idx + 10'd1;
                end else begin
                    logic signed [31:0] v_next;
                    logic spike_now;
                    v_next = $signed(($signed(infer_v_state[infer_neuron_idx]) * $signed(FXP_ALPHA)) >>> 16)
                           + $signed(infer_accum);
                    spike_now = (v_next >= FXP_THRESH);

                    if (spike_now) begin
                        infer_v_state[infer_neuron_idx] <= v_next - FXP_THRESH;
                        infer_spike_count[infer_neuron_idx] <= infer_spike_count[infer_neuron_idx] + 16'd1;
                        infer_total_spikes <= infer_total_spikes + 32'd1;
                    end else begin
                        infer_v_state[infer_neuron_idx] <= v_next;
                    end

                    infer_input_idx <= 10'd0;
                    if (infer_neuron_idx == (N_NEURONS - 1)) begin
                        infer_neuron_idx <= 7'd0;
                        if ((infer_step_idx + 16'd1) >= infer_steps_target[15:0]) begin
                            infer_active <= 1'b0;
                            resp_status <= STATUS_OK;
                            resp_result <= infer_total_spikes + (spike_now ? 32'd1 : 32'd0);
                            resp_checksum <= calc_resp_checksum(
                                STATUS_OK,
                                infer_total_spikes + (spike_now ? 32'd1 : 32'd0)
                            );
                            response_ready <= 1'b1;
                        end
                        infer_step_idx <= infer_step_idx + 16'd1;
                        infer_accum <= neuron_bias(7'd0);
                    end else begin
                        infer_neuron_idx <= infer_neuron_idx + 7'd1;
                        infer_accum <= neuron_bias(infer_neuron_idx + 7'd1);
                    end
                end
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
