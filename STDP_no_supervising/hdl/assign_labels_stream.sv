`default_nettype none

module assign_labels_stream #(
        parameter int N_NEURONS = 100,
        parameter int N_LABELS  = 10,
        parameter int LABEL_BITS = 4
    )(
        input  wire clk,
        input  wire rst,

        input  wire sample_commit, // pulse at start of per-sample count writes
        input  wire [LABEL_BITS-1:0] sample_label,
        input  wire count_we,
        input  wire [$clog2(N_NEURONS)-1:0] count_neuron,
        input  wire [7:0] count_value,

        input  wire start,
        output logic done,

        output logic [N_NEURONS*LABEL_BITS-1:0] assignments
    );

    localparam int FP_SHIFT = 16;

    localparam int SUM_DEPTH = N_NEURONS * N_LABELS;
    localparam int SUM_ADDR_W = (SUM_DEPTH <= 1) ? 1 : $clog2(SUM_DEPTH);

    logic [31:0] n_labeled  [0:N_LABELS-1];
    logic [LABEL_BITS-1:0] current_label;
    logic clear_n_labeled;

    // sum_spikes BRAM (packed as [neuron][label])
    logic                 sum_wr_en;
    logic [SUM_ADDR_W-1:0] sum_wr_addr;
    logic [31:0]          sum_wr_data;
    logic                 sum_rd_en;
    logic [SUM_ADDR_W-1:0] sum_rd_addr;
    logic [31:0]          sum_rd_data;

    // Accumulation pipeline
    logic                 acc_rd_en;
    logic [SUM_ADDR_W-1:0] acc_rd_addr;
    logic [SUM_ADDR_W-1:0] acc_addr_q;
    logic [7:0]           acc_value_q;
    logic                 acc_pending;

    // Clear state
    logic [SUM_ADDR_W-1:0] clear_idx;

    // Divider
    logic [31:0] div_dividend;
    logic [31:0] div_divisor;
    logic        div_valid_in;
    logic [31:0] div_quotient;
    logic [31:0] div_remainder;
    logic        div_valid_out;
    logic        div_error;
    logic        div_busy;

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

    integer i, l;

    function automatic [SUM_ADDR_W-1:0] sum_addr(
        input [$clog2(N_NEURONS)-1:0] neuron,
        input [$clog2(N_LABELS)-1:0] label
    );
        int unsigned addr;
        begin
            addr = (neuron * N_LABELS) + label;
            sum_addr = addr[SUM_ADDR_W-1:0];
        end
    endfunction

`ifdef SYNTHESIS
    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(SUM_ADDR_W),
        .ADDR_WIDTH_B(SUM_ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(32),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE("none"),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(SUM_DEPTH * 32),
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
    ) u_sum_spikes (
        .clka(clk),
        .ena(sum_wr_en),
        .wea(sum_wr_en),
        .addra(sum_wr_addr),
        .dina(sum_wr_data),
        .clkb(clk),
        .enb(sum_rd_en),
        .addrb(sum_rd_addr),
        .doutb(sum_rd_data),
        .rstb(rst),
        .regceb(1'b1),
        .sleep(1'b0),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0)
    );
`else
    logic [31:0] sum_mem_sim [0:SUM_DEPTH-1];
    always_ff @(posedge clk) begin
        if (rst) begin
            sum_rd_data <= '0;
        end else begin
            if (sum_wr_en) begin
                sum_mem_sim[sum_wr_addr] <= sum_wr_data;
            end
            if (sum_rd_en) begin
                sum_rd_data <= sum_mem_sim[sum_rd_addr];
            end
        end
    end
`endif
    always_ff @(posedge clk) begin
        if (rst) begin
            current_label <= '0;
            for (l = 0; l < N_LABELS; l = l + 1) begin
                n_labeled[l] <= '0;
            end
            acc_rd_en <= 1'b0;
            acc_rd_addr <= '0;
            acc_addr_q <= '0;
            acc_value_q <= '0;
            acc_pending <= 1'b0;
        end else begin
            acc_rd_en <= 1'b0;
            if (clear_n_labeled) begin
                for (l = 0; l < N_LABELS; l = l + 1) begin
                    n_labeled[l] <= '0;
                end
            end

            if (sample_commit) begin
                current_label <= sample_label;
                if (sample_label < N_LABELS[LABEL_BITS-1:0]) begin
                    n_labeled[sample_label] <= n_labeled[sample_label] + 1'b1;
                end
            end

            if (state == S_IDLE) begin
                if (count_we && (current_label < N_LABELS[LABEL_BITS-1:0])) begin
                    acc_rd_en <= 1'b1;
                    acc_rd_addr <= sum_addr(count_neuron, current_label);
                    acc_addr_q <= sum_addr(count_neuron, current_label);
                    acc_value_q <= count_value;
                end
            end

            acc_pending <= acc_rd_en;
        end
    end

    typedef enum logic [2:0] {
        S_CLEAR,
        S_IDLE,
        S_LABEL_INIT,
        S_SUM_WAIT,
        S_DIV_WAIT,
        S_LABEL_NEXT,
        S_NEURON_NEXT,
        S_DONE
    } state_e;

    state_e state;
    logic [$clog2(N_NEURONS)-1:0] neuron_idx;
    logic [$clog2(N_LABELS)-1:0] label_idx;
    logic [LABEL_BITS-1:0] best_label;
    logic [31:0] best_rate;
    logic [31:0] rate_val;
    logic start_pending;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_CLEAR;
            done <= 1'b0;
            neuron_idx <= '0;
            label_idx <= '0;
            best_label <= '0;
            best_rate <= '0;
            div_dividend <= '0;
            div_divisor <= '0;
            div_valid_in <= 1'b0;
            clear_idx <= '0;
            start_pending <= 1'b0;
            clear_n_labeled <= 1'b0;
            for (i = 0; i < N_NEURONS; i = i + 1) begin
                assignments[i*LABEL_BITS +: LABEL_BITS] <= '0;
            end
        end else begin
            done <= 1'b0;
            div_valid_in <= 1'b0;
            clear_n_labeled <= 1'b0;

            case (state)
                S_CLEAR: begin
                    // Clear sum_spikes BRAM after reset
                    if (clear_idx == SUM_DEPTH-1) begin
                        clear_idx <= '0;
                        if (start_pending) begin
                            start_pending <= 1'b0;
                            neuron_idx <= '0;
                            label_idx <= '0;
                            best_label <= '0;
                            best_rate <= '0;
                            state <= S_LABEL_INIT;
                        end else begin
                            state <= S_IDLE;
                        end
                    end else begin
                        clear_idx <= clear_idx + 1'b1;
                    end
                end

                S_IDLE: begin
                    if (start) begin
                        start_pending <= 1'b1;
                        clear_idx <= '0;
                        clear_n_labeled <= 1'b1;
                        state <= S_CLEAR;
                    end
                end

                S_LABEL_INIT: begin
                    if (n_labeled[label_idx] != 0) begin
                        state <= S_SUM_WAIT;
                    end else begin
                        rate_val <= '0;
                        state <= S_LABEL_NEXT;
                    end
                end

                S_SUM_WAIT: begin
                    div_dividend <= (sum_rd_data << FP_SHIFT) +
                                    (n_labeled[label_idx] >> 1);
                    div_divisor  <= n_labeled[label_idx];
                    div_valid_in <= 1'b1;
                    state <= S_DIV_WAIT;
                end

                S_DIV_WAIT: begin
                    if (div_valid_out) begin
                        rate_val <= div_quotient;
                        state <= S_LABEL_NEXT;
                    end
                end

                S_LABEL_NEXT: begin
                    if (rate_val > best_rate) begin
                        best_rate <= rate_val;
                        best_label <= label_idx;
                    end
                    if (label_idx == N_LABELS-1) begin
                        assignments[neuron_idx*LABEL_BITS +: LABEL_BITS] <= best_label;
                        state <= S_NEURON_NEXT;
                    end else begin
                        label_idx <= label_idx + 1'b1;
                        state <= S_LABEL_INIT;
                    end
                end

                S_NEURON_NEXT: begin
                    if (neuron_idx == N_NEURONS-1) begin
                        state <= S_DONE;
                    end else begin
                        neuron_idx <= neuron_idx + 1'b1;
                        label_idx <= '0;
                        best_label <= '0;
                        best_rate <= '0;
                        state <= S_LABEL_INIT;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // Sum read/write control
    always_comb begin
        // Defaults
        sum_wr_en = 1'b0;
        sum_wr_addr = '0;
        sum_wr_data = '0;
        sum_rd_en = 1'b0;
        sum_rd_addr = '0;

        if (rst) begin
            // hold off memory access during reset
        end else if (state == S_CLEAR) begin
            // Clear takes priority
            sum_wr_en = 1'b1;
            sum_wr_addr = clear_idx;
            sum_wr_data = 32'd0;
        end else begin
            // Accumulation write (one cycle after read)
            if (acc_pending) begin
                sum_wr_en = 1'b1;
                sum_wr_addr = acc_addr_q;
                sum_wr_data = sum_rd_data + acc_value_q;
            end

            // Read for accumulation or assignment
            if (state == S_LABEL_INIT && (n_labeled[label_idx] != 0)) begin
                sum_rd_en = 1'b1;
                sum_rd_addr = sum_addr(neuron_idx, label_idx);
            end else if (state == S_IDLE && acc_rd_en) begin
                sum_rd_en = 1'b1;
                sum_rd_addr = acc_rd_addr;
            end
        end
    end

endmodule

`default_nettype wire
