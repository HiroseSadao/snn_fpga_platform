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

    logic [31:0] sum_spikes [0:N_NEURONS-1][0:N_LABELS-1];
    logic [31:0] n_labeled  [0:N_LABELS-1];
    logic [LABEL_BITS-1:0] current_label;

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
    always_ff @(posedge clk) begin
        if (rst) begin
            current_label <= '0;
            for (i = 0; i < N_NEURONS; i = i + 1) begin
                for (l = 0; l < N_LABELS; l = l + 1) begin
                    sum_spikes[i][l] <= '0;
                end
            end
            for (l = 0; l < N_LABELS; l = l + 1) begin
                n_labeled[l] <= '0;
            end
        end else begin
            if (sample_commit) begin
                current_label <= sample_label;
                if (sample_label < N_LABELS[LABEL_BITS-1:0]) begin
                    n_labeled[sample_label] <= n_labeled[sample_label] + 1'b1;
                end
            end
            if (count_we) begin
                if (current_label < N_LABELS[LABEL_BITS-1:0]) begin
                    sum_spikes[count_neuron][current_label] <=
                        sum_spikes[count_neuron][current_label] + count_value;
                end
            end
        end
    end

    typedef enum logic [2:0] {
        S_IDLE,
        S_LABEL_INIT,
        S_DIV_START,
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

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            done <= 1'b0;
            neuron_idx <= '0;
            label_idx <= '0;
            best_label <= '0;
            best_rate <= '0;
            div_dividend <= '0;
            div_divisor <= '0;
            div_valid_in <= 1'b0;
            for (i = 0; i < N_NEURONS; i = i + 1) begin
                assignments[i*LABEL_BITS +: LABEL_BITS] <= '0;
            end
        end else begin
            done <= 1'b0;
            div_valid_in <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        neuron_idx <= '0;
                        label_idx <= '0;
                        best_label <= '0;
                        best_rate <= '0;
                        state <= S_LABEL_INIT;
                    end
                end

                S_LABEL_INIT: begin
                    if (n_labeled[label_idx] != 0) begin
                        div_dividend <= (sum_spikes[neuron_idx][label_idx] << FP_SHIFT) +
                                        (n_labeled[label_idx] >> 1);
                        div_divisor  <= n_labeled[label_idx];
                        div_valid_in <= 1'b1;
                        state <= S_DIV_WAIT;
                    end else begin
                        rate_val <= '0;
                        state <= S_LABEL_NEXT;
                    end
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

endmodule

`default_nettype wire
