`default_nettype none

module prediction_stream #(
        parameter int N_NEURONS = 50,
        parameter int N_LABELS  = 10,
        parameter int LABEL_BITS = 4
    )(
        input  wire clk,
        input  wire rst,

        input  wire start, // compute n_assigns from assignments
        output logic ready,

        input  wire sample_begin,
        input  wire [LABEL_BITS-1:0] sample_label_in,
        input  wire [$clog2(N_NEURONS)-1:0] count_neuron,
        input  wire [7:0] count_value,
        input  wire count_we,
        input  wire sample_done,

        input  wire [N_NEURONS*LABEL_BITS-1:0] assignments,

        output logic pred_valid,
        output logic [LABEL_BITS-1:0] pred_label,
        output logic [LABEL_BITS-1:0] true_label
    );

    localparam int FP_SHIFT = 16;

    logic [15:0] n_assigns [0:N_LABELS-1];
    logic [31:0] sum_spikes_label [0:N_LABELS-1];
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

    typedef enum logic [2:0] {
        S_IDLE,
        S_ASSIGN_COUNT,
        S_SAMPLE_WAIT,
        S_LABEL_INIT,
        S_DIV_START,
        S_DIV_WAIT,
        S_LABEL_NEXT
    } state_e;

    state_e state;
    logic [$clog2(N_NEURONS)-1:0] assign_idx;
    logic [$clog2(N_LABELS)-1:0] label_idx;
    logic [LABEL_BITS-1:0] best_label;
    logic [31:0] best_rate;
    logic [31:0] rate_val;

    integer l;
    always_ff @(posedge clk) begin
        if (rst) begin
            ready <= 1'b0;
            pred_valid <= 1'b0;
            pred_label <= '0;
            true_label <= '0;
            state <= S_IDLE;
            assign_idx <= '0;
            label_idx <= '0;
            best_label <= '0;
            best_rate <= '0;
            div_dividend <= '0;
            div_divisor <= '0;
            div_valid_in <= 1'b0;
            current_label <= '0;
            for (l = 0; l < N_LABELS; l = l + 1) begin
                n_assigns[l] <= '0;
                sum_spikes_label[l] <= '0;
            end
        end else begin
            pred_valid <= 1'b0;
            div_valid_in <= 1'b0;

            case (state)
                S_IDLE: begin
                    ready <= 1'b0;
                    if (start) begin
                        for (l = 0; l < N_LABELS; l = l + 1) begin
                            n_assigns[l] <= '0;
                        end
                        assign_idx <= '0;
                        state <= S_ASSIGN_COUNT;
                    end
                end

                S_ASSIGN_COUNT: begin
                    n_assigns[assignments[assign_idx*LABEL_BITS +: LABEL_BITS]] <=
                        n_assigns[assignments[assign_idx*LABEL_BITS +: LABEL_BITS]] + 1'b1;
                    if (assign_idx == N_NEURONS-1) begin
                        ready <= 1'b1;
                        state <= S_SAMPLE_WAIT;
                    end else begin
                        assign_idx <= assign_idx + 1'b1;
                    end
                end

                S_SAMPLE_WAIT: begin
                    if (sample_begin) begin
                        current_label <= sample_label_in;
                        for (l = 0; l < N_LABELS; l = l + 1) begin
                            sum_spikes_label[l] <= '0;
                        end
                    end
                    if (count_we) begin
                        sum_spikes_label[assignments[count_neuron*LABEL_BITS +: LABEL_BITS]] <=
                            sum_spikes_label[assignments[count_neuron*LABEL_BITS +: LABEL_BITS]] + count_value;
                    end
                    if (sample_done) begin
                        label_idx <= '0;
                        best_label <= '0;
                        best_rate <= '0;
                        state <= S_LABEL_INIT;
                    end
                end

                S_LABEL_INIT: begin
                    if (n_assigns[label_idx] != 0) begin
                        div_dividend <= (sum_spikes_label[label_idx] << FP_SHIFT) +
                                        (n_assigns[label_idx] >> 1);
                        div_divisor  <= n_assigns[label_idx];
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
                        pred_label <= best_label;
                        true_label <= current_label;
                        pred_valid <= 1'b1;
                        state <= S_SAMPLE_WAIT;
                    end else begin
                        label_idx <= label_idx + 1'b1;
                        state <= S_LABEL_INIT;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
