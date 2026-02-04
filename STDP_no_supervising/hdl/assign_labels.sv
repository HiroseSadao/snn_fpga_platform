`default_nettype none

module assign_labels #(
        parameter int N_SAMPLES = 10000,
        parameter int N_NEURONS = 100,
        parameter int N_LABELS  = 10,
        parameter int LABEL_BITS = 4
    )(
        input  wire clk,
        input  wire rst,   // synchronous reset
        input  wire start,
        output logic done,

        // Write interface for spikes and labels
        input  wire                    spikes_we,
        input  wire [$clog2(N_SAMPLES)-1:0] spikes_sample_addr,
        input  wire [$clog2(N_NEURONS)-1:0] spikes_neuron_addr,
        input  wire [7:0]              spikes_wdata,

        input  wire                    labels_we,
        input  wire [$clog2(N_SAMPLES)-1:0] labels_addr,
        input  wire [LABEL_BITS-1:0]   labels_wdata,

        output logic [N_NEURONS*LABEL_BITS-1:0] assignments
);

    localparam int FP_SHIFT = 16;
    localparam int FP_SCALE = (1 << FP_SHIFT);

    // Memory for inputs
    logic [7:0] spikes_mem [0:N_SAMPLES-1][0:N_NEURONS-1];
    logic [LABEL_BITS-1:0] labels_mem [0:N_SAMPLES-1];

    // Rates in S16.16
    logic [31:0] rates [0:N_NEURONS-1][0:N_LABELS-1];
    logic [31:0] sum_spikes [0:N_NEURONS-1];
    logic [31:0] avg_fixed_reg;

    // Control counters
    logic [$clog2(N_SAMPLES)-1:0] sample_idx;
    logic [$clog2(N_NEURONS)-1:0] neuron_idx;
    logic [$clog2(N_LABELS)-1:0]  label_idx;
    logic [$clog2(N_LABELS)-1:0]  best_label;
    logic [31:0]                  best_rate;
    logic [$clog2(N_LABELS):0]    label_scan;
    logic [31:0]                  n_labeled;
    logic                         label_match;

    typedef enum logic [3:0] {
        S_IDLE,
        S_LABEL_CLEAR,
        S_SAMPLE_CHECK,
        S_NEURON_ACCUM,
        S_RATE_INIT,
        S_RATE_UPDATE,
        S_RATE_DIV_WAIT,
        S_ASSIGN_INIT,
        S_ASSIGN_SCAN,
        S_DONE
    } state_e;

    state_e state;

    // Divider for avg computation (S16.16)
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

    // Write inputs
    always_ff @(posedge clk) begin
        if (spikes_we) begin
            spikes_mem[spikes_sample_addr][spikes_neuron_addr] <= spikes_wdata;
        end
        if (labels_we) begin
            labels_mem[labels_addr] <= labels_wdata;
        end
    end

    // Main FSM
    integer n, l;
    always_ff @(posedge clk) begin
        if (rst) begin
            done <= 1'b0;
            state <= S_IDLE;
            sample_idx <= '0;
            neuron_idx <= '0;
            label_idx <= '0;
            n_labeled <= '0;
            best_label <= '0;
            best_rate <= '0;
            label_scan <= '0;
            label_match <= 1'b0;
            div_dividend <= '0;
            div_divisor  <= '0;
            div_valid_in <= 1'b0;
            avg_fixed_reg <= '0;
            for (n = 0; n < N_NEURONS; n = n + 1) begin
                assignments[n*LABEL_BITS +: LABEL_BITS] <= '0;
                sum_spikes[n] <= '0;
                for (l = 0; l < N_LABELS; l = l + 1) begin
                    rates[n][l] <= '0;
                end
            end
        end else begin
            div_valid_in <= 1'b0;
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        label_idx <= '0;
                        neuron_idx <= '0;
                        sample_idx <= '0;
                        state <= S_LABEL_CLEAR;
                    end
                end

                S_LABEL_CLEAR: begin
                    sum_spikes[neuron_idx] <= '0;
                    if (neuron_idx == N_NEURONS-1) begin
                        neuron_idx <= '0;
                        sample_idx <= '0;
                        n_labeled <= '0;
                        state <= S_SAMPLE_CHECK;
                    end else begin
                        neuron_idx <= neuron_idx + 1'b1;
                    end
                end

                S_SAMPLE_CHECK: begin
                    label_match <= (labels_mem[sample_idx] == label_idx);
                    neuron_idx <= '0;
                    state <= S_NEURON_ACCUM;
                end

                S_NEURON_ACCUM: begin
                    if (label_match) begin
                        sum_spikes[neuron_idx] <= sum_spikes[neuron_idx] + spikes_mem[sample_idx][neuron_idx];
                    end
                    if (neuron_idx == N_NEURONS-1) begin
                        if (label_match) begin
                            n_labeled <= n_labeled + 1'b1;
                        end
                        if (sample_idx == N_SAMPLES-1) begin
                            neuron_idx <= '0;
                            state <= S_RATE_INIT;
                        end else begin
                            sample_idx <= sample_idx + 1'b1;
                            state <= S_SAMPLE_CHECK;
                        end
                    end else begin
                        neuron_idx <= neuron_idx + 1'b1;
                    end
                end

                S_RATE_INIT: begin
                    neuron_idx <= '0;
                    state <= S_RATE_UPDATE;
                end

                S_RATE_UPDATE: begin
                    if (n_labeled != 0) begin
                        // avg = (sum_spikes<<16)/n_labeled with rounding
                        // NOTE: sum_spikes<<16 must fit in 32 bits for divider2b.
                        div_dividend <= (sum_spikes[neuron_idx] << FP_SHIFT) + (n_labeled >> 1);
                        div_divisor  <= n_labeled;
                        div_valid_in <= 1'b1;
                        state <= S_RATE_DIV_WAIT;
                    end else begin
                        if (neuron_idx == N_NEURONS-1) begin
                            if (label_idx == N_LABELS-1) begin
                                neuron_idx <= '0;
                                state <= S_ASSIGN_INIT;
                            end else begin
                                label_idx <= label_idx + 1'b1;
                                neuron_idx <= '0;
                                sample_idx <= '0;
                                n_labeled <= '0;
                                state <= S_LABEL_CLEAR;
                            end
                        end else begin
                            neuron_idx <= neuron_idx + 1'b1;
                        end
                    end
                end

                S_RATE_DIV_WAIT: begin
                    if (div_valid_out) begin
                        avg_fixed_reg <= div_quotient;
                        rates[neuron_idx][label_idx] <= rates[neuron_idx][label_idx] + div_quotient;
                        if (neuron_idx == N_NEURONS-1) begin
                            if (label_idx == N_LABELS-1) begin
                                neuron_idx <= '0;
                                state <= S_ASSIGN_INIT;
                            end else begin
                                label_idx <= label_idx + 1'b1;
                                neuron_idx <= '0;
                                sample_idx <= '0;
                                n_labeled <= '0;
                                state <= S_LABEL_CLEAR;
                            end
                        end else begin
                            neuron_idx <= neuron_idx + 1'b1;
                            state <= S_RATE_UPDATE;
                        end
                    end
                end

                S_ASSIGN_INIT: begin
                    best_label <= '0;
                    best_rate <= rates[neuron_idx][0];
                    label_scan <= 1;
                    state <= S_ASSIGN_SCAN;
                end

                S_ASSIGN_SCAN: begin
                    if (label_scan < N_LABELS) begin
                        if (rates[neuron_idx][label_scan] > best_rate) begin
                            best_rate <= rates[neuron_idx][label_scan];
                            best_label <= label_scan[LABEL_BITS-1:0];
                        end
                        label_scan <= label_scan + 1'b1;
                    end else begin
                        assignments[neuron_idx*LABEL_BITS +: LABEL_BITS] <= best_label;
                        if (neuron_idx == N_NEURONS-1) begin
                            state <= S_DONE;
                        end else begin
                            neuron_idx <= neuron_idx + 1'b1;
                            state <= S_ASSIGN_INIT;
                        end
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    if (!start) begin
                        state <= S_IDLE;
                    end
                end
            endcase
        end
    end

endmodule

`default_nettype wire
