`default_nettype none

module prediction #(
        parameter int N_SAMPLES  = 500,
        parameter int N_NEURONS  = 50,
        parameter int N_LABELS   = 10,
        parameter int LABEL_BITS = 4
    )(
        input  wire clk,
        input  wire rst,   // synchronous reset
        input  wire start,
        output logic done,

        // Write interface for spikes and assignments
        input  wire                     spikes_we,
        input  wire [$clog2(N_SAMPLES)-1:0] spikes_sample_addr,
        input  wire [$clog2(N_NEURONS)-1:0] spikes_neuron_addr,
        input  wire [7:0]               spikes_wdata,

        input  wire                     assignments_we,
        input  wire [$clog2(N_NEURONS)-1:0] assignments_addr,
        input  wire [LABEL_BITS-1:0]    assignments_wdata,

        output logic [N_SAMPLES*LABEL_BITS-1:0] predictions
    );

    localparam int FP_SHIFT = 16;

    // Memories
    logic [7:0] spikes_mem [0:N_SAMPLES-1][0:N_NEURONS-1];
    logic [LABEL_BITS-1:0] assignments_mem [0:N_NEURONS-1];

    // n_assigns per label
    logic [15:0] n_assigns [0:N_LABELS-1];

    // Counters / accumulators
    logic [$clog2(N_SAMPLES)-1:0] sample_idx;
    logic [$clog2(N_NEURONS)-1:0] neuron_idx;
    logic [$clog2(N_LABELS)-1:0]  label_idx;
    logic [31:0] sum_spikes;

    logic [LABEL_BITS-1:0] best_label;
    logic [31:0] best_rate;

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

    // Write inputs
    always_ff @(posedge clk) begin
        if (spikes_we) begin
            spikes_mem[spikes_sample_addr][spikes_neuron_addr] <= spikes_wdata;
        end
        if (assignments_we) begin
            assignments_mem[assignments_addr] <= assignments_wdata;
        end
    end

    typedef enum logic [3:0] {
        S_IDLE,
        S_CLEAR_ASSIGNS,
        S_COUNT_ASSIGNS,
        S_SAMPLE_INIT,
        S_LABEL_INIT,
        S_NEURON_ACCUM,
        S_DIV_START,
        S_DIV_WAIT,
        S_LABEL_NEXT,
        S_SAMPLE_NEXT,
        S_DONE
    } state_e;

    state_e state;

    integer i;
    always_ff @(posedge clk) begin
        if (rst) begin
            done <= 1'b0;
            state <= S_IDLE;
            sample_idx <= '0;
            neuron_idx <= '0;
            label_idx <= '0;
            sum_spikes <= '0;
            best_label <= '0;
            best_rate <= '0;
            div_dividend <= '0;
            div_divisor <= '0;
            div_valid_in <= 1'b0;
            for (i = 0; i < N_LABELS; i = i + 1) begin
                n_assigns[i] <= '0;
            end
            for (i = 0; i < N_SAMPLES; i = i + 1) begin
                predictions[i*LABEL_BITS +: LABEL_BITS] <= '0;
            end
        end else begin
            div_valid_in <= 1'b0;
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        label_idx <= '0;
                        state <= S_CLEAR_ASSIGNS;
                    end
                end

                S_CLEAR_ASSIGNS: begin
                    n_assigns[label_idx] <= '0;
                    if (label_idx == N_LABELS-1) begin
                        neuron_idx <= '0;
                        state <= S_COUNT_ASSIGNS;
                    end else begin
                        label_idx <= label_idx + 1'b1;
                    end
                end

                S_COUNT_ASSIGNS: begin
                    n_assigns[assignments_mem[neuron_idx]] <= n_assigns[assignments_mem[neuron_idx]] + 1'b1;
                    if (neuron_idx == N_NEURONS-1) begin
                        sample_idx <= '0;
                        state <= S_SAMPLE_INIT;
                    end else begin
                        neuron_idx <= neuron_idx + 1'b1;
                    end
                end

                S_SAMPLE_INIT: begin
                    best_label <= '0;
                    best_rate <= '0;
                    label_idx <= '0;
                    state <= S_LABEL_INIT;
                end

                S_LABEL_INIT: begin
                    sum_spikes <= '0;
                    neuron_idx <= '0;
                    state <= S_NEURON_ACCUM;
                end

                S_NEURON_ACCUM: begin
                    if (assignments_mem[neuron_idx] == label_idx) begin
                        sum_spikes <= sum_spikes + spikes_mem[sample_idx][neuron_idx];
                    end
                    if (neuron_idx == N_NEURONS-1) begin
                        state <= S_DIV_START;
                    end else begin
                        neuron_idx <= neuron_idx + 1'b1;
                    end
                end

                S_DIV_START: begin
                    if (n_assigns[label_idx] != 0) begin
                        div_dividend <= (sum_spikes << FP_SHIFT) + (n_assigns[label_idx] >> 1);
                        div_divisor  <= n_assigns[label_idx];
                        div_valid_in <= 1'b1;
                        state <= S_DIV_WAIT;
                    end else begin
                        // no assigned neurons: rate = 0
                        state <= S_LABEL_NEXT;
                    end
                end

                S_DIV_WAIT: begin
                    if (div_valid_out) begin
                        if (div_quotient > best_rate) begin
                            best_rate <= div_quotient;
                            best_label <= label_idx;
                        end
                        state <= S_LABEL_NEXT;
                    end
                end

                S_LABEL_NEXT: begin
                    if (label_idx == N_LABELS-1) begin
                        predictions[sample_idx*LABEL_BITS +: LABEL_BITS] <= best_label;
                        state <= S_SAMPLE_NEXT;
                    end else begin
                        label_idx <= label_idx + 1'b1;
                        state <= S_LABEL_INIT;
                    end
                end

                S_SAMPLE_NEXT: begin
                    if (sample_idx == N_SAMPLES-1) begin
                        state <= S_DONE;
                    end else begin
                        sample_idx <= sample_idx + 1'b1;
                        state <= S_SAMPLE_INIT;
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
