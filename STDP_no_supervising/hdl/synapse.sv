`default_nettype none

module synapse(
        input  wire              clk,
        input  wire              rst,    // synchronous reset
        input  wire              tick,   // step update
        input  wire              spike,  // 1 when presynaptic spike
        output logic signed [31:0] r_out  // S16.16
    );

    // -----------------------------
    // Single exponential synapse (S16.16)
    // r = r*(1 - dt/td) + spike/td
    // Use dt=1 step, td=50 steps (matches dt=1e-4, td=5e-3 ratio)
    // -----------------------------
    localparam int FP_SHIFT = 16;
    localparam int FP_SCALE = (1 << FP_SHIFT);

    localparam int TD_STEPS = 50;
    localparam int TD_HALF  = TD_STEPS / 2;
    localparam int SPIKE_ADD = FP_SCALE / TD_STEPS; // 1/td in S16.16 (rounded by integer div)

    typedef enum logic [0:0] {S_IDLE, S_DIV_WAIT} state_e;
    state_e state;

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

    always_ff @(posedge clk) begin
        if (rst) begin
            r_out        <= '0;
            state        <= S_IDLE;
            div_dividend <= '0;
            div_divisor  <= '0;
            div_valid_in <= 1'b0;
        end else begin
            div_valid_in <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (tick) begin
                        // rounded r/td
                        div_dividend <= r_out[31:0] + TD_HALF[31:0];
                        div_divisor  <= TD_STEPS[31:0];
                        div_valid_in <= 1'b1;
                        state        <= S_DIV_WAIT;
                    end
                end

                S_DIV_WAIT: begin
                    if (div_valid_out) begin
                        // r_next = r - r/td + spike/td
                        if (spike) begin
                            r_out <= $signed(r_out) - $signed(div_quotient) + $signed(SPIKE_ADD);
                        end else begin
                            r_out <= $signed(r_out) - $signed(div_quotient);
                        end
                        state <= S_IDLE;
                    end
                end
            endcase
        end
    end

endmodule

`default_nettype wire
