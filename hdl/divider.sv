module divider2b #(parameter WIDTH = 32) (
    input  wire                   clk_in,
    input  wire                   rst_in,
    input  wire [WIDTH-1:0]       dividend_in,
    input  wire [WIDTH-1:0]       divisor_in,
    input  wire                   data_valid_in,
    output logic [WIDTH-1:0]      quotient_out,
    output logic [WIDTH-1:0]      remainder_out,
    output logic                  data_valid_out,
    output logic                  error_out,
    output logic                  busy_out
);

    typedef enum logic { RESTING, DIVIDING } state_e;
    state_e                      state;

    logic [WIDTH-1:0]            dividend;   // holds running quotient bits in LSBs
    logic [WIDTH-1:0]            divisor;
    logic [WIDTH-1:0]            quotient;   // (not used internally; final comes from dividend)
    logic [31:0]                 p;          // partial remainder (N+1 bits ok with 32)
    logic [5:0]                  count;      // counts remaining iterations (32 -> 0)

    // -------- First iteration (combinational “temp” results) --------
    logic [31:0]                 p_temp;     // result of first iteration this cycle
    logic [31:0]                 div_temp;   // dividend after first iteration this cycle

    // Compute the FIRST iteration purely combinationally from current (p, dividend).
    // Do NOT mix blocking/non-blocking in the sequential always_ff.
    always_comb begin
        if ( {p[30:0], dividend[31]} >= divisor[31:0] ) begin
            p_temp   = {p[30:0], dividend[31]} - divisor[31:0];
            div_temp = {dividend[30:0], 1'b1};
        end else begin
            p_temp   = {p[30:0], dividend[31]};
            div_temp = {dividend[30:0], 1'b0};
        end
    end

    // ------------------------------ Sequential ------------------------------
    always_ff @(posedge clk_in) begin
        if (rst_in) begin
            quotient        <= '0;
            dividend        <= '0;
            divisor         <= '0;
            remainder_out   <= '0;
            data_valid_out  <= 1'b0;
            error_out       <= 1'b0;
            busy_out        <= 1'b0;
            state           <= RESTING;
            count           <= '0;
            p               <= '0;
            quotient_out    <= '0;
        end else begin
            case (state)
                // ------------------------------------------------------------------
                RESTING: begin
                    data_valid_out <= 1'b0;
                    busy_out       <= 1'b0;

                    if (data_valid_in) begin
                        if (divisor_in == 0) begin
                            quotient_out   <= '0;
                            remainder_out  <= '0;
                            error_out      <= 1'b1;
                            data_valid_out <= 1'b1;
                            busy_out       <= 1'b0;
                            state          <= RESTING; // stay idle
                        end else begin
                            state      <= DIVIDING;
                            quotient   <= '0;
                            dividend   <= dividend_in;
                            divisor    <= divisor_in;
                            p          <= '0;
                            count      <= 6'd31;
                            error_out  <= 1'b0;
                            busy_out   <= 1'b1;
                            data_valid_out <= 1'b0;
                        end
                    end
                end


                // ------------------------------------------------------------------
                DIVIDING: begin
                    if (count == 6'd1) begin
                        // Last iteration: we already computed the first half (p_temp/div_temp).
                        // Final output uses that ONE remaining step's result.
                        state <= RESTING;

                        if ( {p_temp[30:0], div_temp[31]} >= divisor[31:0] ) begin
                            remainder_out <= {p_temp[30:0], div_temp[31]} - divisor[31:0];
                            quotient_out  <= {div_temp[30:0], 1'b1};
                        end else begin
                            remainder_out <= {p_temp[30:0], div_temp[31]};
                            quotient_out  <= {div_temp[30:0], 1'b0};
                        end

                        data_valid_out <= 1'b1;   // good stuff!
                        error_out      <= 1'b0;
                        busy_out       <= 1'b0;
                    end else begin
                        // Do the SECOND iteration this cycle, using p_temp/div_temp
                        if ( {p_temp[30:0], div_temp[31]} >= divisor[31:0] ) begin
                            p        <= {p_temp[30:0], div_temp[31]} - divisor[31:0];
                            dividend <= {div_temp[30:0], 1'b1};
                        end else begin
                            p        <= {p_temp[30:0], div_temp[31]};
                            dividend <= {div_temp[30:0], 1'b0};
                        end
                        count          <= count - 6'd2;  // two iterations consumed
                        data_valid_out <= 1'b0;
                    end
                end
            endcase
        end
    end

endmodule
