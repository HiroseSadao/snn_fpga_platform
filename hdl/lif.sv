`default_nettype none

module lif(
        input  wire        clk,
        input  wire        start,     // pulse to reset and start
        input  wire        tick,      // LIF step tick
        output logic [9:0] spike_count,
        output logic       spike_pulse,
        output logic       running,
        output logic       done
    );

    // -----------------------------
    // LIF model (fixed-point, S16.16)
    // -----------------------------
    localparam int FP_SHIFT = 16;
    localparam int FP_SCALE = (1 << FP_SHIFT);

    localparam int V_REST   = -60;
    localparam int V_RESET  = -65;
    localparam int V_THR    = -40;
    // Keep parameter ratios consistent with the original Python model:
    // tau_m / dt = 0.01 / 0.00005 = 200 steps
    // tref  / dt = 0.002 / 0.00005 = 40 steps
    localparam int I_IN     =  21;  // constant input while running (matches Python)
    localparam int TAU_M    =  200; // steps (ratio preserved)
    localparam int REFRACT  =  40;  // steps (ratio preserved)

    localparam int V_REST_FP  = V_REST  * FP_SCALE;
    localparam int V_RESET_FP = V_RESET * FP_SCALE;
    localparam int V_THR_FP   = V_THR   * FP_SCALE;
    localparam int I_IN_FP    = I_IN    * FP_SCALE;

    logic signed [31:0] v_mem;
    logic [15:0]        refr_cnt;
    logic signed [31:0] num_calc;
    logic [31:0]        num_abs;
    logic               num_sign;

    typedef enum logic [1:0] {LIF_IDLE, LIF_DIV_WAIT} lif_state_e;
    lif_state_e lif_state;

    // Divider interface
    logic [31:0] div_dividend;
    logic [31:0] div_divisor;
    logic        div_valid_in;
    logic [31:0] div_quotient;
    logic [31:0] div_remainder;
    logic        div_valid_out;
    logic        div_error;
    logic        div_busy;

    logic signed [31:0] dv_signed;
    logic signed [31:0] v_next;

    always_comb begin
        num_calc = V_REST_FP - v_mem + I_IN_FP;
        num_sign = num_calc[31];
        num_abs  = num_sign ? (~num_calc + 1'b1) : num_calc;

        dv_signed = num_sign ? -$signed(div_quotient) : $signed(div_quotient);
        v_next    = v_mem + dv_signed;
    end

    divider2b #(.WIDTH(32)) u_divider(
        .clk_in        (clk),
        .rst_in        (start),
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
        if (start) begin
            running      <= 1'b1;
            done         <= 1'b0;
            spike_count  <= '0;
            v_mem        <= V_RESET_FP;
            refr_cnt     <= '0;
            spike_pulse  <= 1'b0;
            lif_state    <= LIF_IDLE;
            div_dividend <= '0;
            div_divisor  <= '0;
            div_valid_in <= 1'b0;
        end else begin
            div_valid_in <= 1'b0;
            spike_pulse  <= 1'b0;

            case (lif_state)
                LIF_IDLE: begin
                    if (tick && running && !done) begin
                        if (refr_cnt != 0) begin
                            refr_cnt <= refr_cnt - 1'b1;
                            v_mem    <= V_RESET_FP;
                        end else begin
                            div_dividend <= num_abs;
                            div_divisor  <= TAU_M[31:0];
                            div_valid_in <= 1'b1;
                            lif_state    <= LIF_DIV_WAIT;
                        end
                    end
                end

                LIF_DIV_WAIT: begin
                    if (div_valid_out) begin
                        if (v_next >= V_THR_FP) begin
                            spike_pulse <= 1'b1;
                            v_mem       <= V_RESET_FP;
                            refr_cnt    <= REFRACT;
                            spike_count <= spike_count + 1'b1;
                            if (spike_count >= 10'd99) begin
                                done    <= 1'b1;
                                running <= 1'b0;
                            end
                        end else begin
                            v_mem <= v_next;
                        end

                        lif_state <= LIF_IDLE;
                    end
                end

                default: lif_state <= LIF_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
