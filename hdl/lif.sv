`default_nettype none

module lif(
        input  wire        clk,
        input  wire        start,     // pulse to reset and start
        input  wire        tick,      // LIF step tick
        output logic [9:0] spike_count,
        output logic       spike_pulse,
        output logic       running,
        output logic       done,
        output logic signed [31:0] theta_out,
        output logic signed [31:0] vthr_out
    );

    // -----------------------------
    // LIF model (fixed-point, S16.16)
    // -----------------------------
    localparam int FP_SHIFT = 16;
    localparam int FP_SCALE = (1 << FP_SHIFT);

    localparam int V_REST   = -65;
    localparam int V_RESET  = -65;
    localparam int INIT_VTHR = -52;
    localparam int V_PEAK   =  20;
    // Keep parameter ratios consistent with the original Python model:
    // tau_m / dt = 0.01 / 0.00005 = 200 steps
    // tref  / dt = 0.002 / 0.00005 = 40 steps
    localparam int TAU_M    =  200;   // steps (ratio preserved)
    localparam int REFRACT  =  40;    // steps (ratio preserved)
    localparam int TC_THETA =  10000; // steps (ratio preserved)
    localparam int THETA_MAX = 35;

    // Synapse params (fixed g_exc/g_inh)
    localparam int E_EXC    =   0;
    localparam int E_INH    = -100;
    localparam int THETA_PLUS_FP = 3277; // 0.05 in S16.16
    // g_exc chosen to approximate constant input of 21 at v=-65: 21/65 ≈ 0.3230769
    localparam int G_EXC_FP = 21134; // 0.3230769 in S16.16
    localparam int G_INH_FP = 0;

    localparam int V_REST_FP   = V_REST   * FP_SCALE;
    localparam int V_RESET_FP  = V_RESET  * FP_SCALE;
    localparam int INIT_VTHR_FP = INIT_VTHR * FP_SCALE;
    localparam int V_PEAK_FP   = V_PEAK   * FP_SCALE;
    localparam int E_EXC_FP    = E_EXC    * FP_SCALE;
    localparam int E_INH_FP    = E_INH    * FP_SCALE;
    localparam int THETA_MAX_FP = THETA_MAX * FP_SCALE;

    logic signed [31:0] v_mem;
    logic signed [31:0] vthr_reg;
    logic signed [31:0] theta_reg;
    logic [15:0]        refr_cnt;
    logic signed [31:0] num_calc;
    logic [31:0]        num_abs;
    logic               num_sign;
    logic               spike_latched;

    typedef enum logic [1:0] {LIF_IDLE, LIF_DIV_WAIT, LIF_THETA_WAIT} lif_state_e;
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
    logic signed [63:0] mul_tmp_exc;
    logic signed [63:0] mul_tmp_inh;
    logic signed [31:0] i_syn_exc;
    logic signed [31:0] i_syn_inh;
    logic signed [31:0] theta_div;
    logic signed [31:0] theta_next;

    always_comb begin
        mul_tmp_exc = $signed(G_EXC_FP) * $signed(E_EXC_FP - v_mem);
        mul_tmp_inh = $signed(G_INH_FP) * $signed(E_INH_FP - v_mem);
        i_syn_exc = $signed(mul_tmp_exc >>> FP_SHIFT);
        i_syn_inh = $signed(mul_tmp_inh >>> FP_SHIFT);

        num_calc = V_REST_FP - v_mem + i_syn_exc + i_syn_inh;
        num_sign = num_calc[31];
        num_abs  = num_sign ? (~num_calc + 1'b1) : num_calc;

        dv_signed = num_sign ? -$signed(div_quotient) : $signed(div_quotient);
        v_next    = v_mem + dv_signed;

        theta_div = $signed(div_quotient);
        if (spike_latched) begin
            theta_next = $signed(theta_reg - theta_div + THETA_PLUS_FP);
        end else begin
            theta_next = $signed(theta_reg - theta_div);
        end
        if (theta_next < 0) begin
            theta_next = 0;
        end else if (theta_next > $signed(THETA_MAX_FP)) begin
            theta_next = THETA_MAX_FP;
        end
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
            theta_reg    <= '0;
            vthr_reg     <= INIT_VTHR_FP;
            spike_latched <= 1'b0;
        end else begin
            div_valid_in <= 1'b0;
            spike_pulse  <= 1'b0;

            case (lif_state)
                LIF_IDLE: begin
                    if (tick && running && !done) begin
                        if (refr_cnt != 0) begin
                            refr_cnt <= refr_cnt - 1'b1;
                            v_mem    <= V_RESET_FP;
                            spike_latched <= 1'b0;
                            div_dividend <= theta_reg[31:0];
                            div_divisor  <= TC_THETA[31:0];
                            div_valid_in <= 1'b1;
                            lif_state    <= LIF_THETA_WAIT;
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
                        if (v_next >= vthr_reg) begin
                            spike_pulse <= 1'b1;
                            spike_latched <= 1'b1;
                            v_mem       <= V_RESET_FP;
                            refr_cnt    <= REFRACT;
                            spike_count <= spike_count + 1'b1;
                            if (spike_count >= 10'd99) begin
                                done    <= 1'b1;
                                running <= 1'b0;
                            end
                        end else begin
                            v_mem <= v_next;
                            spike_latched <= 1'b0;
                        end

                        // Start theta decay division
                        div_dividend <= theta_reg[31:0];
                        div_divisor  <= TC_THETA[31:0];
                        div_valid_in <= 1'b1;
                        lif_state    <= LIF_THETA_WAIT;
                    end
                end

                LIF_THETA_WAIT: begin
                    if (div_valid_out) begin
                        theta_reg <= theta_next;
                        vthr_reg <= $signed(INIT_VTHR_FP + theta_next);
                        lif_state <= LIF_IDLE;
                    end
                end

                default: lif_state <= LIF_IDLE;
            endcase
        end
    end

    assign theta_out = theta_reg;
    assign vthr_out  = vthr_reg;

endmodule

`default_nettype wire
