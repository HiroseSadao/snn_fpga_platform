`default_nettype none

module lif_inh(
        input  wire        clk,
        input  wire        start,     // pulse to reset and start
        input  wire        tick,      // LIF step tick
        input  wire signed [31:0] g_exc, // S16.16
        input  wire signed [31:0] g_inh, // S16.16
        output logic [9:0] spike_count,
        output logic       spike_pulse,
        output logic       running,
        output logic       done,
        output logic       step_done
    );

    // -----------------------------
    // Inhibitory LIF (fixed-point, S16.16)
    // Conductance-based without adaptive threshold
    // -----------------------------
    localparam int FP_SHIFT = 16;
    localparam int FP_SCALE = (1 << FP_SHIFT);

    localparam int V_REST   = -60;
    localparam int V_RESET  = -45;
    localparam int V_THR    = -40;
    localparam int V_PEAK   =  20;
    // dt=1e-3, tc_m=1e-2 => 10 steps
    // dt=1e-3, tref=2e-3 => 2 steps
    localparam int TAU_M    =  10;
    localparam int REFRACT  =  2;
    localparam int TAU_M_HALF = TAU_M / 2;

    localparam int E_EXC    =   0;
    localparam int E_INH    = -85;

    localparam int V_REST_FP   = V_REST   * FP_SCALE;
    localparam int V_RESET_FP  = V_RESET  * FP_SCALE;
    localparam int V_THR_FP    = V_THR    * FP_SCALE;
    localparam int V_PEAK_FP   = V_PEAK   * FP_SCALE;
    localparam int E_EXC_FP    = E_EXC    * FP_SCALE;
    localparam int E_INH_FP    = E_INH    * FP_SCALE;

    logic signed [31:0] v_mem;
    logic [15:0]        refr_cnt;
    logic signed [31:0] num_calc;
    logic [31:0]        num_abs;
    logic               num_sign;

    typedef enum logic [0:0] {LIF_IDLE, LIF_DIV_WAIT} lif_state_e;
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

    always_comb begin
        mul_tmp_exc = $signed(g_exc) * $signed(E_EXC_FP - v_mem);
        mul_tmp_inh = $signed(g_inh) * $signed(E_INH_FP - v_mem);
        i_syn_exc = $signed(mul_tmp_exc >>> FP_SHIFT);
        i_syn_inh = $signed(mul_tmp_inh >>> FP_SHIFT);

        num_calc = V_REST_FP - v_mem + i_syn_exc + i_syn_inh;
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
            step_done    <= 1'b0;
            lif_state    <= LIF_IDLE;
            div_dividend <= '0;
            div_divisor  <= '0;
            div_valid_in <= 1'b0;
        end else begin
            div_valid_in <= 1'b0;
            spike_pulse  <= 1'b0;
            step_done    <= 1'b0;

            case (lif_state)
                LIF_IDLE: begin
                    if (tick && running && !done) begin
                        if (refr_cnt != 0) begin
                            refr_cnt <= refr_cnt - 1'b1;
                            v_mem    <= V_RESET_FP;
                            step_done <= 1'b1;
                        end else begin
                            div_dividend <= num_abs + TAU_M_HALF[31:0];
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
                        step_done <= 1'b1;
                        lif_state <= LIF_IDLE;
                    end
                end

                default: lif_state <= LIF_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
