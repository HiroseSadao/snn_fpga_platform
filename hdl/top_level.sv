`default_nettype none // prevents system from inferring an undeclared logic (good practice)

module top_level(
        input  wire        clk_100mhz,
        input  wire [15:0] sw, //all 16 input slide switches
        input  wire [3:0]  btn, //all four momentary button switches
        output logic [15:0] led, //16 green output LEDs (located right above switches)
        output logic [2:0] rgb0, //RGB channels of RGB LED0
        output logic [2:0] rgb1, //RGB channels of RGB LED1
        output logic [3:0] ss0_an,//anode control for upper four digits of seven-seg display
        output logic [3:0] ss1_an,//anode control for lower four digits of seven-seg display
        output logic [6:0] ss0_c, //cathode controls for the segments of upper four digits
        output logic [6:0] ss1_c //cathod controls for the segments of lower four digits
    );

    // -----------------------------
    // Button sync + edge detection
    // -----------------------------
    logic btn0_ff1, btn0_ff2;
    always_ff @(posedge clk_100mhz) begin
        btn0_ff1 <= btn[0];
        btn0_ff2 <= btn0_ff1;
    end
    wire btn0_rise = btn0_ff1 & ~btn0_ff2;

    // -----------------------------
    // LIF simulation timing (slow)
    // -----------------------------
    localparam int CLK_HZ        = 100_000_000;
    localparam int LIF_TICK_HZ   = 500; // 2ms per step (10x faster than previous)
    localparam int LIF_DIV       = CLK_HZ / LIF_TICK_HZ;
    localparam int LIF_DIV_W     = $clog2(LIF_DIV);

    logic [LIF_DIV_W-1:0] lif_cnt;
    logic lif_tick;

    always_ff @(posedge clk_100mhz) begin
        if (btn0_rise) begin
            lif_cnt  <= '0;
            lif_tick <= 1'b0;
        end else if (lif_cnt == LIF_DIV - 1) begin
            lif_cnt  <= '0;
            lif_tick <= 1'b1;
        end else begin
            lif_cnt  <= lif_cnt + 1'b1;
            lif_tick <= 1'b0;
        end
    end

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

    logic running;
    logic done;
    logic [9:0] spike_count;
    logic signed [31:0] v_mem;
    logic [3:0] refr_cnt;
    logic spike_pulse;
    logic signed [31:0] num_calc;
    logic [31:0] num_abs;
    logic num_sign;

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
        .clk_in        (clk_100mhz),
        .rst_in        (btn0_rise),
        .dividend_in   (div_dividend),
        .divisor_in    (div_divisor),
        .data_valid_in (div_valid_in),
        .quotient_out  (div_quotient),
        .remainder_out (div_remainder),
        .data_valid_out(div_valid_out),
        .error_out     (div_error),
        .busy_out      (div_busy)
    );

    always_ff @(posedge clk_100mhz) begin
        if (btn0_rise) begin
            running     <= 1'b1;
            done        <= 1'b0;
            spike_count <= '0;
            v_mem       <= V_RESET_FP;
            refr_cnt    <= '0;
            spike_pulse <= 1'b0;
            lif_state   <= LIF_IDLE;
            div_dividend <= '0;
            div_divisor  <= '0;
            div_valid_in <= 1'b0;
        end else begin
            div_valid_in <= 1'b0;
            spike_pulse  <= 1'b0;

            case (lif_state)
                LIF_IDLE: begin
                    if (lif_tick && running && !done) begin
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

    // -----------------------------
    // Binary -> 8-digit decimal
    // -----------------------------
    logic [3:0] digit [7:0];
    integer tmp;
    always_comb begin
        tmp = spike_count;
        digit[0] = tmp % 10; tmp = tmp / 10;
        digit[1] = tmp % 10; tmp = tmp / 10;
        digit[2] = tmp % 10; tmp = tmp / 10;
        digit[3] = tmp % 10; tmp = tmp / 10;
        digit[4] = tmp % 10; tmp = tmp / 10;
        digit[5] = tmp % 10; tmp = tmp / 10;
        digit[6] = tmp % 10; tmp = tmp / 10;
        digit[7] = tmp % 10;
    end

    // -----------------------------
    // 7-seg scan (8 digits)
    // -----------------------------
    localparam int SCAN_HZ    = 1000;
    localparam int SCAN_DIV   = CLK_HZ / SCAN_HZ;
    localparam int SCAN_DIV_W = $clog2(SCAN_DIV);

    logic [SCAN_DIV_W-1:0] scan_cnt;
    logic [2:0] digit_idx;
    logic [3:0] digit_val;
    logic [6:0] seg_cn;

    always_ff @(posedge clk_100mhz) begin
        if (btn0_rise) begin
            scan_cnt  <= '0;
            digit_idx <= '0;
        end else if (scan_cnt == SCAN_DIV - 1) begin
            scan_cnt  <= '0;
            digit_idx <= digit_idx + 1'b1;
        end else begin
            scan_cnt <= scan_cnt + 1'b1;
        end
    end

    // Power-on init for stable display before first button press.
    initial begin
        lif_cnt     = '0;
        lif_tick    = 1'b0;
        running     = 1'b0;
        done        = 1'b0;
        spike_count = '0;
        v_mem       = V_RESET_FP;
        refr_cnt    = '0;
        spike_pulse = 1'b0;
        scan_cnt    = '0;
        digit_idx   = '0;
    end

    always_comb begin
        ss0_an = 4'hF; // all off (active low)
        ss1_an = 4'hF; // all off (active low)

        digit_val = digit[digit_idx];
        if (digit_idx < 3'd4) begin
            ss1_an[digit_idx] = 1'b0;
        end else begin
            ss0_an[digit_idx - 3'd4] = 1'b0;
        end
    end

    bto7s u_bto7s(
        .x(digit_val),
        .s(seg_cn)
    );

    assign ss0_c = ~seg_cn; // active low cathodes
    assign ss1_c = ~seg_cn;

    // -----------------------------
    // LEDs and RGBs (quiet)
    // -----------------------------
    assign led  = 16'b0;
    assign rgb0 = 3'b0;
    assign rgb1 = 3'b0;

endmodule // top_level
/* I usually add a comment to associate my endmodule line with the module name
 * this helps when if you have multiple module definitions in a file
 */

// reset the default net type to wire, sometimes other code expects this.
`default_nettype wire
