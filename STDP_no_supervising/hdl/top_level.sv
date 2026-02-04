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
    // LIF model (separate module)
    // -----------------------------
    logic running;
    logic done;
    logic [9:0] spike_count;
    logic spike_pulse;
    logic signed [31:0] theta_dbg;
    logic signed [31:0] vthr_dbg;

    lif u_lif(
        .clk         (clk_100mhz),
        .start       (btn0_rise),
        .tick        (lif_tick),
        .spike_count (spike_count),
        .spike_pulse (spike_pulse),
        .running     (running),
        .done        (done),
        .theta_out   (theta_dbg),
        .vthr_out    (vthr_dbg)
    );

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
