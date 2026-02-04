`default_nettype none

module top_level(
        input  wire        clk_100mhz,
        input  wire [15:0] sw,  // unused
        input  wire [3:0]  btn, // btn[0] used as reset
        output logic [15:0] led,
        output logic [2:0] rgb0,
        output logic [2:0] rgb1,
        output logic [3:0] ss0_an,
        output logic [3:0] ss1_an,
        output logic [6:0] ss0_c,
        output logic [6:0] ss1_c,

        // SD card socket signals (SPI mode)
        input  wire SD_DQ0,   // MISO
        output wire SD_DQ1,   // hold HIGH
        output wire SD_DQ2,   // hold HIGH
        output wire SD_DQ3,   // CS
        output wire SD_CMD,   // MOSI
        output wire SD_CLK,   // SCLK
        input  wire SD_CD_N   // card detect, active low
    );

    // -----------------------------
    // Clocks and reset
    // -----------------------------
    logic stop_latched;
    always_ff @(posedge clk_100mhz) begin
        if (btn[0]) begin
            stop_latched <= 1'b1;
        end
    end
    wire reset = stop_latched;

    // 25 MHz clock from 100 MHz input (divide by 4)
    logic [1:0] clk_div;
    always_ff @(posedge clk_100mhz) begin
        if (reset) begin
            clk_div <= 2'b0;
        end else begin
            clk_div <= clk_div + 2'b1;
        end
    end
    wire clk_25mhz = clk_div[1];

    // -----------------------------
    // SD controller instance
    // -----------------------------
    logic        rd;
    logic        wr;
    logic [31:0] address;
    logic [7:0]  dout;
    logic        byte_available;
    logic        ready;
    logic [4:0]  status;

    sd_controller u_sd(
        .cs                 (SD_DQ3),
        .mosi               (SD_CMD),
        .miso               (SD_DQ0),
        .sclk               (SD_CLK),
        .rd                 (rd),
        .dout               (dout),
        .byte_available     (byte_available),
        .wr                 (wr),
        .din                (8'h00),
        .ready_for_next_byte(),
        .reset              (reset),
        .ready              (ready),
        .address            (address),
        .clk                (clk_25mhz),
        .status             (status)
    );

    assign SD_DQ1 = 1'b1; // SPI mode: DAT1/2 should be HIGH
    assign SD_DQ2 = 1'b1;

    // -----------------------------
    // Simple read of sector 2048 (LBA2048)
    // -----------------------------
    localparam int SECTOR_BYTES = 512;
    logic [8:0]  byte_count;
    logic [7:0]  last_byte;
    logic [7:0]  first_bytes [0:15];
    logic        in_read;

    always_ff @(posedge clk_25mhz) begin
        if (reset) begin
            rd         <= 1'b0;
            wr         <= 1'b0;
            address    <= 32'd2048;
            byte_count <= 9'd0;
            last_byte  <= 8'd0;
            in_read    <= 1'b0;
        end else begin
            rd <= 1'b0; // default: pulse

            if (!in_read && ready && (SD_CD_N == 1'b0)) begin
                // start a single-block read at address 0
                rd      <= 1'b1;
                in_read <= 1'b1;
                byte_count <= 9'd0;
            end

            if (byte_available) begin
                last_byte <= dout;
                if (byte_count < 16) begin
                    first_bytes[byte_count] <= dout;
                end
                if (byte_count == SECTOR_BYTES - 1) begin
                    in_read <= 1'b0; // done
                end else begin
                    byte_count <= byte_count + 1'b1;
                end
            end
        end
    end

    // -----------------------------
    // Debug outputs
    // -----------------------------
    // led[7:0]   = selected byte from first 16 bytes (sw[3:0])
    // led[8]     = header match 'S'
    // led[9]     = header match 'P'
    // led[10]    = header match 'K'
    // led[11]    = header match '1'
    // led[12]    = ready
    // led[13]    = byte_available
    // led[14]    = in_read
    // led[15]    = card present (active low)
    always_comb begin
        led = 16'b0;
        if (!stop_latched) begin
            led[7:0]   = first_bytes[sw[3:0]];
            led[8]     = (first_bytes[0] == 8'h53); // 'S'
            led[9]     = (first_bytes[1] == 8'h50); // 'P'
            led[10]    = (first_bytes[2] == 8'h4B); // 'K'
            led[11]    = (first_bytes[3] == 8'h31); // '1'
            led[12]    = ready;
            led[13]    = byte_available;
            led[14]    = in_read;
            led[15]    = ~SD_CD_N;
        end
    end

    // Keep unused outputs quiet
    assign rgb0  = 3'b0;
    assign rgb1  = 3'b0;
    assign ss0_an = 4'hF;
    assign ss1_an = 4'hF;
    assign ss0_c  = 7'h7F;
    assign ss1_c  = 7'h7F;

endmodule

`default_nettype wire
