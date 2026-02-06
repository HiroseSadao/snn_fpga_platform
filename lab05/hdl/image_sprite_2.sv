`timescale 1ns / 1ps
`default_nettype none

`ifdef SYNTHESIS
`define FPATH(X) `"X`"
`else /* ! SYNTHESIS */
`define FPATH(X) `"../data/X`"
`endif  /* ! SYNTHESIS */

module image_sprite_2 #(
        parameter WIDTH=256, HEIGHT=256)
    (
        input wire pixel_clk,
        input wire rst,
        input wire [10:0] x, h_count,
        input wire [9:0]  y, v_count,
        input wire pop,
        output logic [7:0] pixel_red,
        output logic [7:0] pixel_green,
        output logic [7:0] pixel_blue
    );

    // calculate ROM address
    logic [$clog2(WIDTH*HEIGHT*2)-1:0] image_addr;
    always_comb begin
        if (pop)begin
            image_addr = (h_count - x) + ((v_count - y) * WIDTH);
        end else begin
            image_addr = (h_count - x) + ((v_count - y) * WIDTH) + WIDTH*HEIGHT;
        end
    end

    logic in_sprite;
    assign in_sprite = ((h_count >= x && h_count < (x + WIDTH)) &&
                        (v_count >= y && v_count < (y + HEIGHT)));

    logic dina;
    logic clka;
    logic wea;
    logic ena;
    logic rsta;
    logic regcea;
    logic [7:0] douta_from_image;
    logic [23:0] douta_from_palette;

    // Modify the module below to use your BRAMs!
    // this will not do anything without you doing that!
    always_comb begin
        if (in_sprite)begin
            pixel_red = douta_from_palette[23:16];
            pixel_green = douta_from_palette[15:8];
            pixel_blue = douta_from_palette[7:0];
        end else begin
            pixel_red = 0;
            pixel_green = 0;
            pixel_blue = 0;
        end
    end

    assign dina = 1'b0;
    assign clka = pixel_clk;
    assign wea = 1'b0;
    assign ena = 1'b1;
    assign rsta = rst;
    assign regcea = 1'b1;

    xilinx_single_port_ram_read_first #(
        .RAM_WIDTH(8),                       // Specify RAM data width
        .RAM_DEPTH(131072),                     // Specify RAM depth (number of entries)
        .RAM_PERFORMANCE("HIGH_PERFORMANCE"), // Select "HIGH_PERFORMANCE" or "LOW_LATENCY" 
        .INIT_FILE(`FPATH(image2.mem))          // Specify name/location of RAM initialization file if using one (leave blank if not)
    ) for_image (
        .addra(image_addr),     // Address bus, width determined from RAM_DEPTH
        .dina(dina),       // RAM input data, width determined from RAM_WIDTH
        .clka(clka),       // Clock
        .wea(wea),         // Write enable
        .ena(ena),         // RAM Enable, for additional power savings, disable port when not in use
        .rsta(rsta),       // Output reset (does not affect memory contents)
        .regcea(regcea),   // Output register enable
        .douta(douta_from_image)      // RAM output data, width determined from RAM_WIDTH
    );

    xilinx_single_port_ram_read_first #(
        .RAM_WIDTH(24),                       // Specify RAM data width
        .RAM_DEPTH(256),                     // Specify RAM depth (number of entries)
        .RAM_PERFORMANCE("HIGH_PERFORMANCE"), // Select "HIGH_PERFORMANCE" or "LOW_LATENCY" 
        .INIT_FILE(`FPATH(palette2.mem))          // Specify name/location of RAM initialization file if using one (leave blank if not)
    ) for_palette (
        .addra(douta_from_image),     // Address bus, width determined from RAM_DEPTH
        .dina(dina),       // RAM input data, width determined from RAM_WIDTH
        .clka(clka),       // Clock
        .wea(wea),         // Write enable
        .ena(ena),         // RAM Enable, for additional power savings, disable port when not in use
        .rsta(rsta),       // Output reset (does not affect memory contents)
        .regcea(regcea),   // Output register enable
        .douta(douta_from_palette)      // RAM output data, width determined from RAM_WIDTH
    );
endmodule






`default_nettype none
