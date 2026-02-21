`timescale 1ns / 1ps
`default_nettype none

//module takes in a 8 bit pixel and given two threshold values it:
//produces a 1 bit output indicating if the pixel is between (inclusive)
//those two threshold values
module threshold(
        input wire clk,
        input wire rst,
        input wire [7:0] pixel,
        input wire [7:0] lower_bound, upper_bound,
        output logic mask
    );
    always_ff @(posedge clk)begin
        if (rst)begin
            mask <= 0;
        end else begin
            mask <= (pixel > lower_bound) && (pixel <= upper_bound);
        end
    end
endmodule

`default_nettype wire
