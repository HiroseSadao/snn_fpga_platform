`timescale 1ns / 1ps
`default_nettype none

//module takes in six different channels of color (all 8 bit)
// based on switches, combinationally routes one of them to selected_channel
module channel_select(
        input wire [2:0] select,
        input wire [7:0] r, g, b,
        input wire [7:0] y, cr, cb,
        output logic [7:0] selected_channel
    );
    always_comb begin
        case (select)
            3'b000:   selected_channel = g;
            3'b001:   selected_channel = r;
            3'b010:   selected_channel = b;
            3'b011:   selected_channel = 5'b0;
            3'b100:   selected_channel = y;
            3'b101:   selected_channel = cr;
            3'b110:   selected_channel = cb;
            3'b111:   selected_channel = 5'b0;
            default:  selected_channel = 5'b0;
        endcase
    end
endmodule


`default_nettype wire
