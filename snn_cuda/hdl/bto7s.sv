`default_nettype none
module bto7s(
        input wire [3:0]   x,
        output logic [6:0] s
        );

        always_comb begin
            if      (x == 4'h0) s = 7'b0111111;
            else if (x == 4'h1) s = 7'b0000110;
            else if (x == 4'h2) s = 7'b1011011;
            else if (x == 4'h3) s = 7'b1001111;
            else if (x == 4'h4) s = 7'b1100110;
            else if (x == 4'h5) s = 7'b1101101;
            else if (x == 4'h6) s = 7'b1111101;
            else if (x == 4'h7) s = 7'b0000111;
            else if (x == 4'h8) s = 7'b1111111;
            else if (x == 4'h9) s = 7'b1101111;
            else if (x == 4'hA) s = 7'b1110111;
            else if (x == 4'hB) s = 7'b1111100;
            else if (x == 4'hC) s = 7'b0111001;
            else if (x == 4'hD) s = 7'b1011110;
            else if (x == 4'hE) s = 7'b1111001;
            else if (x == 4'hF) s = 7'b1110001;
            else                s = 7'b0000000;
        end

endmodule
