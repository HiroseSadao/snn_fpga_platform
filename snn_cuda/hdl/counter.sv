module counter(     input wire clk,
                    input wire rst,
                    input wire [31:0] period,
                    output logic [31:0] count
              );

    //your code here
    always_ff @(posedge clk) begin
        if (rst) begin
            count <= 32'd0;
        end else if (count >= period - 1) begin
            count <= 32'd0;
        end else begin
            count <= count + 1;
        end
    end
endmodule