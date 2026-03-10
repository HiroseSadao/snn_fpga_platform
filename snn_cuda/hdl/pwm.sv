module pwm(   input wire clk,
              input wire rst,
              input wire [7:0] dc_in,
              output logic sig_out);
 
    logic [31:0] count;
    logic [7:0]  dc_latched;

    counter mc (.clk(clk),
                .rst(rst),
                .period(255),
                .count(count));
    
    always_ff @(posedge clk) begin
        if (rst) begin
            dc_latched <= '0;
        end else if (count == 0) begin
            dc_latched <= dc_in;
        end
    end

    assign sig_out = count<dc_latched;
endmodule