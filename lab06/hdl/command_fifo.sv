`timescale 1ns / 1ps
`default_nettype none
 
module command_fifo #(parameter DEPTH=16, parameter WIDTH=16)(
        input  wire                 clk,
        input  wire                 rst,
        input  wire                 write,
        input  wire [WIDTH-1:0]     command_in,
        output logic                full,
 
        output logic [WIDTH-1:0]    command_out,   // async (combinational) read
        input  wire                 read,
        output logic                empty
    );

    localparam int AW = $clog2(DEPTH);

    logic [AW-1:0] write_pointer;
    logic [AW-1:0] read_pointer;

    logic [WIDTH-1:0] fifo [0:DEPTH-1];

    logic [AW-1:0] write_pointer_plus1;
    assign write_pointer_plus1 = write_pointer + {{(AW-1){1'b0}}, 1'b1};

    always_comb begin
        empty = (write_pointer == read_pointer);
        full = (write_pointer_plus1 == read_pointer);
        command_out = fifo[read_pointer];
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            write_pointer <= '0;
            read_pointer <= '0;
        end else begin
            if (write && !full) begin
                fifo[write_pointer] <= command_in;
                write_pointer <= write_pointer + {{(AW-1){1'b0}}, 1'b1};
            end

            if (read && !empty) begin
                read_pointer <= read_pointer + {{(AW-1){1'b0}}, 1'b1};
            end
        end
    end
 
endmodule
`default_nettype wire
