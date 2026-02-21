`timescale 1ns / 1ps
`default_nettype none

/*
 * stacker
 *
 * AXI-Stream (approximately) module that takes in serialized 16-bit messages
 * and stacks them together into 128-bit messages. Least-significant bytes
 * received first.
 */

module stacker(
        input wire           clk,
        input wire           rst,
        // input axis: 16 bit pixels
        input wire           pixel_tvalid,
        output logic         pixel_tready,
        input wire [15:0]    pixel_tdata,
        input wire           pixel_tlast,
        // output axis: 128 bit mig-phrases
        output logic         chunk_tvalid,
        input wire           chunk_tready,
        output logic [127:0] chunk_tdata,
        output logic         chunk_tlast
    );
    logic [127:0] data_recent;
    logic [2:0]   count;
    logic [7:0]   tlast_recent;

    logic         accept_in;
    assign accept_in = pixel_tvalid && pixel_tready;

    assign pixel_tready = (count == 7) ? chunk_tready : 1'b1;

    logic accept_out;
    assign accept_out = chunk_tready && chunk_tvalid;

    always_ff @(posedge clk) begin
        if(rst) begin
            data_recent  <= 127'b0;
            count        <= 0;
            tlast_recent <= 8'b0;
            chunk_tvalid <= 1'b0;
        end else begin
            if (accept_in) begin
                data_recent  <= { pixel_tdata[15:0], data_recent[127:16] };
                tlast_recent <= { pixel_tlast, tlast_recent[7:1] };
                count        <= count + 1;
                if (count == 7) begin
                    chunk_tdata  <= { pixel_tdata[15:0], data_recent[127:16] };
                    chunk_tlast <= (tlast_recent > 0);
                    chunk_tvalid <= 1'b1;
                end
            end
            if (accept_out) begin
                chunk_tvalid <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
