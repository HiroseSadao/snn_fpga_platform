`timescale 1ns / 1ps
`default_nettype none

module unstacker(
        input wire 	       clk,
        input wire 	       rst,
        // input axis: 128 bit phrases
        input wire 	       chunk_tvalid,
        output logic        chunk_tready,
        input wire [127:0]  chunk_tdata,
        input wire 	       chunk_tlast,
        // output axis: 16 bit words
        output logic        pixel_tvalid,
        input wire 	       pixel_tready,
        output logic [15:0] pixel_tdata,
        output logic        pixel_tlast
    );
    logic [2:0] offset;
    logic       accept_in;
    logic       accept_out;

    assign accept_in = chunk_tvalid && chunk_tready;
    assign accept_out = pixel_tvalid && pixel_tready;

    logic [127:0] shift_phrase;
    assign pixel_tdata = shift_phrase[15:0];

    logic tlast_hold;
    assign pixel_tlast = offset == 7 ? tlast_hold : 1'b0;

    logic need_phrase;

    assign chunk_tready = need_phrase || (offset == 7 && accept_out);
    assign pixel_tvalid = !need_phrase;

    always_ff @(posedge clk) begin
        if (rst) begin
            shift_phrase <= 128'b0;
            need_phrase  <= 1'b1;
            offset       <= 0;
            tlast_hold <= 1'b1;
        end else begin
            if (accept_out) begin
                offset <= offset+1;
            if (offset==7) begin
                if (chunk_tvalid) begin
                    shift_phrase <= chunk_tdata;
                    tlast_hold <= chunk_tlast;
                end else begin
                    need_phrase <= 1'b1;
                end
            end else begin
                shift_phrase <= {16'b0, shift_phrase[127:16]};
            end
        end else if (accept_in) begin
            need_phrase  <= 1'b0;
            shift_phrase <= chunk_tdata;
            tlast_hold <= chunk_tlast;
            offset       <= 0;
            end
        end
    end
endmodule

`default_nettype wire
