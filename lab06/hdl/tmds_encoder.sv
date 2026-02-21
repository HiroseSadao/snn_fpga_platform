`timescale 1ns/1ps
`default_nettype none

module tmds_encoder(
    input  wire        clk,
    input  wire        rst,
    input  wire [7:0]  video_data,
    input  wire [1:0]  control,
    input  wire        video_enable,
    output logic [9:0] tmds
);
    logic [8:0] q_m;

    tm_choice mtm(
        .d(video_data),
        .q_m(q_m)
    );

    logic signed [5:0] rd;
    wire [8:0] q_lo = q_m[8:0];

    logic [9:0] q_out;

    always_comb tmds = q_out;

    always_ff @(posedge clk) begin
        if (rst) begin
            q_out <= 10'b0;
            rd    <= 6'sd0;
        end else if (!video_enable) begin
            unique case (control)
                2'b00: q_out <= 10'b1101010100;
                2'b01: q_out <= 10'b0010101011;
                2'b10: q_out <= 10'b0101010100;
                2'b11: q_out <= 10'b1010101011;
            endcase
            rd <= 6'sd0;
        end else begin
            logic [3:0] ones8;
            logic inv;
            logic [9:0] out_next;
            logic signed [5:0] rd_next;

            ones8 = $countones(q_m[7:0]);

            if (rd == 0 || ones8 == 4) begin
                out_next[8]   = q_m[8];
                out_next[9]   = ~q_m[8];
                out_next[7:0] = q_m[8] ? q_m[7:0] : ~q_m[7:0];
                rd_next = rd + (q_m[8] ? ($signed($signed({2'b00,ones8}) - $signed(6'sd8 - $signed({2'b00,ones8}))))
                                    : ($signed($signed(6'sd8 - $signed({2'b00,ones8})) - $signed({2'b00,ones8}))));
            end else if ( (rd > 0 && ones8 > 4) || (rd < 0 && ones8 < 4) ) begin
                out_next[9]   = 1'b1;
                out_next[8]   = q_m[8];
                out_next[7:0] = ~q_m[7:0];
                rd_next = rd + $signed({4'b0, q_m[8],1'b0}) + ($signed($signed(6'sd8 - $signed({2'b00,ones8})) - $signed({2'b00,ones8})));
            end else begin
                out_next[9]   = 1'b0;
                out_next[8]   = q_m[8];
                out_next[7:0] =  q_m[7:0];
                rd_next = rd - $signed({4'b0, (~q_m[8]),1'b0}) + ($signed($signed({2'b00,ones8}) - $signed(6'sd8 - $signed({2'b00,ones8}))));
            end

            q_out <= out_next;
            rd    <= rd_next;
        end
    end

endmodule

`default_nettype wire
