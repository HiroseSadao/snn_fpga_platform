`timescale 1ns / 1ps
`default_nettype none

module pixel_reconstruct
    #(
        parameter HCOUNT_WIDTH = 11,
        parameter VCOUNT_WIDTH = 10
    )
    (
     input wire                         clk,
     input wire                         rst,
     input wire                         camera_pclk,
     input wire                         camera_h_sync,
     input wire                         camera_v_sync,
     input wire [7:0]                   camera_data,
     output logic                       pixel_valid,
     output logic [HCOUNT_WIDTH-1:0]    pixel_h_count,
     output logic [VCOUNT_WIDTH-1:0]    pixel_v_count,
     output logic [15:0]                pixel_data
     );
    // your code here! and here's a handful of logics that you may find helpful to utilize.

    // previous value of PCLK
    logic  pclk_prev;
    logic sample_en;// 1 when we should sample camera_* signals
    // can be assigned combinationally:
    //  true when pclk transitions from 0 to 1
    logic camera_sample_valid;
    assign camera_sample_valid = sample_en; // TODO: fix this assign
    // previous value of camera data, from last valid sample!
    // should NOT update on every cycle of clk, only
    // when samples are valid.
    logic last_sampled_hs, last_sampled_vs;
    logic [7:0] last_sampled_data;
    // flag indicating whether the last byte has been transmitted or not.
    logic half_pixel_ready;
    logic [7:0] hi_byte;

    // always_ff @(posedge clk) begin
    //     if (rst) begin
    //         pclk_prev <= 1'b0;
    //     end else begin
    //         pclk_prev <= camera_pclk;
    //     end
    // end

    assign sample_en = (~pclk_prev) & camera_pclk;
    logic active;
    assign active = camera_h_sync & camera_v_sync;

    logic hs_now, vs_now;

    logic pending_row_reset;   // Flag to process line start on next pixel_valid after HS falling edge
    logic bump_h_on_next;      // If the previous pixel was emitted, increment h_count on the next pixel
    logic will_emit_pixel;

    always_ff @(posedge clk) begin
        if (rst) begin
            pclk_prev <= 1'b0;
            last_sampled_hs <= 1'b0;
            last_sampled_vs <= 1'b0;
            last_sampled_data <= 8'h00;

            half_pixel_ready <= 1'b0;
            hi_byte <= 8'h00;

            pixel_valid <= 1'b0;
            pixel_data <= 16'h0000;

            pixel_h_count <= '0;
            pixel_v_count <= '0;

            pending_row_reset <= 1'b0;
            bump_h_on_next <= 1'b0;
        end else begin
            pixel_valid <= 1'b0;
            pclk_prev <= camera_pclk;

            if (sample_en) begin
                hs_now = camera_h_sync;
                vs_now = camera_v_sync;

                if (~vs_now) begin
                    pixel_h_count <= '0;
                    pixel_v_count <= '0;
                    half_pixel_ready <= 1'b0;
                    pending_row_reset <= 1'b0;
                    bump_h_on_next <= 1'b0;
                end else begin
                    if (last_sampled_hs & ~hs_now) begin
                        pending_row_reset <= 1'b1;
                        half_pixel_ready <= 1'b0;
                    end
                end

                will_emit_pixel = active & half_pixel_ready;

                if (will_emit_pixel) begin
                    if (pending_row_reset) begin
                        pixel_h_count <= '0;
                        pixel_v_count <= pixel_v_count + 1'b1;
                        pending_row_reset <= 1'b0;
                    end else if (bump_h_on_next) begin
                        pixel_h_count <= pixel_h_count + 1'b1;
                    end
                    bump_h_on_next <= 1'b1;
                end

                if (active) begin
                    if (!half_pixel_ready) begin
                        hi_byte <= camera_data;
                        half_pixel_ready <= 1'b1;
                    end else begin
                        pixel_data <= {hi_byte, camera_data};
                        pixel_valid <= 1'b1;
                        half_pixel_ready <= 1'b0;
                    end
                end else begin
                    half_pixel_ready <= 1'b0;
                end

                last_sampled_hs <= hs_now;
                last_sampled_vs <= vs_now;
                last_sampled_data <= camera_data;
            end
        end
    end
endmodule

`default_nettype wire
