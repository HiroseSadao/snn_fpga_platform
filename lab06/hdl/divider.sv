`timescale 1ns / 1ps
`default_nettype none

module divider #(parameter WIDTH = 32)
    (   input wire clk,
        input wire rst,
        input wire[WIDTH-1:0] dividend,
        input wire[WIDTH-1:0] divisor,
        input wire data_in_valid,
        output logic[WIDTH-1:0] quotient,
        output logic[WIDTH-1:0] remainder,
        output logic data_out_valid,
        output logic error,
        output logic busy
    );

    logic [WIDTH-1:0] quotient_g;
    logic [WIDTH-1:0] dividend_h;
    logic [WIDTH-1:0] divisor_h;

    enum {RESTING, DIVIDING} state;

    always_ff @(posedge clk)begin
        if (rst)begin
            quotient_g <= 0;
            dividend_h <= 0;
            divisor_h <= 0;
            remainder <= 0;
            busy <= 1'b0;
            error <= 1'b0;
            state <= RESTING;
            data_out_valid <= 1'b0;
        end else begin
            case (state)
                RESTING: begin
                    if (data_in_valid)begin
                        state <= DIVIDING;
                        quotient_g <= 0;
                        dividend_h <= dividend;
                        divisor_h <= divisor;
                        busy <= 1'b1;
                        error <= 1'b0;
                    end
                    data_out_valid <= 1'b0;
                end
                DIVIDING: begin
                    if (dividend_h<=0)begin
                        state <= RESTING; //similar to return statement
                        remainder <= dividend_h;
                        quotient <= quotient_g;
                        busy <= 1'b0; //tell outside world i'm done
                        error <= 1'b0;
                        data_out_valid <= 1'b1; //good stuff!
                    end else if (divisor_h==0)begin
                        state <= RESTING;
                        remainder <= 0;
                        quotient <= 0;
                        busy <= 1'b0; //tell outside world i'm done
                        error <= 1'b1; //ERROR
                        data_out_valid <= 1'b1; //valid ERROR
                    end else if (dividend_h < divisor_h) begin
                        state <= RESTING;
                        remainder <= dividend_h;
                        quotient <= quotient_g;
                        busy <= 1'b0;
                        error <= 1'b0;
                        data_out_valid <= 1'b1; //good stuff!
                    end else begin
                        //continuing onwards.
                        state <= DIVIDING;
                        quotient_g <= quotient_g + 1'b1;
                        dividend_h <= dividend_h-divisor_h;
                        busy <= 1'b1;
                        error <= 1'b0;
                    end
                end
            endcase
        end
    end
endmodule

`default_nettype wire
