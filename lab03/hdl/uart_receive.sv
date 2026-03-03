`timescale 1ns / 1ps
`default_nettype none
 
module uart_receive
  #(
    parameter INPUT_CLOCK_FREQ = 100_000_000,
    parameter BAUD_RATE = 9600
    )
   (
    input wire 	       clk,
    input wire 	       rst,
    input wire 	       din,
    output logic       dout_valid,
    output logic [7:0] dout
    );
 
    typedef enum {
        IDLE = 0,
        // TODO: define the rest of the states your receiver needs to operate
        START = 1,
        DATA = 2,
        STOP = 3,
        TRANSMIT = 4
    } uart_state;

    localparam int BAUD_BIT_PERIOD = ((INPUT_CLOCK_FREQ + BAUD_RATE - 1) / BAUD_RATE) / 2;
    logic [$clog2(BAUD_BIT_PERIOD)-1:0] baud_bit_period_counter;
    logic [4:0] bit_counter;
    localparam int BIT_COUNT_MAX = 10;
 
    // note: for the online checker, don't rename this variable
    uart_state state;
 
    // TODO: module to read UART rx wire
    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            dout_valid <= 1'b0;
            dout <= '0;
            bit_counter <= '0;
            baud_bit_period_counter <= '0;
        end else begin
            if (state != IDLE) begin
                if (baud_bit_period_counter == BAUD_BIT_PERIOD-1) begin
                    bit_counter <= bit_counter + 1;
                    baud_bit_period_counter <= '0;
                    if (bit_counter[0] == 1'b0) begin
                        case (state)
                            START: begin
                                if (din == 1'b0) begin
                                    state <= DATA;
                                end else begin
                                    state <= IDLE;
                                    dout_valid <= 1'b0;
                                    dout <= '0;
                                    bit_counter <= '0;
                                    baud_bit_period_counter <= '0;
                                end
                            end
                            DATA: begin
                                dout <= {din, dout[7:1]};
                                if (bit_counter == 16) begin
                                    state <= STOP;
                                end
                            end
                            STOP: begin
                                if (din == 1'b1) begin
                                    state <= TRANSMIT;
                                end else begin
                                    state <= IDLE;
                                    dout_valid <= 1'b0;
                                    dout <= '0;
                                    bit_counter <= '0;
                                    baud_bit_period_counter <= '0;
                                end
                            end
                        endcase
                    end
                end else begin
                    baud_bit_period_counter <= baud_bit_period_counter + 1;
                    if (state == TRANSMIT) begin
                        state <= IDLE;
                        dout_valid <= 1'b1;
                    end
                end
            end else begin
                baud_bit_period_counter <= '0;
                bit_counter <= '0;
                dout_valid <= 1'b0;
                dout <= '0;
                if (din == 1'b0) begin
                    state <= START;
                end
            end
        end
    end
 
endmodule // uart_receive
 
`default_nettype wire