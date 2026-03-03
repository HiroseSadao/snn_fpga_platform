`timescale 1ns / 1ps
`default_nettype none

module uart_transmit
    #(
      parameter INPUT_CLOCK_FREQ = 100_000_000,
      parameter BAUD_RATE = 9600
      )
      (
      input wire 	     clk,
      input wire 	     rst,
      input wire [7:0] din,
      input wire 	     trigger,
      output logic     busy,
      output logic     dout
      );

    // TODO: module to transmit on UART
    logic trigger_q;
    wire trigger_rise = trigger & ~trigger_q;
    
    logic [7:0] shifter;
    logic [3:0] count;
    
    localparam int BAUD_BIT_PERIOD = (INPUT_CLOCK_FREQ + BAUD_RATE - 1) / BAUD_RATE;
    // localparam BAUD_BIT_PERIOD_WIDTH = $clog2(BAUD_BIT_PERIOD);
    logic [BAUD_BIT_PERIOD-1:0] baud_bit_period_counter;

    always_ff @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            dout <= 1'b1;
            count <= '0;
            trigger_q <= 1'b0;
            baud_bit_period_counter <= '0;
            shifter <= '0;
        end else begin
            trigger_q <= trigger;

            if (!busy && trigger_rise) begin
                baud_bit_period_counter <= '0;
                busy <= 1'b1;
                shifter <= din;
                count <= 4'b0;
                dout <= 1'b0;
            end

            if (busy) begin
                if (baud_bit_period_counter == BAUD_BIT_PERIOD-1) begin
                    if (count < 4'b1000) begin
                        baud_bit_period_counter <= '0;
                        dout <= shifter[0];
                        shifter <= shifter >> 1;
                        count <= count + 1;
                    end else if (count == 4'b1000) begin
                        baud_bit_period_counter <= '0;
                        dout <= 1'b1;
                        count <= count + 1;
                    end else begin
                        baud_bit_period_counter <= '0;
                        busy <= 1'b0;
                        dout <= 1'b1;
                        count <= 4'b0;
                    end
                end else begin
                    baud_bit_period_counter <= baud_bit_period_counter + 1;
                end                
            end
        end
    end

endmodule // uart_transmit

`default_nettype wire
