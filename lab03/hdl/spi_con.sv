`timescale 1ns / 1ps
`default_nettype none

module spi_con
  #(parameter int DATA_WIDTH = 8,
    parameter int DATA_CLK_PERIOD = 100)
(
    input  wire                    clk,
    input  wire                    rst,
    input  wire [DATA_WIDTH-1:0]   data_in,
    input  wire                    trigger,
    output logic [DATA_WIDTH-1:0]  data_out,
    output logic                   data_valid,
    output logic                   copi,
    input  wire                    cipo,
    output logic                   dclk,
    output logic                   cs
);

  logic trigger_q;
  wire  trigger_rise = trigger & ~trigger_q;

  localparam int P_EFF = (DATA_CLK_PERIOD < 2) ? 2 : DATA_CLK_PERIOD;
  localparam int P_EVEN = (P_EFF & 1) ? (P_EFF - 1) : P_EFF;
  localparam int HPER = P_EVEN/2;
  localparam int DIVW = (HPER <= 1) ? 1 : $clog2(HPER);

  logic [DIVW-1:0] divcnt;

  logic [DATA_WIDTH-1:0] tx_reg;
  logic [$clog2(DATA_WIDTH+1)-1:0] tx_idx;
  logic [DATA_WIDTH-1:0] rx_reg;
  logic [$clog2(DATA_WIDTH+1)-1:0] bit_cnt;

  logic busy;
  logic finishing;

  always_ff @(posedge clk) begin
    if (rst) begin
      cs <= 1'b1;
      dclk <= 1'b0;
      data_valid <= 1'b0;
      data_out <= '0;
      copi <= 1'b0;

      trigger_q <= 1'b0;
      divcnt <= '0;

      tx_reg <= '0;
      tx_idx <= '0;

      rx_reg <= '0;
      bit_cnt <= '0;
      busy <= 1'b0;
      finishing <= 1'b0;

    end else begin
      trigger_q <= trigger;
      data_valid <= 1'b0;

      if (!busy && trigger_rise) begin
        busy <= 1'b1;
        cs <= 1'b0;
        dclk <= 1'b0;
        divcnt <= '0;

        tx_reg <= data_in;
        tx_idx <= 1;
        copi <= data_in[DATA_WIDTH-1];

        rx_reg <= '0;

        bit_cnt <= '0;
        finishing <= 1'b0;
      end

      if (cs == 1'b0) begin
        if (divcnt == HPER-1) begin
          divcnt <= '0;

          if (dclk == 1'b0) begin
            dclk <= 1'b1;

            rx_reg[DATA_WIDTH-1 - bit_cnt] <= cipo;
            bit_cnt <= bit_cnt + 1'b1;

            if (bit_cnt + 1'b1 == DATA_WIDTH) begin
              finishing <= 1'b1;
            end

          end else begin
            dclk <= 1'b0;

            if (finishing) begin
              data_out <= rx_reg;
              data_valid <= 1'b1;
              cs <= 1'b1;
              busy <= 1'b0;
              finishing <= 1'b0;
              copi <= 1'b0;
            end else begin
              if (bit_cnt < DATA_WIDTH) begin
                copi <= tx_reg[DATA_WIDTH-1 - bit_cnt];
              end else begin
                copi <= 1'b0;
              end
            end
          end

        end else begin
          divcnt <= divcnt + 1'b1;
        end
      end else begin
        dclk   <= 1'b0;
        divcnt <= '0;
      end
    end
  end

endmodule

`default_nettype wire
