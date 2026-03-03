`default_nettype none


module  debouncer #(parameter CLK_PERIOD_NS = 10,
                    parameter DEBOUNCE_TIME_MS = 5
                    )
  ( input wire clk,
    input wire rst,
    input wire dirty,
    output logic clean);

    localparam int unsigned DEBOUNCE_CYCLES =
        (DEBOUNCE_TIME_MS*1_000_000 + CLK_PERIOD_NS - 1) / CLK_PERIOD_NS;
    localparam int unsigned COUNTER_SIZE =
        (DEBOUNCE_CYCLES <= 1) ? 1 : $clog2(DEBOUNCE_CYCLES);

    logic [COUNTER_SIZE-1:0] counter;
    logic old_dirty;

    always_ff @(posedge clk)begin
        old_dirty <= dirty;
        if (rst)begin
            counter <= 0;
            clean <= dirty;
        end else begin
            if ( dirty != old_dirty)begin
                counter <= 0;
            end else if ( counter < DEBOUNCE_CYCLES-1)begin
                counter <= counter + 1;
            end else begin
                clean <= dirty;
            end
        end
    end
endmodule


`default_nettype wire