module cocotb_iverilog_dump();
initial begin
    $dumpfile("C:/Users/kamer/Documents/2025年度東大/MIT留学/留学後/digital_systems_laboratory/lab03/sim/sim_build/uart_receive.fst");
    $dumpvars(0, uart_receive);
end
endmodule
