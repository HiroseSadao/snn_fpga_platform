module cocotb_iverilog_dump();
initial begin
    $dumpfile("C:/Users/kamer/Documents/2025年度東大/MIT留学/留学後/digital_systems_laboratory/lab06/sim/sim_build/command_fifo.fst");
    $dumpvars(0, command_fifo);
end
endmodule
