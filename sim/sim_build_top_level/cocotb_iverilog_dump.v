module cocotb_iverilog_dump();
initial begin
    $dumpfile("C:/Users/kamer/Documents/brain_inspired_computer/tombo_snn/sim/sim_build_top_level/top_level.fst");
    $dumpvars(0, top_level);
end
endmodule
