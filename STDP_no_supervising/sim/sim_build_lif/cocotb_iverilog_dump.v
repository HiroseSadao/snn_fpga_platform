module cocotb_iverilog_dump();
initial begin
    $dumpfile("C:/Users/kamer/Documents/brain_inspired_computer/tombo_snn/sim/sim_build_lif/lif.fst");
    $dumpvars(0, lif);
end
endmodule
