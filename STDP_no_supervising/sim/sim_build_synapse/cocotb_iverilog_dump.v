module cocotb_iverilog_dump();
initial begin
    $dumpfile("C:/Users/kamer/Documents/brain_inspired_computer/tombo_snn/sim/sim_build_synapse/synapse.fst");
    $dumpvars(0, synapse);
end
endmodule
