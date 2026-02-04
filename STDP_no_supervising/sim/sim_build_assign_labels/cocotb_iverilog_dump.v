module cocotb_iverilog_dump();
initial begin
    $dumpfile("C:/Users/kamer/Documents/brain_inspired_computer/tombo_snn/STDP_no_supervising/sim/sim_build_assign_labels/assign_labels.fst");
    $dumpvars(0, assign_labels);
end
endmodule
