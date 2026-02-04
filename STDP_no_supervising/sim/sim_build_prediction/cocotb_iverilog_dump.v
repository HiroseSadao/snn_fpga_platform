module cocotb_iverilog_dump();
initial begin
    $dumpfile("C:/Users/kamer/Documents/brain_inspired_computer/tombo_snn/STDP_no_supervising/sim/sim_build_prediction/prediction.fst");
    $dumpvars(0, prediction);
end
endmodule
