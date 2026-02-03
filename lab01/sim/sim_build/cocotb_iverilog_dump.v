module cocotb_iverilog_dump();
initial begin
    $dumpfile("C:/Users/kamer/Documents/2025年度東大/MIT留学/留学後/digital_systems_laboratory/lab01/sim/sim_build/rgb_controller.fst");
    $dumpvars(0, rgb_controller);
end
endmodule
