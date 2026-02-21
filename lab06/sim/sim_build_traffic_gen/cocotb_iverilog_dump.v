module cocotb_iverilog_dump();
initial begin
    $dumpfile("C:/Users/kamer/Documents/2025年度東大/MIT留学/留学後/digital_systems_laboratory/lab06/sim/sim_build_traffic_gen/traffic_generator.fst");
    $dumpvars(0, traffic_generator);
end
endmodule
