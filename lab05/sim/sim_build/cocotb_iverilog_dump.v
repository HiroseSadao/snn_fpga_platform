module cocotb_iverilog_dump();
initial begin
    $dumpfile("C:/Users/kamer/Documents/2025年度東大/MIT留学/留学後/digital_systems_laboratory/lab05/sim/sim_build/center_of_mass.fst");
    $dumpvars(0, center_of_mass);
end
endmodule
