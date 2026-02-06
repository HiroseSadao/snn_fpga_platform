module tm_choice (
        input wire [7:0] d, //data byte in
        output logic [8:0] q_m //transition minimized output
    );
    integer i;

    always_comb begin
        if ($countones(d) > 4 || ($countones(d) == 4 && d[0] == 0)) begin
            q_m[0] = d[0];
            for (i = 1; i < 8; i++) begin
                q_m[i] = ~(d[i] ^ q_m[i-1]);
            end
            q_m[8] = 1'b0;
        end else begin
            q_m[0] = d[0];
            for (i = 1; i < 8; i++) begin
                q_m[i] = (d[i] ^ q_m[i-1]);
            end
            q_m[8] = 1'b1;
        end
    end
 
endmodule