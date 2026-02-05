`default_nettype none

module axis_gain #(
        parameter int TSTEP_W = 16,
        parameter int GAIN_FP = 65536 // 1.0 in S16.16
    )(
        input  wire              clk,
        input  wire              rst, // synchronous reset

        input  wire              s_tvalid,
        output logic             s_tready,
        input  wire [TSTEP_W+31:0] s_tdata, // {tstep_id, g_in}

        output logic             m_tvalid,
        input  wire              m_tready,
        output logic signed [TSTEP_W+31:0] m_tdata // {tstep_id, g_out}
    );

    localparam int FP_SHIFT = 16;

    typedef enum logic [0:0] {S_IDLE, S_OUT} state_e;
    state_e state;

    logic [TSTEP_W-1:0] tstep_id_reg;
    logic signed [31:0] g_in_reg;
    logic signed [63:0] mul_tmp;
    logic signed [31:0] g_out_reg;
    logic signed [63:0] mul_tmp_next;
    logic signed [31:0] g_out_next;

    always_comb begin
        mul_tmp = $signed(GAIN_FP) * $signed(g_in_reg);
        g_out_reg = $signed(mul_tmp >>> FP_SHIFT);
    end

    always_comb begin
        mul_tmp_next = $signed(GAIN_FP) * $signed(s_tdata[31:0]);
        g_out_next = $signed(mul_tmp_next >>> FP_SHIFT);
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state        <= S_IDLE;
            s_tready     <= 1'b1;
            m_tvalid     <= 1'b0;
            m_tdata      <= '0;
            tstep_id_reg <= '0;
            g_in_reg     <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    s_tready <= 1'b1;
                    if (s_tvalid && s_tready) begin
                        tstep_id_reg <= s_tdata[TSTEP_W+31 -: TSTEP_W];
                        g_in_reg     <= s_tdata[31:0];
                        m_tdata      <= {s_tdata[TSTEP_W+31 -: TSTEP_W], g_out_next};
                        m_tvalid     <= 1'b1;
                        s_tready     <= 1'b0;
                        state        <= S_OUT;
                    end
                end

                S_OUT: begin
                    if (m_tvalid && m_tready) begin
                        m_tvalid <= 1'b0;
                        state    <= S_IDLE;
                    end
                end
            endcase
        end
    end

endmodule

`default_nettype wire
