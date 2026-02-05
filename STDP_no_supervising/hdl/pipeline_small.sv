`default_nettype none

module pipeline_small #(
        parameter int TSTEP_W = 16
    )(
        input  wire              clk,
        input  wire              rst, // synchronous reset

        input  wire              s_tvalid,
        output wire              s_tready,
        input  wire [TSTEP_W:0]  s_tdata, // {tstep_id, spike}

        output wire              m_tvalid,
        input  wire              m_tready,
        output wire [TSTEP_W:0]  m_tdata  // {tstep_id, spike}
    );

    wire              syn_tvalid;
    wire              syn_tready;
    wire signed [TSTEP_W+31:0] syn_tdata; // {tstep_id, g_exc}

    wire              lif_tvalid;
    wire              lif_tready;
    wire [TSTEP_W+63:0] lif_tdata; // {tstep_id, g_exc, g_inh}

    axis_synapse #(
        .TSTEP_W (TSTEP_W),
        .TD_STEPS(1)
    ) u_synapse (
        .clk     (clk),
        .rst     (rst),
        .s_tvalid(s_tvalid),
        .s_tready(s_tready),
        .s_tdata (s_tdata),
        .m_tvalid(syn_tvalid),
        .m_tready(syn_tready),
        .m_tdata (syn_tdata)
    );

    assign lif_tdata = {syn_tdata[TSTEP_W+31 -: TSTEP_W], syn_tdata[31:0], 32'sd0};
    assign lif_tvalid = syn_tvalid;
    assign syn_tready = lif_tready;

    axis_lif #(
        .TSTEP_W(TSTEP_W)
    ) u_lif (
        .clk     (clk),
        .rst     (rst),
        .s_tvalid(lif_tvalid),
        .s_tready(lif_tready),
        .s_tdata (lif_tdata),
        .m_tvalid(m_tvalid),
        .m_tready(m_tready),
        .m_tdata (m_tdata)
    );

endmodule

`default_nettype wire
