`default_nettype none

module axis_lif_inh #(
        parameter int TSTEP_W = 16
    )(
        input  wire              clk,
        input  wire              rst, // synchronous reset

        input  wire              s_tvalid,
        output logic             s_tready,
        input  wire [TSTEP_W+63:0] s_tdata, // {tstep_id, g_exc, g_inh}

        output logic             m_tvalid,
        input  wire              m_tready,
        output logic [TSTEP_W:0] m_tdata // {tstep_id, spike}
    );

    typedef enum logic [1:0] {L_IDLE, L_BUSY, L_OUT} state_e;
    state_e state;

    logic [TSTEP_W-1:0] tstep_id_reg;
    logic signed [31:0] g_exc_reg;
    logic signed [31:0] g_inh_reg;
    logic               tick_pulse;

    logic [9:0]  spike_count;
    logic        spike_pulse;
    logic        running;
    logic        done;
    logic        step_done;

    lif_inh u_lif(
        .clk         (clk),
        .start       (rst),
        .tick        (tick_pulse),
        .g_exc       (g_exc_reg),
        .g_inh       (g_inh_reg),
        .spike_count (spike_count),
        .spike_pulse (spike_pulse),
        .running     (running),
        .done        (done),
        .step_done   (step_done)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            state        <= L_IDLE;
            s_tready     <= 1'b1;
            m_tvalid     <= 1'b0;
            m_tdata      <= '0;
            tstep_id_reg <= '0;
            g_exc_reg    <= '0;
            g_inh_reg    <= '0;
            tick_pulse   <= 1'b0;
        end else begin
            tick_pulse <= 1'b0;

            case (state)
                L_IDLE: begin
                    s_tready <= 1'b1;
                    if (s_tvalid && s_tready) begin
                        tstep_id_reg <= s_tdata[TSTEP_W+63 -: TSTEP_W];
                        g_exc_reg    <= s_tdata[63:32];
                        g_inh_reg    <= s_tdata[31:0];
                        tick_pulse   <= 1'b1;
                        s_tready     <= 1'b0;
                        state        <= L_BUSY;
                    end
                end

                L_BUSY: begin
                    if (step_done) begin
                        m_tdata  <= {tstep_id_reg, spike_pulse};
                        m_tvalid <= 1'b1;
                        state    <= L_OUT;
                    end
                end

                L_OUT: begin
                    if (m_tvalid && m_tready) begin
                        m_tvalid <= 1'b0;
                        state    <= L_IDLE;
                    end
                end

                default: state <= L_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
