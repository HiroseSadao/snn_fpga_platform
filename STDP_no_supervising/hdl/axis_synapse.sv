`default_nettype none

module axis_synapse #(
        parameter int TSTEP_W = 16,
        parameter int TD_STEPS = 1
    )(
        input  wire              clk,
        input  wire              rst, // synchronous reset

        input  wire              s_tvalid,
        output logic             s_tready,
        input  wire [TSTEP_W:0]  s_tdata, // {tstep_id[TSTEP_W-1:0], spike}

        output logic             m_tvalid,
        input  wire              m_tready,
        output logic signed [TSTEP_W+31:0] m_tdata // {tstep_id, r_out}
    );

    localparam int FP_SHIFT = 16;
    localparam int FP_SCALE = (1 << FP_SHIFT);
    localparam int TD_HALF  = TD_STEPS / 2;
    localparam int SPIKE_ADD = (TD_STEPS == 0) ? 0 : (FP_SCALE / TD_STEPS);

    typedef enum logic [1:0] {S_IDLE, S_DIV_WAIT, S_OUT} state_e;
    state_e state;

    logic [TSTEP_W-1:0] tstep_id_reg;
    logic               spike_reg;
    logic signed [31:0] r_state;

    logic [31:0] div_dividend;
    logic [31:0] div_divisor;
    logic        div_valid_in;
    logic [31:0] div_quotient;
    logic [31:0] div_remainder;
    logic        div_valid_out;
    logic        div_error;
    logic        div_busy;
    logic signed [31:0] r_next;

    divider2b #(.WIDTH(32)) u_divider(
        .clk_in        (clk),
        .rst_in        (rst),
        .dividend_in   (div_dividend),
        .divisor_in    (div_divisor),
        .data_valid_in (div_valid_in),
        .quotient_out  (div_quotient),
        .remainder_out (div_remainder),
        .data_valid_out(div_valid_out),
        .error_out     (div_error),
        .busy_out      (div_busy)
    );

    always_comb begin
        if (spike_reg) begin
            r_next = $signed(r_state) - $signed(div_quotient) + $signed(SPIKE_ADD);
        end else begin
            r_next = $signed(r_state) - $signed(div_quotient);
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state        <= S_IDLE;
            s_tready     <= 1'b1;
            m_tvalid     <= 1'b0;
            m_tdata      <= '0;
            tstep_id_reg <= '0;
            spike_reg    <= 1'b0;
            r_state      <= '0;
            div_dividend <= '0;
            div_divisor  <= '0;
            div_valid_in <= 1'b0;
        end else begin
            div_valid_in <= 1'b0;

            case (state)
                S_IDLE: begin
                    s_tready <= 1'b1;
                    if (s_tvalid && s_tready) begin
                        tstep_id_reg <= s_tdata[TSTEP_W:1];
                        spike_reg    <= s_tdata[0];
                        div_dividend <= r_state[31:0] + TD_HALF[31:0];
                        div_divisor  <= TD_STEPS[31:0];
                        div_valid_in <= 1'b1;
                        s_tready     <= 1'b0;
                        state        <= S_DIV_WAIT;
                    end
                end

                S_DIV_WAIT: begin
                    if (div_valid_out) begin
                        r_state  <= r_next;
                        m_tdata  <= {tstep_id_reg, r_next};
                        m_tvalid <= 1'b1;
                        state    <= S_OUT;
                    end
                end

                S_OUT: begin
                    if (m_tvalid && m_tready) begin
                        m_tvalid <= 1'b0;
                        state    <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
