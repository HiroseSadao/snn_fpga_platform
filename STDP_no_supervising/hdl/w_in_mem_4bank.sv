`default_nettype none

module w_in_mem_4bank #(
    parameter int N_IN = 784,
    parameter int N_NEURONS = 100,
    parameter logic signed [31:0] INIT_VAL = 32'sd66,
    parameter bit INIT_FROM_FILE = 0
) (
    input  wire                         clk,
    input  wire                         rst,

    input  wire                         r_en,
    input  wire [$clog2(N_NEURONS)-1:0] r_neuron0,
    input  wire [$clog2(N_IN)-1:0]      r_in0,
    output logic signed [31:0]          r_data0,

    input  wire                         w_en0,
    input  wire [$clog2(N_NEURONS)-1:0] w_neuron0,
    input  wire [$clog2(N_IN)-1:0]      w_in0,
    input  wire signed [31:0]           w_data0,

    input  wire                         dbg_en,
    input  wire [$clog2(N_NEURONS)-1:0] dbg_neuron,
    input  wire [$clog2(N_IN)-1:0]      dbg_in,
    output logic                        dbg_valid,
    output logic signed [31:0]          dbg_data,

    output logic                        init_done
);

    localparam int DEPTH = N_NEURONS * N_IN;
    localparam int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH);

    localparam string MEM_INIT_FILE = "data/w_init0.mem";

`ifndef SYNTHESIS
    logic dbg_pending;
    wire  [ADDR_W-1:0] dbg_addr_i = addr_from(dbg_neuron, dbg_in);
`endif

    function automatic [ADDR_W-1:0] addr_from;
        input int neuron;
        input int in_idx;
        int row;
        begin
            row = neuron;
            addr_from = row * N_IN + in_idx;
        end
    endfunction

    logic init_active;
    logic [ADDR_W-1:0] init_addr;
    logic [ADDR_W-1:0] rd_addr;
    logic rd_en;
    logic [ADDR_W-1:0] wr_addr;
    logic wr_en;
    logic signed [31:0] wr_data;
    logic signed [31:0] rd_data;

    // Control/initialization FSM
    always_ff @(posedge clk) begin
        if (rst) begin
            dbg_valid <= 1'b0;
            dbg_data <= INIT_VAL;
`ifndef SYNTHESIS
            dbg_pending <= 1'b0;
`endif
            init_active <= !INIT_FROM_FILE;
            init_addr <= '0;
            init_done <= INIT_FROM_FILE;
        end else begin
            if (init_active) begin
                dbg_valid <= 1'b0;
                dbg_data <= INIT_VAL;
`ifndef SYNTHESIS
                dbg_pending <= 1'b0;
`endif

                if (init_addr == DEPTH-1) begin
                    init_active <= 1'b0;
                    init_done <= 1'b1;
                end else begin
                    init_addr <= init_addr + 1'b1;
                end
            end else begin
                dbg_valid <= 1'b0;
`ifndef SYNTHESIS
                if (dbg_pending) begin
                    dbg_data <= rd_data;
                    dbg_valid <= 1'b1;
                end

                dbg_pending <= dbg_en;
`else
                dbg_data <= '0;
                dbg_valid <= 1'b0;
`endif
            end
        end
    end

    // Read port (sync). Kept separate from write for BRAM inference.
    always_comb begin
        rd_en = r_en;
        rd_addr = addr_from(r_neuron0, r_in0);
`ifndef SYNTHESIS
        if (dbg_en) begin
            rd_en = 1'b1;
            rd_addr = dbg_addr_i;
        end
`endif
    end

    // Write port (sync). Kept separate from read for BRAM inference.
    always_comb begin
        if (init_active) begin
            wr_en = 1'b1;
            wr_addr = init_addr;
            wr_data = INIT_VAL;
        end else begin
            wr_en = w_en0;
            wr_addr = addr_from(w_neuron0, w_in0);
            wr_data = w_data0;
        end
    end

`ifdef SYNTHESIS
    // XPM SDPRAM (1R1W) for BRAM inference in synthesis
    xpm_memory_sdpram #(
        .ADDR_WIDTH_A(ADDR_W),
        .ADDR_WIDTH_B(ADDR_W),
        .AUTO_SLEEP_TIME(0),
        .BYTE_WRITE_WIDTH_A(32),
        .CLOCKING_MODE("common_clock"),
        .ECC_MODE("no_ecc"),
        .MEMORY_INIT_FILE(MEM_INIT_FILE),
        .MEMORY_INIT_PARAM("0"),
        .MEMORY_OPTIMIZATION("true"),
        .MEMORY_PRIMITIVE("block"),
        .MEMORY_SIZE(DEPTH * 32),
        .MESSAGE_CONTROL(0),
        .READ_DATA_WIDTH_B(32),
        .READ_LATENCY_B(1),
        .READ_RESET_VALUE_B("00000000"),
        .RST_MODE_A("SYNC"),
        .RST_MODE_B("SYNC"),
        .SIM_ASSERT_CHK(0),
        .USE_MEM_INIT(INIT_FROM_FILE),
        .WAKEUP_TIME("disable_sleep"),
        .WRITE_DATA_WIDTH_A(32),
        .WRITE_MODE_B("read_first")
    ) u_wmem_xpm (
        .clka(clk),
        .ena(wr_en),
        .wea(wr_en),
        .addra(wr_addr),
        .dina(wr_data),
        .clkb(clk),
        .enb(rd_en),
        .addrb(rd_addr),
        .doutb(rd_data),
        .rstb(rst),
        .regceb(1'b1),
        .sleep(1'b0),
        .injectsbiterra(1'b0),
        .injectdbiterra(1'b0)
    );
`else
    // Behavioral model for simulation
    logic signed [31:0] mem0_sim [0:DEPTH-1];

    initial begin
        if (INIT_FROM_FILE) begin
            $readmemh("data/w_init0.mem", mem0_sim);
        end
    end

    always_ff @(posedge clk) begin
        if (wr_en) begin
            mem0_sim[wr_addr] <= wr_data;
        end
    end

    always_ff @(posedge clk) begin
        if (rd_en) begin
            rd_data <= mem0_sim[rd_addr];
        end
    end
`endif

    assign r_data0 = rd_data;

endmodule

`default_nettype wire
