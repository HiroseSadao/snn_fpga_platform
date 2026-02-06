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
    input  wire [$clog2(N_NEURONS)-1:0] r_neuron1,
    input  wire [$clog2(N_NEURONS)-1:0] r_neuron2,
    input  wire [$clog2(N_NEURONS)-1:0] r_neuron3,
    input  wire [$clog2(N_IN)-1:0]      r_in0,
    input  wire [$clog2(N_IN)-1:0]      r_in1,
    input  wire [$clog2(N_IN)-1:0]      r_in2,
    input  wire [$clog2(N_IN)-1:0]      r_in3,
    output logic signed [31:0]          r_data0,
    output logic signed [31:0]          r_data1,
    output logic signed [31:0]          r_data2,
    output logic signed [31:0]          r_data3,

    input  wire                         w_en0,
    input  wire                         w_en1,
    input  wire                         w_en2,
    input  wire                         w_en3,
    input  wire [$clog2(N_NEURONS)-1:0] w_neuron0,
    input  wire [$clog2(N_NEURONS)-1:0] w_neuron1,
    input  wire [$clog2(N_NEURONS)-1:0] w_neuron2,
    input  wire [$clog2(N_NEURONS)-1:0] w_neuron3,
    input  wire [$clog2(N_IN)-1:0]      w_in0,
    input  wire [$clog2(N_IN)-1:0]      w_in1,
    input  wire [$clog2(N_IN)-1:0]      w_in2,
    input  wire [$clog2(N_IN)-1:0]      w_in3,
    input  wire signed [31:0]           w_data0,
    input  wire signed [31:0]           w_data1,
    input  wire signed [31:0]           w_data2,
    input  wire signed [31:0]           w_data3,

    input  wire                         dbg_en,
    input  wire [$clog2(N_NEURONS)-1:0] dbg_neuron,
    input  wire [$clog2(N_IN)-1:0]      dbg_in,
    output logic                        dbg_valid,
    output logic signed [31:0]          dbg_data
);

    localparam int LANES = 4;
    localparam int NEURON_GROUPS = (N_NEURONS + LANES - 1) / LANES;
    localparam int DEPTH = NEURON_GROUPS * N_IN;
    localparam int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH);

    (* ram_style = "block" *) logic signed [31:0] mem0 [0:DEPTH-1];
    (* ram_style = "block" *) logic signed [31:0] mem1 [0:DEPTH-1];
    (* ram_style = "block" *) logic signed [31:0] mem2 [0:DEPTH-1];
    (* ram_style = "block" *) logic signed [31:0] mem3 [0:DEPTH-1];

    initial begin
        if (INIT_FROM_FILE) begin
            $readmemh("w_init0.hex", mem0);
            $readmemh("w_init1.hex", mem1);
            $readmemh("w_init2.hex", mem2);
            $readmemh("w_init3.hex", mem3);
        end
    end

    logic dbg_pending;
    logic [1:0] dbg_bank;
    logic [ADDR_W-1:0] dbg_addr;

    function automatic [ADDR_W-1:0] addr_from;
        input int neuron;
        input int in_idx;
        int row;
        begin
            row = neuron >> 2; // divide by 4
            addr_from = row * N_IN + in_idx;
        end
    endfunction

    integer i;

    always_ff @(posedge clk) begin
        if (rst) begin
            if (!INIT_FROM_FILE) begin
                for (i = 0; i < DEPTH; i = i + 1) begin
                    mem0[i] <= INIT_VAL;
                    mem1[i] <= INIT_VAL;
                    mem2[i] <= INIT_VAL;
                    mem3[i] <= INIT_VAL;
                end
            end
            r_data0 <= INIT_VAL;
            r_data1 <= INIT_VAL;
            r_data2 <= INIT_VAL;
            r_data3 <= INIT_VAL;
            dbg_valid <= 1'b0;
            dbg_data <= INIT_VAL;
            dbg_pending <= 1'b0;
            dbg_bank <= 2'b0;
            dbg_addr <= '0;
        end else begin
            if (r_en) begin
                r_data0 <= mem0[addr_from(r_neuron0, r_in0)];
                r_data1 <= mem1[addr_from(r_neuron1, r_in1)];
                r_data2 <= mem2[addr_from(r_neuron2, r_in2)];
                r_data3 <= mem3[addr_from(r_neuron3, r_in3)];
            end

            if (w_en0) mem0[addr_from(w_neuron0, w_in0)] <= w_data0;
            if (w_en1) mem1[addr_from(w_neuron1, w_in1)] <= w_data1;
            if (w_en2) mem2[addr_from(w_neuron2, w_in2)] <= w_data2;
            if (w_en3) mem3[addr_from(w_neuron3, w_in3)] <= w_data3;

            dbg_valid <= 1'b0;
            if (dbg_pending) begin
                case (dbg_bank)
                    2'd0: dbg_data <= mem0[dbg_addr];
                    2'd1: dbg_data <= mem1[dbg_addr];
                    2'd2: dbg_data <= mem2[dbg_addr];
                    2'd3: dbg_data <= mem3[dbg_addr];
                    default: dbg_data <= INIT_VAL;
                endcase
                dbg_valid <= 1'b1;
            end

            dbg_pending <= dbg_en;
            if (dbg_en) begin
                dbg_bank <= dbg_neuron[1:0];
                dbg_addr <= addr_from(dbg_neuron, dbg_in);
            end
        end
    end

endmodule

`default_nettype wire
