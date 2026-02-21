`timescale 1ns / 1ps
`default_nettype none

/*
 * traffic_generator
 *
 * Module to provide the memory interface IP with the signals it needs,
 * decides what commands are issued in what order. Arbitrates between
 * write requests issued by write_axis, and read requests cycling through
 * the 720p frame buffer, whose responses are fed into read_axis.
 *
 * We've provided the state machine that manages the arbitration between
 * these requests, and the connections to each AXI-Stream. Your job is
 * to determine the address needed for each read and write request, likely
 * using some evt_counters!
 */

module traffic_generator(
        input wire           clk,      // should be ui clk of DDR3!
        input wire           rst,

        // UberDDR3 Control Signals
        output logic [23:0]  memrequest_addr,
        output logic         memrequest_en,
        output logic [127:0] memrequest_write_data,
        output logic         memrequest_write_enable,
        input wire [127:0]   memrequest_resp_data,
        input wire           memrequest_complete,
        input wire           memrequest_busy,
        // Write AXIS FIFO input
        input wire [127:0]   write_axis_data,
        input wire           write_axis_tlast,
        input wire           write_axis_valid,
        output logic         write_axis_ready,
        // Read AXIS FIFO output
        output logic [127:0] read_axis_data,
        output logic         read_axis_tlast,
        output logic         read_axis_valid,
        input wire           read_axis_af, // almost full signal
        input wire           read_axis_ready
);


    // state machine used to alternate between read & write requests
    typedef enum {
        RST,
      	WAIT_INIT,
      	RD_HDMI,
      	WR_CAM
    } tg_state;
    tg_state state;

    // Define ready/valid signals to output to our input+output AXI Streams!

    // give the write FIFO a "ready" signal when the MI is ready and our state machine
    // indicates it's the write AXIS' turn.

    assign write_axis_ready = !memrequest_busy && (state == WR_CAM);

    // Not an AXI-Stream, but the signals that define when we actually issue a read request.

    logic read_request_valid; // defined further below, based on state machine + address info
    assign read_request_valid = ~read_axis_af && state == RD_HDMI;
    logic read_request_ready;
    assign read_request_ready = !memrequest_busy && state == RD_HDMI;

    
    localparam MAX_ADDR = 115_199; // TODO change me
    
    // TODO: define the addresses associated with each read or write command+response!
    // you likely want to use an evt_counter that wraps at the right point, and increments
    // on the event of a valid/ready handshake on the proper signals!

    // for defining the write requests: your event should be a handshake on the write AXIStream,
    //     and the address should **reset** if a valid write axis transaction carries a TLAST !
    // for defining the read RESPONSES: your event should be a handshake on the read AXIStream
    // for defining the read REQUESTS: your event should be a "handshake" on the read requests

    logic [23:0] write_address;
    logic [23:0] read_request_address;

    evt_counter #(
        .MAX_COUNT(MAX_ADDR)
    ) u_wr_addr_ctr (
        .clk   (clk),
        .rst   (write_axis_tlast && write_axis_valid && write_axis_ready),
        .evt   (write_axis_valid && write_axis_ready),
        .count (write_address)
    );

    evt_counter #(
        .MAX_COUNT(MAX_ADDR)
    ) u_rd_addr_ctr (
        .clk   (clk),
        .rst   (rst),
        .evt   (read_request_valid && read_request_ready),
        .count (read_request_address)
    );
    
    // Your command FIFO, getting used to write down commands
    logic cf_full;
    logic cf_empty;
    logic [23:0] read_response_address;
    logic    queued_command_write_enable;

    // Feed the read output from the memory controller to the axi-stream output
    // the queued_command_write_enable won't be set until you've instantiated your FIFO below.
    
    // TODO: instantiate your command FIFO to write down an entry
    // whenever the memory controller receives a signal, and reads
    // an entry out whenever a command is completed.
    // Each entry should include the currently-issued command's
    // address and write-enable signals, and those output command
    // values should specify the read_response_address and
    // write enable signals.
    logic [24:0] cf_din;
    logic [24:0] cf_dout;
    assign cf_din  = {memrequest_write_enable, memrequest_addr};

    command_fifo #(
        .DEPTH(64),
        .WIDTH(25)
    ) mcf (
        .clk   (clk),
        .rst   (rst),
        .command_in   (cf_din),
        .write (memrequest_en),
        .read (memrequest_complete),
        .command_out  (cf_dout),
        .full  (cf_full),
        .empty (cf_empty)
    );

    assign read_response_address = cf_dout[23:0];
    assign queued_command_write_enable = cf_dout[24];
    assign read_axis_valid = memrequest_complete && !queued_command_write_enable;
    always_ff @(posedge clk) begin
        if(rst) begin
            read_axis_data <= 0;
        end else if(read_axis_valid && read_axis_ready) begin
            read_axis_data <= memrequest_resp_data;
        end else begin
            read_axis_data <= read_axis_data;
        end
    end

    
    // TODO: TLAST generation for the read output!
    // assign a tlast value based on the address your response is up to!

    assign read_axis_tlast = read_axis_valid && (read_response_address == MAX_ADDR); // change me!!


    // -----------------------------------
    // For lab 06, no need to change anything below here!

    
    // State Machine behavior
    
    // Signals for determining our conditions to switch between states:
    // By tying these signals to 1, we switch between issuing a read command
    // and a write command every clock cycle. The signals could be made
    // more complex in order to send commands in bursts, which could
    // increase throughput of data out of the memory chip.
    logic go_to_wr, go_to_rd, initialization_complete;
    assign go_to_wr = 1'b1;
    assign go_to_rd = 1'b1;
    assign initialization_complete = 1'b1;

    always_ff @(posedge clk) begin
        if(rst) begin
            state <= RST;
        end else begin
            case(state)
                RST: begin
                    state <= WAIT_INIT;
                end
                WAIT_INIT: begin
                    state <= initialization_complete ? RD_HDMI : WAIT_INIT;
                end
                RD_HDMI: begin
                    state <= go_to_wr ? WR_CAM : RD_HDMI;
                end
                WR_CAM: begin
                    state <= go_to_rd ? RD_HDMI : WR_CAM;
                end
            endcase // case (state)
        end
    end
    // communication signals to send to the controller in each state; assigned combinationally
    always_comb begin
        case(state)
            RST, WAIT_INIT: begin
                memrequest_addr = 0;
                memrequest_en = 0;
                memrequest_write_data = 0;
                memrequest_write_enable = 0;
            end
            WR_CAM: begin
                memrequest_addr = write_address;
                memrequest_en       = write_axis_valid && !memrequest_busy;
                memrequest_write_enable = write_axis_valid && !memrequest_busy;
                memrequest_write_data = write_axis_data;
            end
            RD_HDMI: begin
                memrequest_addr = read_request_address;
                memrequest_en = read_request_valid && !memrequest_busy;
                memrequest_write_enable = 1'b0;
                memrequest_write_data = 0;
            end
            default: begin
                memrequest_addr = 0;
                memrequest_en = 0;
                memrequest_write_data = 0;
                memrequest_write_enable = 0;
            end
        endcase // case (state)
    end // always_comb
endmodule

`default_nettype wire
