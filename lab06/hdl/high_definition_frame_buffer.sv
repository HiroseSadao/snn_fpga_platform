`default_nettype none
/*
 Top-level design, broken out to a different file
 for readability; this handles the entire pipeline 
 carrying pixel data in and out of the DRAM chip;
 including the UberDDR3 controller and all supporting
 components. Implements "the new way" data path from
 the block diagram figure.
 */
module high_definition_frame_buffer(

    // Input data from camera/pixel reconstructor
    input wire          clk_camera,
    input wire          sys_rst_camera,
    input wire          camera_valid,
    input wire [15:0]   camera_pixel,
    input wire [10:0]   camera_h_count,
    input wire [9:0]    camera_v_count,

    // Output data to HDMI display pipeline
    input wire          clk_pixel,
    input wire          sys_rst_pixel,
    input wire          active_draw_hdmi,
    input wire [10:0]   h_count_hdmi,
    input wire [9:0]   v_count_hdmi,
    
    output logic [15:0] frame_buff_dram,

    // Clock/reset signals for UberDDR controller
    input wire          clk_controller,
    input wire          clk_ddr3,
    input wire          clk_ddr3_90,
    input wire          i_ref_clk,
    input wire          i_rst,
    input wire          ddr3_clk_locked,
    
    // Bus wires to connect FPGA to SDRAM chip
    inout wire [15:0]   ddr3_dq,      //data input/output
    inout wire [1:0]    ddr3_dqs_n,   //data input/output differential strobe (negative)
    inout wire [1:0]    ddr3_dqs_p,   //data input/output differential strobe (positive)
    output wire [13:0]  ddr3_addr,    //address
    output wire [2:0]   ddr3_ba,      //bank address
    output wire         ddr3_ras_n,   //row active strobe
    output wire         ddr3_cas_n,   //column active strobe
    output wire         ddr3_we_n,    //write enable
    output wire         ddr3_reset_n, //reset (active low!!!)
    output wire         ddr3_clk_p,   //general differential clock (p)
    output wire         ddr3_clk_n,   //general differential clock (n)
    output wire         ddr3_clke,    //clock enable
    output wire [1:0]   ddr3_dm,      //data mask
    output wire         ddr3_odt      //on-die termination (helps impedance match)
);
    
    logic [127:0] camera_axis_tdata;
    logic         camera_axis_tlast;
    logic         camera_axis_tready;
    logic         camera_axis_tvalid;

    // takes our 16-bit values and deserialize/stack them into 128-bit messages to write to DRAM
    // the data pipeline is designed such that we can fairly safely assume its always ready.
    stacker stacker_inst(
        .clk(clk_camera),
        .rst(sys_rst_camera),
        .pixel_tvalid(camera_valid),
        .pixel_tready(),
        .pixel_tdata(camera_pixel),
        // TODO: define the tlast value! you can do it in one line, based on camera h_count/v_count values
        .pixel_tlast((camera_h_count == 1279 && camera_v_count == 719)?1:0), // CHANGE ME
        .chunk_tvalid(camera_axis_tvalid),
        .chunk_tready(camera_axis_tready),
        .chunk_tdata(camera_axis_tdata),
        .chunk_tlast(camera_axis_tlast)
    );

    logic [127:0] camera_memclk_axis_tdata;
    logic         camera_memclk_axis_tlast;
    logic         camera_memclk_axis_tready;
    logic         camera_memclk_axis_tvalid;
    logic         camera_memclk_axis_prog_empty;

    // FIFO data queue of 128-bit messages, crosses clock domains to the 83.333MHz
    // controller clock of the memory interface
    clockdomain_fifo camera_data_fifo(
        .sender_rst(sys_rst_camera),

        .sender_clk(clk_camera),
        .sender_axis_tvalid(camera_axis_tvalid),
        .sender_axis_tready(camera_axis_tready),
        .sender_axis_tdata(camera_axis_tdata),
        .sender_axis_tlast(camera_axis_tlast),

        .receiver_clk(clk_controller),
        .receiver_axis_tvalid(camera_memclk_axis_tvalid),
        .receiver_axis_tready(camera_memclk_axis_tready),
        .receiver_axis_tdata(camera_memclk_axis_tdata),
        .receiver_axis_tlast(camera_memclk_axis_tlast),
        .receiver_axis_prog_empty(camera_memclk_axis_prog_empty)
    );

    logic [127:0] display_memclk_axis_tdata;
    logic         display_memclk_axis_tlast;
    logic         display_memclk_axis_tready;
    logic         display_memclk_axis_tvalid;
    logic         display_memclk_axis_prog_full;

    // Input/Output signals to drive the UberDDR3 memory controller
    // the traffic generator uses a state machine and the FIFOs to determine these signals
    logic [23:0]  memrequest_addr;
    logic         memrequest_en;
    logic [127:0] memrequest_write_data;
    logic         memrequest_write_enable;
    logic [127:0] memrequest_resp_data;
    logic         memrequest_complete;
    logic	      memrequest_busy;
    logic         memrequest_rdy;


    // this traffic generator handles reads and writes issued to the MIG IP,
    // which in turn handles the bus to the DDR chip.
    traffic_generator traffic_generator_inst(
        .memrequest_addr         (memrequest_addr),
        .memrequest_en           (memrequest_en),
        .memrequest_write_data   (memrequest_write_data[127:0]),
        .memrequest_write_enable (memrequest_write_enable),

        .clk                     (clk_controller),
        .rst                     (i_rst),
        .memrequest_resp_data    (memrequest_resp_data[127:0]),
        .memrequest_busy         (memrequest_busy), 
        .memrequest_complete     (memrequest_complete), 

        .write_axis_data        (camera_memclk_axis_tdata),
        .write_axis_tlast       (camera_memclk_axis_tlast),
        .write_axis_valid       (camera_memclk_axis_tvalid),
        .write_axis_ready       (camera_memclk_axis_tready),


        .read_axis_data         (display_memclk_axis_tdata),
        .read_axis_tlast        (display_memclk_axis_tlast),
        .read_axis_valid        (display_memclk_axis_tvalid),
        .read_axis_af           (display_memclk_axis_prog_full),
        .read_axis_ready        (display_memclk_axis_tready) //,
    );

    ddr3_top #(
        .CONTROLLER_CLK_PERIOD(12_000), //ps, clock period of the controller interface
        .DDR3_CLK_PERIOD(3_000), //ps, clock period of the DDR3 RAM device (must be 1/4 of the CONTROLLER_CLK_PERIOD)
        .ROW_BITS(14), //width of row address
        .COL_BITS(10), //width of column address
        .BA_BITS(3), //width of bank address
        .BYTE_LANES(2), //number of DDR3 modules to be controlled
        .AUX_WIDTH(16), //width of aux line (must be >= 4)
        .WB2_ADDR_BITS(32), //width of 2nd wishbone address bus
        .WB2_DATA_BITS(32), //width of 2nd wishbone data bus
        .MICRON_SIM(0), //enable faster simulation for micron ddr3 model (shorten POWER_ON_RESET_HIGH and INITIAL_CKE_LOW)
        .ODELAY_SUPPORTED(0), //set to 1 when ODELAYE2 is supported
        .SECOND_WISHBONE(0), //set to 1 if 2nd wishbone is needed
        .ECC_ENABLE(0), // set to 1 or 2 to add ECC (1 = Side-band ECC per burst, 2 = Side-band ECC per 8 bursts , 3 = Inline ECC )
        .WB_ERROR(0) // set to 1 to support Wishbone error (asserts at ECC double bit error)
      ) ddr3_top
      (
        //clock and reset
        .i_controller_clk(clk_controller),
        .i_ddr3_clk(clk_ddr3), //i_controller_clk has period of CONTROLLER_CLK_PERIOD, i_ddr3_clk has period of DDR3_CLK_PERIOD
        .i_ref_clk(i_ref_clk),
        .i_ddr3_clk_90(clk_ddr3_90),
        .i_rst_n(!i_rst && ddr3_clk_locked),

        // Inputs
        .i_wb_cyc(1), //bus cycle active (1 = normal operation, 0 = all ongoing transaction are to be cancelled)
        .i_wb_stb(memrequest_en), //request a transfer
        .i_wb_we(memrequest_write_enable), //write-enable (1 = write, 0 = read)
        .i_wb_addr(memrequest_addr), //burst-addressable {row,bank,col}
        .i_wb_data(memrequest_write_data), //write data, for a 4:1 controller data width is 8 times the number of pins on the device
        .i_wb_sel(16'hffff), //byte strobe for write (1 = write the byte)
        .i_aux(memrequest_write_enable), //for AXI-interface compatibility (given upon strobe)

        // Outputs
        .o_wb_stall(memrequest_busy), //1 = busy, cannot accept requests
        .o_wb_ack(memrequest_complete), //1 = read/write request has completed
        .o_wb_err(), //1 = Error due to ECC double bit error (fixed to 0 if WB_ERROR = 0)
        .o_wb_data(memrequest_resp_data), //read data, for a 4:1 controller data width is 8 times the number of pins on the device
        .o_aux(),

        // DDR3 I/O Interface
        .o_ddr3_clk_p(ddr3_clk_p),
        .o_ddr3_clk_n(ddr3_clk_n),
        .o_ddr3_reset_n(ddr3_reset_n),
        .o_ddr3_cke(ddr3_clke), // CKE
        .o_ddr3_cs_n(), // chip select signal (controls rank 1 only) tied to 0 on this board by default.
        .o_ddr3_ras_n(ddr3_ras_n), // RAS#
        .o_ddr3_cas_n(ddr3_cas_n), // CAS#
        .o_ddr3_we_n(ddr3_we_n), // WE#
        .o_ddr3_addr(ddr3_addr),
        .o_ddr3_ba_addr(ddr3_ba),
        .io_ddr3_dq(ddr3_dq),
        .io_ddr3_dqs(ddr3_dqs_p),
        .io_ddr3_dqs_n(ddr3_dqs_n),
        .o_ddr3_dm(ddr3_dm),
        .o_ddr3_odt(ddr3_odt), // on-die termination
        .o_debug1()
        //.o_debug1(o_debug1)
      );


    logic [127:0] display_axis_tdata;
    logic         display_axis_tlast;
    logic         display_axis_tready;
    logic         display_axis_tvalid;
    logic         display_axis_prog_empty;

    clockdomain_fifo pdfifo(
        .sender_rst(i_rst),
        .sender_clk(clk_controller),
        .sender_axis_tvalid(display_memclk_axis_tvalid),
        .sender_axis_tready(display_memclk_axis_tready),
        .sender_axis_tdata(display_memclk_axis_tdata),
        .sender_axis_tlast(display_memclk_axis_tlast),
        .sender_axis_prog_full(display_memclk_axis_prog_full),
        .receiver_clk(clk_pixel),
        .receiver_axis_tvalid(display_axis_tvalid),
        .receiver_axis_tready(display_axis_tready),
        .receiver_axis_tdata(display_axis_tdata),
        .receiver_axis_tlast(display_axis_tlast),
        .receiver_axis_prog_empty(display_axis_prog_empty)
    );

    logic frame_buff_tvalid;
    logic frame_buff_tready;
    logic [15:0] frame_buff_tdata;
    logic        frame_buff_tlast;

    unstacker unstacker_inst(
        .clk(clk_pixel),
        .rst(sys_rst_pixel),
        .chunk_tvalid(display_axis_tvalid),
        .chunk_tready(display_axis_tready),
        .chunk_tdata(display_axis_tdata),
        .chunk_tlast(display_axis_tlast),
        .pixel_tvalid(frame_buff_tvalid),
        .pixel_tready(frame_buff_tready),
        .pixel_tdata(frame_buff_tdata),
        .pixel_tlast(frame_buff_tlast)
    );

    // TODO: assign frame_buff_tready
    // This should be done combinationally (either in one-line assign or an always_comb block)
    always_comb begin
        frame_buff_tready = active_draw_hdmi && ( !frame_buff_tlast || (h_count_hdmi == 1279 && v_count_hdmi == 719) ); // change me!!
    end


    assign frame_buff_dram = frame_buff_tvalid ? frame_buff_tdata : 16'h2277;

endmodule

`default_nettype wire
