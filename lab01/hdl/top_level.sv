`default_nettype none // prevents system from inferring an undeclared logic (good practice)
 
module top_level(
    input  wire        clk_100mhz,
    input  wire [3:0]  btn,
    input  wire [15:0] sw,
    output logic [2:0] rgb0,
    output logic [2:0] rgb1,
    output logic [2:0] pmoda
);

    assign rgb1 = 3'b000;

    wire [7:0] r_in = {sw[15:11], 3'b000};
    wire [7:0] g_in = {sw[10:5],  2'b00};
    wire [7:0] b_in = {sw[4:0],   3'b000};

    rgb_controller u_rgb_controller (
        .clk   (clk_100mhz),
        .rst   (btn[0]),
        .r_in  (r_in),
        .g_in  (g_in),
        .b_in  (b_in),
        .r_out (rgb0[0]),
        .g_out (rgb0[1]),
        .b_out (rgb0[2])
    );

    assign pmoda = {rgb0[0], rgb0[1], rgb0[2]};

endmodule // top_level
/* I usually add a comment to associate my endmodule line with the module name
 * this helps when if you have multiple module definitions in a file
 */
 
// reset the default net type to wire, sometimes other code expects this.
`default_nettype wire