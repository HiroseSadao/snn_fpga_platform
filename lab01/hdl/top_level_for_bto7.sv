`default_nettype none // prevents system from inferring an undeclared logic (good practice)
 
module top_level(
        input wire [15:0] sw, //all 16 input slide switches
        input wire [3:0] btn, //all four momentary button switches
        output logic [15:0] led, //16 green output LEDs (located right above switches)
        output logic [2:0] rgb0, //RGB channels of RGB LED0
        output logic [2:0] rgb1, //RGB channels of RGB LED1
        output logic [3:0] ss0_an,//anode control for upper four digits of seven-seg display
        output logic [3:0] ss1_an,//anode control for lower four digits of seven-seg display
        output logic [6:0] ss0_c, //cathode controls for the segments of upper four digits
        output logic [6:0] ss1_c //cathod controls for the segments of lower four digits
    );
 
    logic [6:0] ss0_cn, ss1_cn; //used for inverting output signal
 
    // instantiate a bto7s module called 'n1' that controls the upper four 7-seg
    bto7s n1(
        .x(sw[7:4]),
        .s(ss0_cn)
    );
 
    // instantiate a bto7s module called 'n0' that controls the lower four 7-seg
    bto7s n0(
        .x(sw[3:0]),
        .s(ss1_cn)
    );
 
    assign {ss0_an, ss1_an} = 8'h00; // all low (on). to turn off digits, set high
    assign ss0_c = ~ss0_cn; //the cathodes on the seven segment are active low, so invert
    assign ss1_c = ~ss1_cn; //same as above
 
    /* we'll use the LEDs later...for now, just link them to the switches
    * and force some lights on
    */
 
    assign led = sw;
    assign rgb0 = 3'b0; //just shut up rgb0 (it is bright)
    assign rgb1 = 3'b0; //just shut up rgb1 (it is bright)
 
endmodule // top_level
/* I usually add a comment to associate my endmodule line with the module name
 * this helps when if you have multiple module definitions in a file
 */
 
// reset the default net type to wire, sometimes other code expects this.
`default_nettype wire