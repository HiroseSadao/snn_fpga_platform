`timescale 1ns / 1ps
`default_nettype none // prevents system from inferring an undeclared logic (good practice)

module video_mux (
        input wire [1:0] background_choice, //regular video
        input wire [1:0] target_choice, //regular video
        input wire [23:0] camera_pixel, //16 bits from camera 5:6:5
        input wire [7:0] camera_y_channel,  //y channel of ycrcb camera conversion
        input wire [7:0] selected_channel, //the channel from selection module
        input wire thresholded_pixel, //yes/no pixel is from threshold mask
        input wire [23:0] com_sprite_pixel,//input from center of mass sprite
        input wire [23:0] crosshair,
        output logic [23:0] muxed_pixel
    );

    /*
    00: normal camera out
    01: channel image (in grayscale)
    10: (thresholded channel image b/w)
    11: y channel with magenta mask

    upper bits:
    00: nothing:
    01: crosshair
    10: sprite on top
    11: nothing (orange test color)
    */

    logic [23:0] l_1;
    always_comb begin
        case (background_choice)
            2'b00: l_1 = camera_pixel;
            2'b01: l_1 = {selected_channel, selected_channel, selected_channel};
            2'b10: l_1 = (thresholded_pixel !=0)?24'hFFFFFF:24'h000000;
            2'b11: l_1 = (thresholded_pixel != 0) ? 24'hFF77AA : {camera_y_channel,camera_y_channel,camera_y_channel};
        endcase
    end

    logic [23:0] l_2;
    always_comb begin
        case (target_choice)
            2'b00: l_2 = l_1;
            2'b01: l_2 = crosshair > 0 ? crosshair : l_1;
            2'b10: l_2 = (com_sprite_pixel >0)?com_sprite_pixel:l_1;
            2'b11: l_2 = 24'hFF7700; //test color
        endcase
    end

    assign muxed_pixel = l_2;
endmodule

`default_nettype wire
