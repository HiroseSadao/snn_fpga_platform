`default_nettype none
module center_of_mass (
        input  wire        clk,
        input  wire        rst,
        input  wire [10:0] pixel_x,
        input  wire [9:0]  pixel_y,
        input  wire        pixel_valid,
        input  wire        calculate,
        output logic [10:0] com_x,
        output logic [9:0]  com_y,
        output logic        com_valid
    );

    logic [31:0] sum_x, sum_y;
    logic [31:0] pix_cnt;

    logic calculate_d;
    logic calculate_rise;
    assign calculate_rise = calculate & ~calculate_d;
    always_ff @(posedge clk) begin
        if (rst) begin
            calculate_d <= 1'b0;
        end else begin
            calculate_d <= calculate;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            sum_x <= 32'd0;
            sum_y <= 32'd0;
            pix_cnt <= 32'd0;
        end else if (calculate_rise) begin
            sum_x <= 32'd0;
            sum_y <= 32'd0;
            pix_cnt <= 32'd0;
        end else if (pixel_valid) begin
            sum_x <= sum_x + pixel_x;
            sum_y <= sum_y + pixel_y;
            pix_cnt <= pix_cnt + 32'd1;
        end
    end

    logic [31:0] lat_sum_x, lat_sum_y, lat_cnt;
    logic have_work;

    // always_ff @(posedge clk) begin
    //     if (rst) begin
    //         lat_sum_x <= 32'd0;
    //         lat_sum_y <= 32'd0;
    //         lat_cnt <= 32'd0;
    //         have_work <= 1'b0;
    //     end else if (calculate_rise) begin
    //         lat_sum_x <= sum_x;
    //         lat_sum_y <= sum_y;
    //         lat_cnt   <= pix_cnt;
    //         have_work <= (pix_cnt != 32'd0);
    //     end
    // end

    logic start_pending;
    logic fire_div;
    logic busy_x, busy_y;

    always_ff @(posedge clk) begin
        if (rst) begin
            start_pending <= 1'b0;
        end else begin
            if (calculate_rise && (pix_cnt != 32'd0))
                start_pending <= 1'b1;
            if (fire_div)
                start_pending <= 1'b0;
        end
    end

    assign fire_div = start_pending & ~busy_x & ~busy_y;

    logic [31:0] qx, qy;
    logic [31:0] rx_unused, ry_unused;
    logic vout_x, vout_y;
    logic err_x, err_y;

    divider #(.WIDTH(32)) div_x (
        .clk(clk),
        .rst(rst),
        .dividend(lat_sum_x),
        .divisor(lat_cnt),
        .data_in_valid(fire_div),
        .quotient(qx),
        .remainder(rx_unused),
        .data_out_valid(vout_x),
        .error(err_x),
        .busy(busy_x)
    );

    divider #(.WIDTH(32)) div_y (
        .clk(clk),
        .rst(rst),
        .dividend(lat_sum_y),
        .divisor(lat_cnt),
        .data_in_valid(fire_div),
        .quotient(qy),
        .remainder(ry_unused),
        .data_out_valid(vout_y),
        .error(err_y),
        .busy(busy_y)
    );

    logic got_x, got_y;
    logic [31:0] qx_latched, qy_latched;

    always_ff @(posedge clk) begin
        if (rst) begin
            got_x <= 1'b0;
            got_y <= 1'b0;
            qx_latched <= 32'd0;
            qy_latched <= 32'd0;
        end else begin
            if (calculate_rise && (pix_cnt != 32'd0)) begin
                got_x <= 1'b0;
                got_y <= 1'b0;
            end
            if (fire_div) begin
                got_x <= 1'b0;
                got_y <= 1'b0;
            end
            if (vout_x && ~err_x) begin
                qx_latched <= qx;
                got_x <= 1'b1;
            end
            if (vout_y && ~err_y) begin
                qy_latched <= qy;
                got_y <= 1'b1;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            com_x <= '0;
            com_y <= '0;
            com_valid <= 1'b0;
            lat_sum_x <= 32'd0;
            lat_sum_y <= 32'd0;
            lat_cnt <= 32'd0;
            have_work <= 1'b0;
        end else if (calculate_rise) begin
            lat_sum_x <= sum_x;
            lat_sum_y <= sum_y;
            lat_cnt   <= pix_cnt;
            have_work <= (pix_cnt != 32'd0);
        end else begin
            com_valid <= 1'b0;

            if (have_work && got_x && got_y) begin
                com_x <= qx_latched[10:0];
                com_y <= qy_latched[9:0];
                com_valid <= 1'b1;
                have_work <= 1'b0;
            end
        end
    end

endmodule
`default_nettype wire
