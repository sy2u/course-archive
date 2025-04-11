module cpu(
    input   logic           clk,
    input   logic           rst,
    output  logic   [31:0]  imem_addr,
    input   logic   [31:0]  imem_rdata,
    output  logic   [31:0]  dmem_addr,
    output  logic           dmem_write,
    output  logic   [3:0]   dmem_wmask,
    input   logic   [31:0]  dmem_rdata,
    output  logic   [31:0]  dmem_wdata
);

            logic   [31:0]  rs1_sel;
            logic   [31:0]  rs2_sel;
            logic   [31:0]  rd_sel;

            logic           alu_mux_1_sel;
            logic           alu_mux_2_sel;
            logic           alu_inv_rs2;
            logic           alu_cin;
            logic   [1:0]   alu_op;
            logic           shift_msb;
            logic           shift_dir;
            logic           cmp_mux_sel;
            logic           pc_mux_sel;
            logic   [2:0]   mem_mux_sel;
            logic   [2:0]   rd_mux_sel;

            logic           cmp_out;
            logic   [31:0]  imm;

            logic           cmp_lt;
            logic           cmp_eq;
            logic           cmp_a_31;
            logic           cmp_b_31;

            // inverted control signal
            logic   [31:0]  rs1_sel_inv;
            logic   [31:0]  rs2_sel_inv;
            logic   [31:0]  rd_sel_inv;

            logic           alu_mux_1_sel_inv;
            logic           alu_mux_2_sel_inv;
            logic           alu_cin_inv;
            logic   [1:0]   alu_op_inv;
            logic           shift_dir_inv;
            logic           cmp_mux_sel_inv;
            logic           pc_mux_sel_inv;
            logic   [2:0]   mem_mux_sel_inv;
            logic   [2:0]   rd_mux_sel_inv;
            
            logic   rst_inv;
            assign rst_inv = ~rst;

            // verification
            logic [31:0] real_rf_data[32];
            always_comb begin
                for (int i = 0; i < 32; i++) begin
                    real_rf_data[i][0] = datapath.rf_data_0[i];
                    real_rf_data[i][1] = datapath.rf_data_1[i];
                    real_rf_data[i][2] = datapath.rf_data_2[i];
                    real_rf_data[i][3] = datapath.rf_data_3[i];
                    real_rf_data[i][4] = datapath.rf_data_4[i];
                    real_rf_data[i][5] = datapath.rf_data_5[i];
                    real_rf_data[i][6] = datapath.rf_data_6[i];
                    real_rf_data[i][7] = datapath.rf_data_7[i];
                    real_rf_data[i][8] = datapath.rf_data_8[i];
                    real_rf_data[i][9] = datapath.rf_data_9[i];
                    real_rf_data[i][10] = datapath.rf_data_10[i];
                    real_rf_data[i][11] = datapath.rf_data_11[i];
                    real_rf_data[i][12] = datapath.rf_data_12[i];
                    real_rf_data[i][13] = datapath.rf_data_13[i];
                    real_rf_data[i][14] = datapath.rf_data_14[i];
                    real_rf_data[i][15] = datapath.rf_data_15[i];
                    real_rf_data[i][16] = datapath.rf_data_16[i];
                    real_rf_data[i][17] = datapath.rf_data_17[i];
                    real_rf_data[i][18] = datapath.rf_data_18[i];
                    real_rf_data[i][19] = datapath.rf_data_19[i];
                    real_rf_data[i][20] = datapath.rf_data_20[i];
                    real_rf_data[i][21] = datapath.rf_data_21[i];
                    real_rf_data[i][22] = datapath.rf_data_22[i];
                    real_rf_data[i][23] = datapath.rf_data_23[i];
                    real_rf_data[i][24] = datapath.rf_data_24[i];
                    real_rf_data[i][25] = datapath.rf_data_25[i];
                    real_rf_data[i][26] = datapath.rf_data_26[i];
                    real_rf_data[i][27] = datapath.rf_data_27[i];
                    real_rf_data[i][28] = datapath.rf_data_28[i];
                    real_rf_data[i][29] = datapath.rf_data_29[i];
                    real_rf_data[i][30] = datapath.rf_data_30[i];
                    real_rf_data[i][31] = datapath.rf_data_31[i];
                end 
            end
    control control(
        .*
    );

    datapath datapath(
        .*
    );

endmodule
