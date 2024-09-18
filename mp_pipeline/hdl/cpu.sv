module cpu
import rv32i_types::*;
(
    input   logic           clk,
    input   logic           rst,

    output  logic   [31:0]  imem_addr,
    output  logic   [3:0]   imem_rmask,
    input   logic   [31:0]  imem_rdata,
    input   logic           imem_resp,

    output  logic   [31:0]  dmem_addr,
    output  logic   [3:0]   dmem_rmask,
    output  logic   [3:0]   dmem_wmask,
    input   logic   [31:0]  dmem_rdata,
    output  logic   [31:0]  dmem_wdata,
    input   logic           dmem_resp
);

    if_id_stage_reg_t   if_id_reg;
    id_ex_stage_reg_t   id_ex_reg;
    ex_mem_stage_reg_t  ex_mem_reg;
    mem_wb_stage_reg_t  mem_wb_reg;

    logic           regf_we,
    logic   [31:0]  rd_v,
    logic   [4:0]   rs1_s, rs2_s, rd_sel,
    logic   [31:0]  rs1_v, rs2_v

    IF  stage_if( .clk(clk), .rst(rst), .imem_addr(imem_addr), .imem_rmask(imem_rmask), .if_id_reg(if_id_reg) );

    ID  stage_id( .clk(clk), .rst(rst), .imem_resp(imem_resp), .imem_rdata(imem_rdata), 
        .rs1_s(rs1_s), .rs2_s(rs2_s), .rs1_v(rs1_v), .rs2_v(rs2_v), .if_id_reg(if_id_reg), .id_ex_reg(id_ex_reg) );

    EX  stage_ex( .clk(clk), .rst(rst), .id_ex_reg(id_ex_reg), .ex_mem_reg(ex_mem_reg) );

    MEM stage_mem(.clk(clk), .rst(rst), .dmem_addr(dmem_addr), .dmem_rmask(dmem_rmask), 
        .dmem_wmask(dmem_wmask), .dmem_wdata(dmem_wdata), .ex_mem_reg(ex_mem_reg), .mem_wb_reg(mem_wb_reg));
        
    WB  stage_wb( .clk(clk), .rst(rst), .dmem_rdata(dmem_rdata), .dmem_resp(dmem_resp), 
        .regf_we(regf_we), .rd_sel(rd_sel), .rd_v(rd_v), .mem_wb_reg(mem_wb_reg));


    regfile regfile(
        .*
    );

endmodule : cpu
