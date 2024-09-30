// version: cp2 - support multiple cycle memory model
//          stall - using back pressure
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

    // Memory Access
    logic           imem_req, dmem_req;
    logic           move;

    stall stall( .clk(clk), .rst(rst), 
        .imem_req(imem_req), .dmem_req(dmem_req), 
        .imem_resp(imem_resp), .dmem_resp(dmem_resp),
        .move(move)
    );

    // Update Stage Register
    if_id_stage_reg_t   if_id_reg, if_id_reg_next;
    id_ex_stage_reg_t   id_ex_reg, id_ex_reg_next;
    ex_mem_stage_reg_t  ex_mem_reg, ex_mem_reg_next;
    mem_wb_stage_reg_t  mem_wb_reg, mem_wb_reg_next;

    always_ff @( posedge clk ) begin
        if( move ) begin
            id_ex_reg <= id_ex_reg_next;
            ex_mem_reg <= ex_mem_reg_next;
            mem_wb_reg <= mem_wb_reg_next;
            if_id_reg <= if_id_reg_next;
        end
    end

    // Hazard Control
    logic   [31:0]      forward_mem_v, forward_wb_v;
    normal_fw_sel_t     forwardA, forwardB;

    forward forward( 
        .id_ex_reg(id_ex_reg), .ex_mem_reg(ex_mem_reg), .mem_wb_reg(mem_wb_reg), 
        .forwardA(forwardA), .forwardB(forwardB), .mem_v(forward_mem_v), .wb_v(forward_wb_v));

    // Connect Hardware
    logic           regf_we;
    logic   [31:0]  rd_v;
    logic   [4:0]   rs1_s, rs2_s, rd_sel;
    logic   [31:0]  rs1_v, rs2_v;

    IF  stage_if( .clk(clk), .rst(rst),
        .move(move), .imem_req(imem_req), 
        .imem_addr(imem_addr), .imem_rmask(imem_rmask), 
        .if_id_reg(if_id_reg_next) 
    );

    ID  stage_id( .clk(clk),
        .move(move),
        .imem_resp(imem_resp), .imem_rdata(imem_rdata),
        .if_id_reg(if_id_reg), .id_ex_reg(id_ex_reg_next) 
    );

    EX  stage_ex( .move(move),
        .rs1_s(rs1_s), .rs2_s(rs2_s), .reg_rs1_v(rs1_v), .reg_rs2_v(rs2_v), 
        .id_ex_reg(id_ex_reg), .ex_mem_reg(ex_mem_reg_next),
        .ex_mem_curr(ex_mem_reg), .mem_wb_curr(mem_wb_reg), 
        .forwardA(forwardA), .forwardB(forwardB), .forward_mem_v(forward_mem_v), .forward_wb_v(forward_wb_v)
    );

    MEM stage_mem(
        .move(move), .dmem_req(dmem_req),
        .dmem_addr(dmem_addr), .dmem_rmask(dmem_rmask), .dmem_wmask(dmem_wmask), .dmem_wdata(dmem_wdata), 
        .ex_mem_reg(ex_mem_reg), .mem_wb_reg(mem_wb_reg_next)
    );

    WB  stage_wb( .move(move),
        .dmem_rdata(dmem_rdata), .dmem_resp(dmem_resp),
        .regf_we(regf_we), .rd_sel(rd_sel), .rd_v(rd_v), 
        .mem_wb_reg(mem_wb_reg)
    );

    regfile regfile(
        .clk(clk), .rst(rst),
        .regf_we(regf_we),
        .rd_v(rd_v),
        .rs1_s(rs1_s), .rs2_s(rs2_s), .rd_s(rd_sel),
        .rs1_v(rs1_v), .rs2_v(rs2_v)
    );

endmodule : cpu
