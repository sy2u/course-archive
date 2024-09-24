// Main Hardware: Big Mux, Register File
// Function: Update the register file w/ new data

module WB
import rv32i_types::*;
(
    input   logic           move,

    input   logic   [31:0]  dmem_rdata,
    input   logic           dmem_resp,

    output  logic           regf_we,
    output  logic   [4:0]   rd_sel,
    output  logic   [31:0]  rd_v,

    input   mem_wb_stage_reg_t  mem_wb_reg
);

    wb_ctrl_t       wb_ctrl;
    logic           valid;
    logic   [31:0]  inst;
    logic   [63:0]  order;
    logic           br_en;
    logic   [31:0]  mem_addr, dmem_addr, u_imm, alu_out;
    logic   [4:0]   rs1_s, rs2_s;
    logic   [31:0]  rs1_v, rs2_v;
    logic   [31:0]  pc, pc_next;
    logic   [3:0]   mem_rmask, mem_wmask;
    logic   [31:0]  mem_wdata;

    // get value from prev reg
    always_comb begin
        wb_ctrl     = mem_wb_reg.wb_ctrl_s;
        mem_addr    = mem_wb_reg.mem_addr_s;
        regf_we     = wb_ctrl.regf_we;
        if( wb_ctrl.rd_m_sel inside {lb,lbu,lh,lhu,lw} ) regf_we = wb_ctrl.regf_we && dmem_resp;
        // rvfi monitor
        rs1_v       = mem_wb_reg.rs1_v_s;
        rs2_v       = mem_wb_reg.rs2_v_s;
        rs1_s       = mem_wb_reg.rs1_s_s;
        rs2_s       = mem_wb_reg.rs2_s_s;
        pc          = mem_wb_reg.pc_s;
        pc_next     = mem_wb_reg.pc_next_s;
        mem_rmask   = mem_wb_reg.mem_rmask_s;
        mem_wmask   = mem_wb_reg.mem_wmask_s;
        mem_wdata   = mem_wb_reg.mem_wdata_s;
        inst        = mem_wb_reg.inst_s;
        order       = mem_wb_reg.order_s;
        dmem_addr   = mem_wb_reg.dmem_addr_s;
        br_en       = mem_wb_reg.br_en_s;
        u_imm       = mem_wb_reg.u_imm_s;
        alu_out     = mem_wb_reg.alu_out_s;
        // valid
        valid = 1'b0;
        if( mem_wb_reg.valid_s )
            valid = (wb_ctrl.rd_m_sel inside {lb,lbu,lh,lhu,lw} && dmem_resp) || (!(wb_ctrl.rd_m_sel inside {lb,lbu,lh,lhu,lw}) && move);
    end

    // reg file big mux
    always_comb begin
        rd_v = '0;
        rd_sel = '0;
        if( wb_ctrl.regf_we )begin
            rd_sel = mem_wb_reg.rd_s_s;
            unique case (wb_ctrl.rd_m_sel)
                u_imm_m_rd: rd_v = u_imm;
                alu_out_rd: rd_v = alu_out;
                ext_br: rd_v = {31'd0, br_en}; 
                lb : rd_v = {{24{dmem_rdata[7 +8 *mem_addr[1:0]]}}, dmem_rdata[8 *mem_addr[1:0] +: 8 ]};
                lbu: rd_v = {{24{1'b0}}                          , dmem_rdata[8 *mem_addr[1:0] +: 8 ]};
                lh : rd_v = {{16{dmem_rdata[15+16*mem_addr[1]  ]}}, dmem_rdata[16*mem_addr[1]   +: 16]};
                lhu: rd_v = {{16{1'b0}}                          , dmem_rdata[16*mem_addr[1]   +: 16]};
                lw : rd_v = dmem_rdata;
                default: rd_v = 'x;
            endcase
        end
    end

endmodule
