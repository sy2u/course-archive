// CP1: assume mem is always ready, omit mem_resp signal

module WB
import rv32i_types::*;
(
    input   logic   clk,
    input   logic   rst,

    input   logic   [31:0]  dmem_rdata,
    input   logic           dmem_resp,

    output  logic           regf_we,
    output  logic           rd_sel,
    output  logic   [31:0]  rd_v,

    input   mem_wb_stage_reg_t  mem_wb_reg
)

    wb_ctrl_t       wb_ctrl;
    logic           br_en;
    logic   [31:0]  mem_addr;

    // get value from prev reg
    always_comb begin
        mem_addr = mem_wb_reg.dmem_addr_s;
        br_en = mem_wb_reg.br_en_s;
        u_imm = mem_wb_reg.u_imm_s;
        alu_out = mem_wb_reg.alu_out_s;
        rd_sel = mem_wb_reg.rd_s_s;
    end

    // reg file big mux
    always_comb begin
        if( wb_ctrl.regf_we )begin
            unique case (wb_ctrl.rd_m_sel)
                u_imm: rd_v = u_imm;
                alu_out: rd_v = alu_out;
                ext_br: rd_v = {31'd0, br_en}; 
                lb : rd_v = {{24{dmem_rdata[7 +8 *mem_addr[1:0]]}}, dmem_rdata[8 *mem_addr[1:0] +: 8 ]};
                lbu: rd_v = {{24{1'b0}}                          , dmem_rdata[8 *mem_addr[1:0] +: 8 ]};
                lh : rd_v = {{16{dmem_rdata[15+16*mem_addr[1]  ]}}, dmem_rdata[16*mem_addr[1]   +: 16]};
                lhu: rd_v = {{16{1'b0}}                          , dmem_rdata[16*mem_addr[1]   +: 16]};
                lw : rd_v = dmem_rdata;
                default    : rd_v = 'x;
            endcase
        end
    end


endmodule
