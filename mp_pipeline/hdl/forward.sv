module forward
import rv32i_types::*;
(
    input   id_ex_stage_reg_t   id_ex_reg,
    input   ex_mem_stage_reg_t  ex_mem_reg,
    input   mem_wb_stage_reg_t  mem_wb_reg,

    output  normal_fw_sel_t     forwardA,
    output  normal_fw_sel_t     forwardB,
    output  decode_fw_sel_t     fowarDe,
    output  logic   [31:0]      mem_v,
    output  logic   [31:0]      wb_v
);

    // Decode Forwarding
    always_comb begin
        fowarDe = none_d;
        if( mem_wb_reg.wb_ctrl_s.regf_we && (mem_wb_reg.rd_s_s!=0) )begin
            if( mem_wb_reg.rd_s_s == id_ex_reg.rs1_s_s ) fowarDe = rs1_f;
            if( mem_wb_reg.rd_s_s == id_ex_reg.rs2_s_s ) fowarDe = rs2_f;
        end
    end

    // Normal ALU Forwarding
    always_comb begin
        forwardA = none;
        forwardB = none;
        if( ex_mem_reg.wb_ctrl_s.regf_we && (ex_mem_reg.rd_s_s!=0) )begin
            // ex_mem_reg stats have higher priority
            if( ex_mem_reg.rd_s_s == id_ex_reg.rs1_s_s ) forwardA = mem_ex;
            else if( mem_wb_reg.rd_s_s == id_ex_reg.rs1_s_s ) forwardA = wb_ex;
            if( ex_mem_reg.rd_s_s == id_ex_reg.rs2_s_s ) forwardB = mem_ex;
            else if( mem_wb_reg.rd_s_s == id_ex_reg.rs2_s_s ) forwardB = wb_ex;
        end
    end

    always_comb begin
        unique case (ex_mem_reg.wb_ctrl_s.rd_m_sel)
            u_imm_m_rd: mem_v = ex_mem_reg.u_imm_s;
            alu_out_rd: mem_v = ex_mem_reg.alu_out_s;
            ext_br:     mem_v = {31'd0, ex_mem_reg.br_en_s}; 
            default:    mem_v = 'x;
        endcase
        unique case (mem_wb_reg.wb_ctrl_s.rd_m_sel)
            u_imm_m_rd: wb_v = mem_wb_reg.u_imm_s;
            alu_out_rd: wb_v = mem_wb_reg.alu_out_s;
            ext_br:     wb_v = {31'd0, mem_wb_reg.br_en_s}; 
            default:    wb_v = 'x;
        endcase
    end

endmodule