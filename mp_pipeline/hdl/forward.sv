module forward
import rv32i_types::*;
(
    input   id_ex_stage_reg_t   id_ex_reg,
    input   ex_mem_stage_reg_t  ex_mem_reg, // reg read is issued in ex_mem stage
    input   mem_wb_stage_reg_t  mem_wb_reg,

    output  normal_fw_sel_t     forwardA,
    output  normal_fw_sel_t     forwardB,
    output  logic   [31:0]      mem_v,

    input   logic               regf_we,
    output  decode_fw_sel_t     fowarDe, // decode hazard (transparent register) control

    input   logic   [4:0]       id_rs1,
    input   logic   [4:0]       id_rs2,
    output  logic               forward_stall
);
    
    // Dmem Stall Forwarding
    always_comb begin
        forward_stall = 1'b0;
        if( id_ex_reg.mem_ctrl_s.mem_re && ((id_ex_reg.rd_s_s==id_rs1) || (id_ex_reg.rd_s_s==id_rs2) )) 
            forward_stall = 1'b1; // insert nop
    end

    // Decode Forwarding
    always_comb begin
        fowarDe = none_d;
        if( regf_we && (mem_wb_reg.rd_s_s!=0) )begin
            if      ( mem_wb_reg.rd_s_s == id_ex_reg.rs1_s_s )  fowarDe = rs1_f;
            else if ( mem_wb_reg.rd_s_s == id_ex_reg.rs2_s_s )  fowarDe = rs2_f;
        end
    end

    // Normal ALU Forwarding
    always_comb begin
        forwardA = none;
        forwardB = none;
        // ex_mem_reg data have higher priority
        if ( mem_wb_reg.wb_ctrl_s.regf_we && (mem_wb_reg.rd_s_s!=0) ) begin
            if( mem_wb_reg.rd_s_s==id_ex_reg.rs1_s_s )  forwardA = wb_ex;
            if( mem_wb_reg.rd_s_s==id_ex_reg.rs2_s_s )  forwardB = wb_ex;
        end
        if ( ex_mem_reg.wb_ctrl_s.regf_we && (ex_mem_reg.rd_s_s!=0) ) begin
            if( ex_mem_reg.rd_s_s==id_ex_reg.rs1_s_s )  forwardA = mem_ex;
            if( ex_mem_reg.rd_s_s==id_ex_reg.rs2_s_s )  forwardB = mem_ex;
        end
    end

    // Normal ALU Forwarding Data Select
    always_comb begin
        unique case (ex_mem_reg.wb_ctrl_s.rd_m_sel)
            u_imm_m_rd: mem_v = ex_mem_reg.u_imm_s;
            alu_out_rd: mem_v = ex_mem_reg.alu_out_s;
            ext_br:     mem_v = {31'd0, ex_mem_reg.br_en_s}; 
            default:    mem_v = 'x;
        endcase
        // unique case (mem_wb_reg.wb_ctrl_s.rd_m_sel)
        //     u_imm_m_rd: wb_v = mem_wb_reg.u_imm_s;
        //     alu_out_rd: wb_v = mem_wb_reg.alu_out_s;
        //     ext_br:     wb_v = {31'd0, mem_wb_reg.br_en_s}; 
        //     default:    wb_v = 'x;
        // endcase
    end

endmodule