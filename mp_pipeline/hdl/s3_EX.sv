// Main Hardware: ALU, CMP
// Function: Execute alu/cmp operation

module EX
import rv32i_types::*;
(   
    input   logic               rst,

    input   logic   [31:0]      rs1_v,
    input   logic   [31:0]      rs2_v,

    input   id_ex_stage_reg_t   id_ex_reg,
    output  ex_mem_stage_reg_t  ex_mem_reg
);

    ex_ctrl_t       ex_ctrl;
    logic           br_en;
    logic   [31:0]  cmp_b;
    logic   [31:0]  alu_a, alu_b, alu_out;
    logic   [31:0]  u_imm, i_imm, s_imm, pc;

    // get value from prev reg
    always_comb begin
        ex_ctrl = id_ex_reg.ex_ctrl_s;
        u_imm = id_ex_reg.u_imm_s;
        i_imm = id_ex_reg.i_imm_s;
        s_imm = id_ex_reg.s_imm_s;
        pc = id_ex_reg.pc_s;
    end


    always_comb begin
        // alu_mux
        unique case (ex_ctrl.alu_m1_sel)
            rs1_out:alu_a = rs1_v;
            pc_out: alu_a = pc;
            default:alu_a = 'x;
        endcase
        unique case (ex_ctrl.alu_m2_sel)
            rs2_out:  alu_b = rs2_v;
            u_imm_m:  alu_b = u_imm;
            i_imm_m:  alu_b = i_imm;
            s_imm_m:  alu_b = s_imm;
            const4:   alu_b = 'd4;
            default: alu_b = 'x;
        endcase
        // cmp_mux
        unique case (ex_ctrl.cmp_sel)
            rs2_out_cmp:    cmp_b = rs2_v;
            i_imm_m_cmp:    cmp_b = i_imm;
            default:        cmp_b = 'x;
        endcase
    end

    ALU alu(.aluop(ex_ctrl.aluop), .a(alu_a), .b(alu_b), .aluout(alu_out));
    CMP cmp(.cmpop(ex_ctrl.cmpop), .a(rs1_v), .b(cmp_b), .br_en(br_en));

    // assign signals to the register struct
    always_comb begin
            ex_mem_reg.inst_s       = id_ex_reg.inst_s;
            ex_mem_reg.pc_s         = id_ex_reg.pc_s;
            ex_mem_reg.pc_next_s    = id_ex_reg.pc_next_s;
            ex_mem_reg.order_s      = id_ex_reg.order_s;
            ex_mem_reg.valid_s      = id_ex_reg.valid_s;
            ex_mem_reg.mem_ctrl_s   = id_ex_reg.mem_ctrl_s;
            ex_mem_reg.wb_ctrl_s    = id_ex_reg.wb_ctrl_s;
            ex_mem_reg.u_imm_s      = u_imm;
            ex_mem_reg.alu_out_s    = alu_out;
            ex_mem_reg.br_en_s      = br_en;
            ex_mem_reg.rs1_v_s      = rs1_v;
            ex_mem_reg.rs2_v_s      = rs2_v;
            ex_mem_reg.rs1_s_s      = id_ex_reg.rs1_s_s;
            ex_mem_reg.rs2_s_s      = id_ex_reg.rs2_s_s;
            ex_mem_reg.rd_s_s       = id_ex_reg.rd_s_s; 
        if (rst) begin
            ex_mem_reg.valid_s       = '0;
        end 
    end


endmodule