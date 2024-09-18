// Main Hardware: Register File, Control ROM
// Function: Instruction Decode and Register File Read

module ID
import rv32i_types::*;
(
    input   logic               clk,
    input   logic               rst,
    input   logic               imem_resp,
    input   logic   [31:0]      imem_rdata,

    output  logic   [4:0]       rs1_s,
    output  logic   [4:0]       rs2_s,
    input   logic   [31:0]      rs1_v,
    input   logic   [31:0]      rs2_v,

    input   if_id_stage_reg_t   if_id_reg,
    output  id_ex_stage_reg_t   id_ex_reg
)
    ex_ctrl_t   ex_ctrl;
    mem_ctrl_t  mem_ctrl;
    wb_ctrl_t   wb_ctrl;

    logic   [31:0]  inst;
    logic   [2:0]   funct3;
    logic   [6:0]   funct7;
    logic   [6:0]   opcode;
    logic   [31:0]  i_imm;
    logic   [31:0]  s_imm;
    // logic   [31:0]  b_imm;
    logic   [31:0]  u_imm;
    // logic   [31:0]  j_imm;
    logic   [4:0]   rd_s;

    assign funct3 = inst[14:12];
    assign funct7 = inst[31:25];
    assign opcode = inst[6:0];
    assign i_imm  = {{21{inst[31]}}, inst[30:20]};
    assign s_imm  = {{21{inst[31]}}, inst[30:25], inst[11:7]};
    // assign b_imm  = {{20{inst[31]}}, inst[7], inst[30:25], inst[11:8], 1'b0};
    assign u_imm  = {inst[31:12], 12'h000};
    // assign j_imm  = {{12{inst[31]}}, inst[19:12], inst[20], inst[30:21], 1'b0};
    assign rs1_s  = inst[19:15];
    assign rs2_s  = inst[24:20];
    assign rd_s   = inst[11:7];

    // read instruction memory data
    always_ff @(posedge clk) begin
        if (rst) begin
            inst <= '0;
        end else if (imem_resp) begin
            inst <= imem_rdata;
        end
    end

    // control ROM
    always_comb begin
        commit = 1'b1;
        unique case (opcode)
            op_b_lui: begin
                wb_ctrl.rd_m_sel = u_imm_m;
                wb_ctrl.regf_we = 1'b1;
            end
            op_b_auipc: begin
                ex_ctrl.aluop = alu_op_add;
                ex_ctrl.alu_m1_sel = pc_out;
                ex_ctrl.alu_m2_sel = u_imm_m;
                wb_ctrl.rd_m_sel = alu_out;
                wb_ctrl.regf_we = 1'b1;
            end
            op_b_store: begin
                // mem_addr = rs1_v + s_imm;
                ex_ctrl.aluop = alu_op_add;
                ex_ctrl.alu_m1_sel = rs1_out;
                ex_ctrl.alu_m2_sel = s_imm;
                // set mem_rmask
                mem_ctrl.funct3 = funct3;
                mem_ctrl.mem_we = 1'b1;
            end
            op_b_load: begin
                // mem_addr = rs1_v + i_imm;
                ex_ctrl.aluop = alu_op_add;
                ex_ctrl.alu_m1_sel = rs1_out;
                ex_ctrl.alu_m2_sel = i_imm_m;
                // set mem_rmask
                mem_ctrl.funct3 = funct3;
                mem_ctrl.mem_re = 1'b1;
                // set write back register select
                wb_ctrl.regf_we = 1'b1;
                unique case (funct3)
                    load_f3_lb: rd_m_sel = lb;
                    load_f3_lh: rd_m_sel = lh;
                    load_f3_lw: rd_m_sel = lw;
                    load_f3_lbu: rd_m_sel = lbu;
                    load_f3_lhu: rd_m_sel = lhu;
                    default: wb_ctrl.rd_m_sel = 'x;
                endcase
            end
            op_b_imm: begin
                wb_ctrl.regf_we = 1'b1;
                ex_ctrl.cmp_sel = i_imm_m;
                unique case (funct3)
                    arith_f3_slt: begin
                        ex_ctrl.cmpop = cmp_op_blt;
                        wb_ctrl.rd_m_sel = ext_br;
                    end
                    arith_f3_sltu: begin
                        ex_ctrl.cmpop = cmp_op_bltu;
                        wb_ctrl.rd_m_sel = ext_br;
                    end
                    arith_f3_sr: begin
                        if (funct7[5]) begin
                            ex_ctrl.aluop = alu_op_sra;
                        end else begin
                            ex_ctrl.aluop = alu_op_srl;
                        end
                        wb_ctrl.rd_m_sel = alu_out;
                    end
                    default: begin
                        ex_ctrl.aluop = funct3;
                        wb_ctrl.rd_m_sel = alu_out;
                    end
                endcase
            end
            op_b_reg: begin
                ex_ctrl.cmp_sel = rs2_out;
                unique case (funct3)
                    arith_f3_slt: begin
                        ex_ctrl.cmpop = cmp_op_blt;
                        wb_ctrl.rd_m_sel = ext_br;
                    end
                    arith_f3_sltu: begin
                        ex_ctrl.cmpop = cmp_op_bltu;
                        wb_ctrl.rd_m_sel = ext_br;
                    end
                    arith_f3_sr: begin
                        if (funct7[5]) begin
                            ex_ctrl.aluop = alu_op_sra;
                        end else begin
                            ex_ctrl.aluop = alu_op_srl;
                        end
                        wb_ctrl.rd_m_sel = alu_out;
                    end
                    arith_f3_add: begin
                        if (funct7[5]) begin
                            ex_ctrl.aluop = alu_op_sub;
                        end else begin
                            ex_ctrl.aluop = alu_op_add;
                        end
                        wb_ctrl.rd_m_sel = alu_out;
                    end
                    default: begin
                        ex_ctrl.aluop = funct3;
                        wb_ctrl.rd_m_sel = alu_out;
                    end
                endcase
            end
            // CP1: NO Jump or Branch
            // op_b_jal:
            // op_b_jalr:
            // op_b_br: begin
            default: 
        endcase
    end

    // assign signals to the register struct
    always_comb begin
        id_ex_reg.inst_s = inst;
        id_ex_reg.pc_s = if_id_reg.pc_s;
        id_ex_reg.order_s = if_id_reg.order_s;
        id_ex_reg.ex_ctrl_s = ex_ctrl;
        id_ex_reg.mem_ctrl_s = mem_ctrl;
        id_ex_reg.wb_ctrl_s = wb_ctrl;
        id_ex_reg.u_imm_s = u_imm;
        id_ex_reg.s_imm_s = s_imm;
        id_ex_reg.i_imm_s = i_imm;
        id_ex_reg.rs1_v_s = rs1_v;
        id_ex_reg.rs2_v_s = rs2_v;
        id_ex_reg.rd_s_s = rd_s;
    end

endmodule