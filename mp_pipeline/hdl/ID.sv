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

    input   if_id_stage_reg_t   if_id_reg,
    output  id_ex_stage_reg_t   id_ex_reg
);

    ex_ctrl_t   ex_ctrl;
    mem_ctrl_t  mem_ctrl;
    wb_ctrl_t   wb_ctrl;

    logic   [4:0]  rs1_addr, rs2_addr;

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

    logic [63:0] order, order_next;

    // read instruction memory data
    always_ff @(posedge clk) begin
        if (rst) begin
            inst <= '0;
            order <= '0;
        end else if (imem_resp) begin
            inst <= imem_rdata;
            order <= order_next;
        end
    end

    // always_comb begin
    //     inst = 'x;
    //     if (rst) begin
    //         inst = '0;
    //     end else if (imem_resp) begin
    //         inst = imem_rdata;
    //     end
    // end

    always_comb begin
        order_next = order + 'd1;
    end

    // control ROM
    always_comb begin
        // commit = 1'b1;
        ex_ctrl.alu_m1_sel = invalid_alu_m1;
        ex_ctrl.alu_m2_sel = invalid_alu_m2;
        ex_ctrl.aluop = alu_op_add; // random chose
        ex_ctrl.cmp_sel = invalid_cmp;
        ex_ctrl.cmpop = cmp_op_beq; // random chose
        mem_ctrl.funct3 = funct3;
        mem_ctrl.mem_re = 'x;
        mem_ctrl.mem_we = 'x;
        wb_ctrl.regf_we = 1'b0;
        wb_ctrl.rd_m_sel = invalid_rd;
        rs1_addr = '0;
        rs2_addr = '0;
        unique case (opcode)
            op_b_lui: begin
                wb_ctrl.rd_m_sel = u_imm_m_rd;
                wb_ctrl.regf_we = 1'b1;
                // monitor
                rs1_addr = '0;
                rs2_addr = '0;
            end
            op_b_auipc: begin
                ex_ctrl.aluop = alu_op_add;
                ex_ctrl.alu_m1_sel = pc_out;
                ex_ctrl.alu_m2_sel = u_imm_m;
                wb_ctrl.rd_m_sel = alu_out_rd;
                wb_ctrl.regf_we = 1'b1;
                // monitor
                rs1_addr = '0;
                rs2_addr = '0;
            end
            op_b_store: begin
                // mem_addr = rs1_v + s_imm;
                ex_ctrl.aluop = alu_op_add;
                ex_ctrl.alu_m1_sel = rs1_out;
                ex_ctrl.alu_m2_sel = s_imm_m;
                // set mem_rmask
                mem_ctrl.funct3 = funct3;
                mem_ctrl.mem_we = 1'b1;
                // monitor
                rs1_addr = rs1_s;
                rs2_addr = '0;
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
                    load_f3_lb: wb_ctrl.rd_m_sel = lb;
                    load_f3_lh: wb_ctrl.rd_m_sel = lh;
                    load_f3_lw: wb_ctrl.rd_m_sel = lw;
                    load_f3_lbu: wb_ctrl.rd_m_sel = lbu;
                    load_f3_lhu: wb_ctrl.rd_m_sel = lhu;
                    default: wb_ctrl.rd_m_sel = invalid_rd;
                endcase
                // monitor
                rs1_addr = rs1_s;
                rs2_addr = '0;
            end
            op_b_imm: begin
                wb_ctrl.regf_we = 1'b1;
                ex_ctrl.cmp_sel = i_imm_m_cmp;
                ex_ctrl.alu_m1_sel = rs1_out;
                ex_ctrl.alu_m2_sel = i_imm_m;
                // monitor
                rs1_addr = rs1_s;
                rs2_addr = '0;
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
                        wb_ctrl.rd_m_sel = alu_out_rd;
                    end
                    default: begin
                        ex_ctrl.aluop = alu_op_add; // random chose
                        wb_ctrl.rd_m_sel = alu_out_rd;
                    end
                endcase
            end
            op_b_reg: begin
                ex_ctrl.cmp_sel = rs2_out_cmp;
                ex_ctrl.alu_m1_sel = rs1_out;
                ex_ctrl.alu_m2_sel = rs2_out;
                // monitor
                rs1_addr = rs1_s;
                rs2_addr = rs2_s;
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
                        wb_ctrl.rd_m_sel = alu_out_rd;
                    end
                    arith_f3_add: begin
                        if (funct7[5]) begin
                            ex_ctrl.aluop = alu_op_sub;
                        end else begin
                            ex_ctrl.aluop = alu_op_add;
                        end
                        wb_ctrl.rd_m_sel = alu_out_rd;
                    end
                    default: begin
                        ex_ctrl.aluop = alu_op_add; // random chose
                        wb_ctrl.rd_m_sel = alu_out_rd;
                    end
                endcase
            end
            // CP1: NO Jump or Branch
            // op_b_jal:
            // op_b_jalr:
            // op_b_br:
            default: begin
            end
        endcase
    end

    // assign signals to the register struct
    always_comb begin
            id_ex_reg.inst_s    = inst;
            id_ex_reg.pc_s      = if_id_reg.pc_s;
            id_ex_reg.pc_next_s = if_id_reg.pc_next_s;
            id_ex_reg.order_s   = order;
            id_ex_reg.valid_s   = if_id_reg.valid_s;
            id_ex_reg.ex_ctrl_s = ex_ctrl;
            id_ex_reg.mem_ctrl_s= mem_ctrl;
            id_ex_reg.wb_ctrl_s = wb_ctrl;
            id_ex_reg.u_imm_s   = u_imm;
            id_ex_reg.s_imm_s   = s_imm;
            id_ex_reg.i_imm_s   = i_imm;
            id_ex_reg.rs1_s_s   = rs1_addr;
            id_ex_reg.rs2_s_s   = rs2_addr;
            id_ex_reg.rd_s_s    = rd_s;
        if (rst) begin
            id_ex_reg.valid_s   = '0;
        end
    end

endmodule