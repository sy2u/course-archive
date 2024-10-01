// Main Hardware: ALU, CMP
// Function: Execute alu/cmp operation

module EX
import rv32i_types::*;
(   
    input   logic               move,

    output  logic   [4:0]       rs1_s,
    output  logic   [4:0]       rs2_s,
    input   logic   [31:0]      reg_rs1_v,
    input   logic   [31:0]      reg_rs2_v,

    input   id_ex_stage_reg_t   id_ex_reg,
    output  ex_mem_stage_reg_t  ex_mem_reg,

    // forwarding inputs
    input   normal_fw_sel_t     forwardA,
    input   normal_fw_sel_t     forwardB,
    input   logic   [31:0]      forward_wb_v,
    input   logic   [31:0]      forward_mem_v
);

    ex_ctrl_t       ex_ctrl;
    logic           br_en;
    logic   [31:0]  cmp_b;
    logic   [31:0]  alu_a, alu_b, alu_out;
    logic   [31:0]  u_imm, i_imm, s_imm, pc;

    // get value from prev reg
    assign  ex_ctrl = id_ex_reg.ex_ctrl_s;
    assign  u_imm = id_ex_reg.u_imm_s;
    assign  i_imm = id_ex_reg.i_imm_s;
    assign  s_imm = id_ex_reg.s_imm_s;
    assign  pc = id_ex_reg.pc_s;
    assign  rs1_s = id_ex_reg.rs1_s_s;
    assign  rs2_s = id_ex_reg.rs2_s_s;

    // Forwarding
    logic   [31:0]  rs1_v, rs2_v;
    always_comb begin
        unique case (forwardA)
            none:   rs1_v = reg_rs1_v;
            mem_ex: rs1_v = forward_mem_v;
            wb_ex:  rs1_v = forward_wb_v;
            default: rs1_v = reg_rs1_v;
        endcase
        unique case (forwardB)
            none:   rs2_v = reg_rs2_v;
            mem_ex: rs2_v = forward_mem_v;
            wb_ex:  rs2_v = forward_wb_v;
            default: rs2_v = reg_rs2_v;
        endcase  
    end

    // Basic Components
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

    // pull dmem control signal one stage forward, otherwise can't meet timing req
    mem_ctrl_t      mem_ctrl;
    logic   [31:0]  dmem_wdata;
    logic   [3:0]   dmem_rmask, dmem_wmask;
    logic   [31:0]  mem_addr, dmem_addr;
    always_comb begin
        mem_ctrl = id_ex_reg.mem_ctrl_s;
        mem_addr = alu_out;
        dmem_addr = alu_out;
        dmem_addr[1:0] = 2'd0;
        dmem_wmask = '0;
        dmem_rmask = '0;
        dmem_wdata = '0;
        // store: dmem write
        if( mem_ctrl.mem_we )begin
            unique case (mem_ctrl.funct3)
                store_f3_sb: dmem_wmask = 4'b0001 << mem_addr[1:0];
                store_f3_sh: dmem_wmask = 4'b0011 << mem_addr[1:0];
                store_f3_sw: dmem_wmask = 4'b1111;
                default    : dmem_wmask = '0;
            endcase
            unique case (mem_ctrl.funct3)
                store_f3_sb: dmem_wdata[8 * mem_addr[1:0] +: 8 ] = rs2_v[7 :0];
                store_f3_sh: dmem_wdata[16* mem_addr[1]   +: 16] = rs2_v[15:0];
                store_f3_sw: dmem_wdata = rs2_v;
                default    : dmem_wdata = 'x;
            endcase
        end
        // load: dmem read
        else if( mem_ctrl.mem_re )begin
            unique case (mem_ctrl.funct3)
                load_f3_lb, load_f3_lbu: dmem_rmask = 4'b0001 << mem_addr[1:0];
                load_f3_lh, load_f3_lhu: dmem_rmask = 4'b0011 << mem_addr[1:0];
                load_f3_lw             : dmem_rmask = 4'b1111;
                default                : dmem_rmask = '0;
            endcase
        end
    end

    // assign signals to the register struct
    always_comb begin
            ex_mem_reg.valid_s      = '0;
            if( move && id_ex_reg.valid_s ) ex_mem_reg.valid_s = '1;
            ex_mem_reg.inst_s       = id_ex_reg.inst_s;
            ex_mem_reg.pc_s         = id_ex_reg.pc_s;
            ex_mem_reg.pc_next_s    = id_ex_reg.pc_next_s;
            ex_mem_reg.order_s      = id_ex_reg.order_s;
            ex_mem_reg.mem_ctrl_s   = mem_ctrl;
            ex_mem_reg.wb_ctrl_s    = id_ex_reg.wb_ctrl_s;
            ex_mem_reg.u_imm_s      = u_imm;
            ex_mem_reg.alu_out_s    = alu_out;
            ex_mem_reg.br_en_s      = br_en;
            ex_mem_reg.rs1_v_s      = rs1_v;
            ex_mem_reg.rs2_v_s      = rs2_v;
            ex_mem_reg.rs1_s_s      = rs1_s;
            ex_mem_reg.rs2_s_s      = rs2_s;
            ex_mem_reg.rd_s_s       = id_ex_reg.rd_s_s; 
            ex_mem_reg.dmem_wdata_s = dmem_wdata;
            ex_mem_reg.dmem_rmask_s = dmem_rmask;
            ex_mem_reg.dmem_wmask_s = dmem_wmask;
            ex_mem_reg.dmem_addr_s  = dmem_addr;// 32-bit aligned
            ex_mem_reg.mem_addr_s   = mem_addr; // real address
    end


endmodule