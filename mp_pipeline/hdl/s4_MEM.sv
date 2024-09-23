module MEM
import rv32i_types::*;
(
    output  logic   [31:0]  dmem_addr,
    output  logic   [3:0]   dmem_rmask,
    output  logic   [3:0]   dmem_wmask,
    output  logic   [31:0]  dmem_wdata,
    
    input   ex_mem_stage_reg_t  ex_mem_reg,
    output  mem_wb_stage_reg_t  mem_wb_reg
);

    mem_ctrl_t      mem_ctrl;
    logic   [31:0]  rs2_v;
    logic   [31:0]  mem_addr;

    // get value from prev reg
    always_comb begin
        rs2_v = ex_mem_reg.rs2_v_s;
        mem_ctrl = ex_mem_reg.mem_ctrl_s;
    end

    always_comb begin
        mem_addr = ex_mem_reg.alu_out_s;
        dmem_wmask = '0;
        dmem_rmask = '0;
        dmem_wdata = '0;
        // load: dmem read
        if( mem_ctrl.mem_we )begin
            unique case (mem_ctrl.funct3)
                store_f3_sb: dmem_wmask = 4'b0001 << mem_addr[1:0];
                store_f3_sh: 
                    dmem_wmask = 4'b0011 << mem_addr[1:0];
                store_f3_sw: 
                    dmem_wmask = 4'b1111;
                default    : dmem_wmask = '0;
            endcase
            unique case (mem_ctrl.funct3)
                store_f3_sb: dmem_wdata[8 * mem_addr[1:0] +: 8 ] = rs2_v[7 :0];
                store_f3_sh: dmem_wdata[16* mem_addr[1]   +: 16] = rs2_v[15:0];
                store_f3_sw: dmem_wdata = rs2_v;
                default    : dmem_wdata = 'x;
            endcase
        end
    
        // store: dmem write
        if( mem_ctrl.mem_re )begin
            unique case (mem_ctrl.funct3)
                load_f3_lb, load_f3_lbu: dmem_rmask = 4'b0001 << mem_addr[1:0];
                load_f3_lh, load_f3_lhu: dmem_rmask = 4'b0011 << mem_addr[1:0];
                load_f3_lw             : dmem_rmask = 4'b1111;
                default                : dmem_rmask = '0;
            endcase
        end
    end

    always_comb begin
        dmem_addr = ex_mem_reg.alu_out_s;
        dmem_addr[1:0] = 2'd0;
    end

    // assign signals to the register struct
    always_comb begin
        mem_wb_reg.inst_s       = ex_mem_reg.inst_s;
        mem_wb_reg.pc_s         = ex_mem_reg.pc_s;
        mem_wb_reg.pc_next_s    = ex_mem_reg.pc_next_s;
        mem_wb_reg.order_s      = ex_mem_reg.order_s;
        mem_wb_reg.valid_s      = ex_mem_reg.valid_s;
        mem_wb_reg.wb_ctrl_s    = ex_mem_reg.wb_ctrl_s; 
        mem_wb_reg.rd_s_s       = ex_mem_reg.rd_s_s;
        mem_wb_reg.br_en_s      = ex_mem_reg.br_en_s;
        mem_wb_reg.alu_out_s    = ex_mem_reg.alu_out_s;
        mem_wb_reg.rs1_v_s      = ex_mem_reg.rs1_v_s;
        mem_wb_reg.rs2_v_s      = ex_mem_reg.rs2_v_s;
        mem_wb_reg.rs1_s_s      = ex_mem_reg.rs1_s_s;
        mem_wb_reg.rs2_s_s      = ex_mem_reg.rs2_s_s;
        mem_wb_reg.dmem_addr_s  = dmem_addr;
        mem_wb_reg.mem_addr_s   = mem_addr;
        mem_wb_reg.mem_rmask_s  = dmem_rmask;
        mem_wb_reg.mem_wmask_s  = dmem_wmask;
        mem_wb_reg.mem_wdata_s  = dmem_wdata;
        mem_wb_reg.u_imm_s      = ex_mem_reg.u_imm_s;
    end


endmodule