module MEM
import rv32i_types::*;
(
    input   logic   clk,
    input   logic   rst,

    output  logic   [31:0]  dmem_addr,
    output  logic   [3:0]   dmem_rmask,
    output  logic   [3:0]   dmem_wmask,
    output  logic   [31:0]  dmem_wdata,
    
    input   ex_mem_stage_reg_t  ex_mem_reg,
    output  mem_wb_stage_reg_t  mem_wb_reg
)

    mem_ctrl_t      mem_ctrl;
    logic   [31:0]  mem_addr;
    logic   [31:0]  rs2_v;

    // get value from prev reg
    always_comb begin
        rs2_v = ex_mem_reg.rs2_v_s;
        mem_ctrl = ex_mem_reg.mem_ctrl_s;
        mem_addr = ex_mem_reg.alu_out_s;
    end

    // dmem read
    always_comb begin
        if( mem_ctrl.mem_we )begin
            dmem_addr = mem_addr;
            unique case (mem_ctrl.funct3)
                store_f3_sb: dmem_wmask = 4'b0001 << mem_addr[1:0];
                store_f3_sh: dmem_wmask = 4'b0011 << mem_addr[1:0];
                store_f3_sw: dmem_wmask = 4'b1111;
                default    : dmem_wmask = 'x;
            endcase
            unique case (mem_ctrl.funct3)
                store_f3_sb: dmem_wdata[8 *mem_addr[1:0] +: 8 ] = rs2_v[7 :0];
                store_f3_sh: dmem_wdata[16*mem_addr[1]   +: 16] = rs2_v[15:0];
                store_f3_sw: dmem_wdata = rs2_v;
                default    : dmem_wdata = 'x;
            endcase
        end
    end

    // dmem write
    always_comb begin
        if( mem_ctrl.mem_re )begin
            dmem_addr = mem_addr;
            unique case (mem_ctrl.funct3)
                load_f3_lb, load_f3_lbu: dmem_rmask = 4'b0001 << mem_addr[1:0];
                load_f3_lh, load_f3_lhu: dmem_rmask = 4'b0011 << mem_addr[1:0];
                load_f3_lw             : dmem_rmask = 4'b1111;
                default                : dmem_rmask = 'x;
            endcase
            mem_addr[1:0] = 2'd0;
        end
    end

    // assign signals to the register struct
    always_comb begin
        mem_wb_reg.pc_s         = ex_mem_reg.pc_s;
        mem_wb_reg.order_s      = ex_mem_reg.order_s;
        mem_wb_reg.wb_ctrl_s    = ex_mem_reg.wb_ctrl_s; 
        mem_wb_reg.rd_s_s       = ex_mem_reg.rd_s_s;
        mem_wb_reg.br_en_s      = ex_mem_reg.br_en_s;
        mem_wb_reg.alu_out_s    = ex_mem_reg.alu_out_s;
        mem_wb_reg.dmem_addr_s  = mem_addr;
    end


endmodule