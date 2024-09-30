// Main Hardware: Data Memory Port
// Function: Assign dmem control signal for load and store

module MEM
import rv32i_types::*;
(
    input   logic           move,
    output  logic           dmem_req,
    
    output  logic   [31:0]  dmem_addr,
    output  logic   [3:0]   dmem_rmask,
    output  logic   [3:0]   dmem_wmask,
    output  logic   [31:0]  dmem_wdata,
    
    input   ex_mem_stage_reg_t  ex_mem_reg,
    output  mem_wb_stage_reg_t  mem_wb_reg
);

    mem_ctrl_t      mem_ctrl;
    logic           valid;

    always_comb begin
        mem_ctrl = ex_mem_reg.mem_ctrl_s;
        dmem_req = valid && (mem_ctrl.mem_we||mem_ctrl.mem_re);
        dmem_addr = ex_mem_reg.dmem_addr_s;

        dmem_wmask = '0;
        dmem_rmask = '0;
        dmem_wdata = '0;
        if( dmem_req ) begin
            dmem_wmask = ex_mem_reg.dmem_wmask_s;
            dmem_rmask = ex_mem_reg.dmem_rmask_s;
            dmem_wdata = ex_mem_reg.dmem_wdata_s;
        end
    end

    // assign signals to the register struct
    always_comb begin
        valid = 1'b0; // make sure valid is 0 before ex_mem stats arrive
        if( ex_mem_reg.valid_s && move ) valid = 1'b1;
        mem_wb_reg.valid_s      = valid;
        mem_wb_reg.inst_s       = ex_mem_reg.inst_s;
        mem_wb_reg.pc_s         = ex_mem_reg.pc_s;
        mem_wb_reg.pc_next_s    = ex_mem_reg.pc_next_s;
        mem_wb_reg.order_s      = ex_mem_reg.order_s;
        mem_wb_reg.wb_ctrl_s    = ex_mem_reg.wb_ctrl_s; 
        mem_wb_reg.rd_s_s       = ex_mem_reg.rd_s_s;
        mem_wb_reg.br_en_s      = ex_mem_reg.br_en_s;
        mem_wb_reg.alu_out_s    = ex_mem_reg.alu_out_s;
        mem_wb_reg.rs1_v_s      = ex_mem_reg.rs1_v_s;
        mem_wb_reg.rs2_v_s      = ex_mem_reg.rs2_v_s;
        mem_wb_reg.rs1_s_s      = ex_mem_reg.rs1_s_s;
        mem_wb_reg.rs2_s_s      = ex_mem_reg.rs2_s_s;
        mem_wb_reg.dmem_addr_s  = dmem_addr;             // 32-bit aligned
        mem_wb_reg.mem_addr_s   = ex_mem_reg.mem_addr_s; // real address
        mem_wb_reg.mem_rmask_s  = dmem_rmask;
        mem_wb_reg.mem_wmask_s  = dmem_wmask;
        mem_wb_reg.mem_wdata_s  = dmem_wdata;
        mem_wb_reg.u_imm_s      = ex_mem_reg.u_imm_s;
    end


endmodule