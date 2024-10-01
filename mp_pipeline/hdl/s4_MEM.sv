// Main Hardware: Data Memory Port
// Function: Assign dmem control signal for load and store

module MEM
import rv32i_types::*;
(
    input   logic           clk,
    input   logic           move,

    input   logic   [31:0]  dmem_rdata,
    input   logic           dmem_resp,
    
    input   ex_mem_stage_reg_t  ex_mem_reg,
    output  mem_wb_stage_reg_t  mem_wb_reg
);

    logic   [31:0]  dmem_store;

    always_ff @( posedge clk ) begin
        if (dmem_resp)  dmem_store <= dmem_rdata;
    end

    // assign signals to the register struct
    always_comb begin
        // dmem data
        mem_wb_reg.dmem_rdata_s = (move && dmem_resp) ? dmem_rdata : dmem_store;
        // valid
        mem_wb_reg.valid_s = 1'b0;
        if( ex_mem_reg.valid_s && move )    mem_wb_reg.valid_s = 1'b1;
        // monitor
        mem_wb_reg.order_s      = ex_mem_reg.order_s;
        mem_wb_reg.inst_s       = ex_mem_reg.inst_s;
        mem_wb_reg.pc_s         = ex_mem_reg.pc_s;
        mem_wb_reg.pc_next_s    = ex_mem_reg.pc_next_s;
        mem_wb_reg.rd_s_s       = ex_mem_reg.rd_s_s;
        mem_wb_reg.rs1_v_s      = ex_mem_reg.rs1_v_s;
        mem_wb_reg.rs2_v_s      = ex_mem_reg.rs2_v_s;
        mem_wb_reg.rs1_s_s      = ex_mem_reg.rs1_s_s;
        mem_wb_reg.rs2_s_s      = ex_mem_reg.rs2_s_s;
        // write back
        mem_wb_reg.wb_ctrl_s    = ex_mem_reg.wb_ctrl_s;
        mem_wb_reg.dmem_addr_s  = ex_mem_reg.dmem_addr_s; // 32-bit aligned
        mem_wb_reg.mem_addr_s   = ex_mem_reg.mem_addr_s;  // real address
        mem_wb_reg.mem_rmask_s  = ex_mem_reg.dmem_rmask_s;
        mem_wb_reg.mem_wmask_s  = ex_mem_reg.dmem_wmask_s;
        mem_wb_reg.mem_wdata_s  = ex_mem_reg.dmem_wdata_s;
        mem_wb_reg.br_en_s      = ex_mem_reg.br_en_s;
        mem_wb_reg.alu_out_s    = ex_mem_reg.alu_out_s;
        mem_wb_reg.u_imm_s      = ex_mem_reg.u_imm_s;
    end


endmodule