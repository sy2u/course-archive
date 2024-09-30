module forward
import rv32i_types::*;
(
    input   id_ex_stage_reg_t   id_ex_reg,
    input   ex_mem_stage_reg_t  ex_mem_reg,
    input   mem_wb_stage_reg_t  mem_wb_reg,

    output  forward_sel_t       forwardA,
    output  forward_sel_t       forwardB
);

always_comb begin
    forwardA = none;
    forwardB = none;
    if( ex_mem_reg.wb_ctrl_s.regf_we && (ex_mem_reg.rd_s_s!=0) )begin
        if( ex_mem_reg.rd_s_s == id_ex_reg.rs1_s_s ) forwardA = mem_ex;
        if( mem_wb_reg.rd_s_s == id_ex_reg.rs1_s_s ) forwardA = wb_ex;
        if( ex_mem_reg.rd_s_s == id_ex_reg.rs2_s_s ) forwardB = mem_ex;
        if( mem_wb_reg.rd_s_s == id_ex_reg.rs2_s_s ) forwardB = wb_ex;
    end
end

endmodule