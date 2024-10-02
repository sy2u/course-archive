// Main Hardware: Program Counter, Instruction Memory Port
// Function: Instruction Fetch

module IF
import rv32i_types::*;
(
    input   logic               clk,
    input   logic               rst,

    input   logic               move,

    output  logic               imem_req,
    output  logic   [31:0]      imem_addr,
    output  logic   [3:0]       imem_rmask,
    input   logic               imem_resp,
    input   logic   [31:0]      imem_rdata,

    output  if_id_stage_reg_t   if_id_reg,

    // forwarding, stop fecth when load data hazard occur
    input   logic               forward_stall
);

    logic        valid;
    logic [63:0] order, order_next;
    logic [31:0] pc, pc_next;
    logic [31:0] inst_store;

    assign  imem_addr = pc_next;
    assign  order_next = order + 'd1;
    
    // update pc
    always_ff @( posedge clk ) begin
        // pc and order control
        if (rst) begin
            pc <= 32'h1eceb000;
            order <= '0;
        end else begin
            if( move && (!forward_stall) ) begin
                pc <= pc_next;
                order <= order_next;
            end
        end
        // read from memory
        if( imem_resp ) inst_store <= imem_rdata;
    end

    always_comb begin
        if( rst ) begin
            pc_next = pc; 
            valid = 1'b0; 
            imem_req = 1'b1;
            imem_rmask = '1;
        end
        else begin
            pc_next = pc +'d4;
            if( move && (!forward_stall) ) begin
                valid = 1'b1; 
                imem_req = 1'b1;
                imem_rmask = '1;
            end else begin
                valid = 1'b0;
                imem_req = 1'b0;
                imem_rmask = '0;
            end
        end
    end

    // assign signals to the register struct
    always_comb begin
        if(move && imem_resp)   if_id_reg.inst_s = imem_rdata; 
        else                    if_id_reg.inst_s = inst_store;

        if_id_reg.pc_s = pc;
        if_id_reg.pc_next_s = pc_next;
        if_id_reg.valid_s = valid;
        if_id_reg.order_s = order;
    end


endmodule