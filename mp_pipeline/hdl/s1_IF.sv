// Main Hardware: Program Counter, Instruction Memory Port
// Function: Instruction Fetch

module IF
import rv32i_types::*;
(
    input   logic               clk,
    input   logic               rst,

    input   logic               move,
    input   logic               dmem_req,
    output  logic               imem_req,

    output  logic   [31:0]      imem_addr,
    output  logic   [3:0]       imem_rmask,

    output  if_id_stage_reg_t   if_id_reg
);

    logic        valid;
    logic [63:0] order, order_next;
    logic [31:0] pc, pc_next;

    // update pc
    always_ff @( posedge clk ) begin
        if (rst) begin
            pc <= 32'h1eceb000;
            order <= '0;
        end else begin
            if( move ) begin
                pc <= pc_next;
                order <= order_next;
            end
        end
    end

    always_comb begin
        order_next = order + 'd1;
        imem_rmask = '0;
        imem_req = '0;

        if (rst) begin
            pc_next = pc;
        end else begin
            pc_next = pc +'d4;
            if( move ) begin
                imem_rmask = '1;
                if( !dmem_req ) imem_req = '1;
            end
        end
    end

    assign imem_addr = pc_next;

    // assign signals to the register struct
    always_comb begin
        if_id_reg.pc_s = pc;
        if_id_reg.pc_next_s = pc_next;
        if_id_reg.valid_s = valid;
        if_id_reg.order_s = order;
    end


endmodule