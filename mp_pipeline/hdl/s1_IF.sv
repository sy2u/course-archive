// Main Hardware: Program Counter, Instruction Memory Port
// Function: Instruction Fetch

module IF
import rv32i_types::*;
(
    input   logic               clk,
    input   logic               rst,
    output  logic   [31:0]      imem_addr,
    output  logic   [3:0]       imem_rmask,
    output  if_id_stage_reg_t   if_id_reg
);

    logic [63:0] order, order_next;
    logic [31:0] pc, pc_next;
    logic        valid;

    // update pc
    always_ff @( posedge clk ) begin
        if (rst) begin
            pc <= 32'h1eceb000;
            order <= '0;
        end else begin
            pc <= pc_next;
            order <= order_next;
        end
    end

    // update pc_next
    always_comb begin
        if (rst) begin
            pc_next = pc;
            valid = 1'b0;
        end else begin
            pc_next = pc +'d4;
            valid = 1'b1;
        end
    end

    always_comb begin
        // fetch instruction
        imem_addr = pc_next;
        imem_rmask = '1;
        order_next = order + 'd1;
    end

    // assign signals to the register struct
    always_comb begin
        if_id_reg.pc_s = pc;
        if_id_reg.pc_next_s = pc_next;
        if_id_reg.valid_s = valid;
        if_id_reg.order_s   = order;
    end


endmodule