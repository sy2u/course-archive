module CMP
import rv32i_types::*;
(
    input   logic   [2:0]   cmpop,
    input   logic   [31:0]  a,
    input   logic   [31:0]  b,
    output  logic           br_en
);

    logic   signed      [31:0]  as, bs;
    logic   unsigned    [31:0]  au, bu;

    assign as =   signed'(a);
    assign bs =   signed'(b);
    assign au = unsigned'(a);
    assign bu = unsigned'(b);

    always_comb begin
        unique case (cmpop)
            cmp_op_beq : br_en = (au == bu);
            cmp_op_bne : br_en = (au != bu);
            cmp_op_blt : br_en = (as <  bs);
            cmp_op_bge : br_en = (as >=  bs);
            cmp_op_bltu: br_en = (au <  bu);
            cmp_op_bgeu: br_en = (au >=  bu);
            default    : br_en = 1'bx;
        endcase
    end

endmodule : CMP