module lfsr #(
    parameter bit   [15:0]  SEED_VALUE = 'hECEB
) (
    input   logic           clk,
    input   logic           rst,
    input   logic           en,
    output  logic           rand_bit,
    output  logic   [15:0]  shift_reg
);

    // TODO: Fill this out!

    always_ff @(posedge clk) begin
        if(rst) begin
            shift_reg <= SEED_VALUE;
        end
        if(en) begin
            rand_bit <= shift_reg[0];
            shift_reg <= {shift_reg[2]^shift_reg[3]^shift_reg[5]^shift_reg[0], shift_reg[15:1]};
        end
    end

endmodule : lfsr
