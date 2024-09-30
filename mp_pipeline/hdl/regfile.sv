// Imported from mp_verif

module regfile
import rv32i_types::*;
(
    input   logic               clk,
    input   logic               rst,
    input   logic               regf_we,
    input   logic   [31:0]      rd_v,
    input   logic   [4:0]       rs1_s, rs2_s, rd_s,
    output  logic   [31:0]      rs1_v, rs2_v,

    input   decode_fw_sel_t     fowarDe
);

    logic   [31:0]  data [32];

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < 32; i++) begin
                data[i] <= '0;
            end
        end else if (regf_we && (rd_s != 5'd0)) begin
            data[rd_s] <= rd_v;
        end
    end

    always_comb begin
        if (rst) begin
            rs1_v = 'x;
            rs2_v = 'x;
        end else begin
            rs1_v = (rs1_s != 5'd0) ? data[rs1_s] : '0;
            rs2_v = (rs2_s != 5'd0) ? data[rs2_s] : '0;
            // forwarding
            if( fowarDe == rs1_f )           rs1_v = rd_v;
            else if ( fowarDe == rs2_f )     rs2_v = rd_v;
        end
    end

endmodule : regfile