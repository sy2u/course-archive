// Approach: Remember the most recently used way

module lru_ctrl 
import cache_types::*;
(
    input   logic   [2:0]   curr_lru,
    output  logic   [2:0]   next_lru,
    input   logic   [1:0]   access_way,
    output  logic   [1:0]   evict_way
);

    always_comb begin
        // if( evict ) begin
        //     unique case (curr_lru)
        //         3'b000:     next_lru = 3'b101;
        //         3'b001:     next_lru = 3'b100;
        //         3'b010:     next_lru = 3'b111;
        //         3'b011:     next_lru = 3'b110;
        //         3'b100:     next_lru = 3'b010;
        //         3'b101:     next_lru = 3'b011;
        //         3'b110:     next_lru = 3'b000;
        //         3'b111:     next_lru = 3'b001;
        //         default:    next_lru = 'x;
        //     endcase
        // end else begin
        next_lru = curr_lru;
        unique case (access_way)
            2'b00: begin
                next_lru[0] = 1'b0;
                next_lru[1] = 1'b0;
            end
            2'b01: begin
                next_lru[0] = 1'b0;
                next_lru[1] = 1'b1;
            end
            2'b10: begin
                next_lru[0] = 1'b1;
                next_lru[2] = 1'b0;
            end
            2'b11: begin
                next_lru[0] = 1'b1;
                next_lru[2] = 1'b1;
            end
            default:;
        endcase
        // end
    end

    always_comb begin
        unique case (curr_lru)
            3'b000:     evict_way = 2'd3; // way_D;
            3'b001:     evict_way = 2'd2; // way_C;
            3'b010:     evict_way = 2'd3; // way_D;
            3'b011:     evict_way = 2'd2; // way_C;
            3'b100:     evict_way = 2'd1; // way_B;
            3'b101:     evict_way = 2'd1; // way_B;
            3'b110:     evict_way = 2'd0; // way_A;
            3'b111:     evict_way = 2'd0; // way_A;
            default:    evict_way = 'x;
        endcase
    end


endmodule