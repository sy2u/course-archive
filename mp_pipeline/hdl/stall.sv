// FSM for stall control

module stall
import rv32i_types::*;
(
    input   logic           clk,
    input   logic           rst,

    input   logic           imem_req,
    input   logic           dmem_req,
    input   logic           imem_resp,
    input   logic           dmem_resp,

    output  logic           move
);

    stall_state curr_state, next_state, prep_state;

    always_ff @( posedge clk ) begin
        if (rst) begin
            curr_state <= idle;
        end else begin
            curr_state <= next_state;
        end
    end

    always_comb begin
        if( dmem_req && (!imem_req) )       prep_state = wait_dmem;
        else if( (!dmem_req) && imem_req )  prep_state = wait_imem;
        else if( dmem_req && imem_req )     prep_state = imem_dmem;
        else                                prep_state = moving;
    end

    always_comb begin
        unique case (curr_state)
            idle: begin
                if( !rst )                              next_state = wait_imem;
                else                                    next_state = idle;
            end
            wait_imem: begin
                if( imem_resp )                         next_state = prep_state;
                else                                    next_state = wait_imem;
            end
            wait_dmem: begin
                if( dmem_resp )                         next_state = prep_state;
                else                                    next_state = wait_dmem;
            end
            imem_dmem: begin
                if( imem_resp && dmem_resp )            next_state = prep_state;
                else if ( imem_resp )                   next_state = wait_dmem;
                else if ( dmem_resp )                   next_state = wait_imem;
                else                                    next_state = imem_dmem;
            end
            moving:                                     next_state = prep_state;
            default:                                    next_state = curr_state;
        endcase
    end

    always_comb begin
        move = 1'b0;
        unique case (curr_state)
            idle        : if(!rst)                  move = 1'b1;
            wait_imem   : if(imem_resp)             move = 1'b1;
            wait_dmem   : if(dmem_resp)             move = 1'b1;
            imem_dmem   : if(imem_resp&&dmem_resp)  move = 1'b1;
            moving      :                           move = 1'b1;
            default:                                move = 1'b0;
        endcase
    end

endmodule