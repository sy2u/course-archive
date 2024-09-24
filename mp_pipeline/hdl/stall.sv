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

    stall_state curr_state, next_state;

    always_ff @( posedge clk ) begin
        if (rst) begin
            curr_state <= wait_imem;
        end else begin
            curr_state <= next_state;
        end
    end

    always_comb begin
        unique case (curr_state)
            moving: begin
                if( (!dmem_req) && (!imem_req) )    next_state = moving;
                else if( dmem_req && (!imem_req) )  next_state = wait_dmem;
                else if( (!dmem_req) && imem_req )  next_state = wait_imem;
                else                                next_state = imem_dmem;
            end
            wait_imem: begin
                if( imem_resp ) begin
                    if( (!dmem_req) && (!imem_req) )    next_state = moving;
                    else if( dmem_req && (!imem_req) )  next_state = wait_dmem;
                    else if( (!dmem_req) && imem_req )  next_state = wait_imem;
                    else                                next_state = imem_dmem;
                end 
                else next_state = wait_imem;
            end
            wait_dmem: begin
                if( dmem_resp ) begin
                    if( (!dmem_req) && (!imem_req) )    next_state = moving;
                    else if( dmem_req && (!imem_req) )  next_state = wait_dmem;
                    else if( (!dmem_req) && imem_req )  next_state = wait_imem;
                    else                                next_state = imem_dmem;
                end 
                else next_state = wait_dmem;
            end
            imem_dmem: begin
                if( imem_resp && dmem_resp ) begin
                    if( (!dmem_req) && (!imem_req) )    next_state = moving;
                    else if( dmem_req && (!imem_req) )  next_state = wait_dmem;
                    else if( (!dmem_req) && imem_req )  next_state = wait_imem;
                    else                                next_state = imem_dmem;
                end else if ( imem_resp ) begin
                    next_state = wait_dmem;
                end else if ( dmem_resp ) begin
                    next_state = wait_imem;
                end
                else next_state = imem_dmem;
            end
            default: next_state = curr_state;
        endcase
    end

    always_comb begin
        unique case (curr_state)
            moving: move = 1'b1;
            wait_imem: begin
                if( rst ) move = 1'b1;
                else move = 1'b0;
            end
            default: move = 1'b0;
        endcase
    end

endmodule