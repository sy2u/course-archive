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

    output  logic           move,
    output  logic           stop_fetch
);

    stall_state  prev_state, curr_state, next_state;

    always_ff @( posedge clk ) begin
        if (rst) begin
            prev_state <= moving;
            curr_state <= wait_imem;
        end else begin
            prev_state <= curr_state;
            curr_state <= next_state;
        end
    end

    always_comb begin
        unique case (curr_state)
            moving: begin
                if( !imem_resp ) next_state = wait_imem;
                // else if( dmem_req && (!imem_req) ) next_state = wait_dmem;
                // else if( dmem_req && imem_req ) next_state = imem_dmem;
                else next_state = moving;
            end
            wait_imem: begin
                if( imem_resp ) begin
                    if( (!dmem_req) && (!imem_req) ) next_state = moving;
                    else if( dmem_req && (!imem_req) ) next_state = wait_dmem;
                    else if( (!dmem_req) && imem_req ) next_state = wait_imem;
                    else next_state = imem_dmem;
                end 
                else next_state = wait_imem;
            end
            wait_dmem: begin
                if( dmem_resp ) begin
                    if( (!dmem_req) && (!imem_req) ) next_state = moving;
                    else if( dmem_req && (!imem_req) ) next_state = wait_dmem;
                    else if( (!dmem_req) && imem_req ) next_state = wait_imem;
                    else next_state = imem_dmem;
                end 
                else next_state = wait_dmem;
            end
            imem_dmem: begin
                if( imem_resp && dmem_resp ) begin
                    if( (!dmem_req) && (!imem_req) ) next_state = moving;
                    else if( dmem_req && (!imem_req) ) next_state = wait_dmem;
                    else if( (!dmem_req) && imem_req ) next_state = wait_imem;
                    else next_state = imem_dmem;
                end else if ( imem_resp ) begin
                    if( imem_req ) next_state = imem_dmem;
                    else next_state = wait_dmem;
                end else if ( dmem_resp ) begin
                    if( dmem_req ) next_state = imem_dmem;
                    else next_state = wait_imem;
                end
                else next_state = imem_dmem;
            end
            default: next_state = curr_state;
        endcase
    end

    always_comb begin
        stop_fetch = 1'b0;
        unique case (curr_state)
            moving: begin
                move = 1'b1;
                if( !imem_resp ) move = 1'b0;
                if( prev_state==wait_imem && next_state==wait_dmem ) stop_fetch = 1'b1; 
                // if( prev == imem_dmem ) stop_fetch = 1'b1;
            end
            wait_imem: begin
                move = 1'b0;
                if( rst || imem_resp ) move = 1'b1;
                if( prev_state==wait_imem && next_state==wait_dmem ) stop_fetch = 1'b1; 
            end
            wait_dmem: begin
                move = 1'b0;
                if( dmem_resp ) move = 1'b1;
            end
            imem_dmem: begin
                move = 1'b0;
                if( imem_resp && dmem_resp ) move = 1'b1;
            end
            default: move = 1'b0;
        endcase
    end

endmodule