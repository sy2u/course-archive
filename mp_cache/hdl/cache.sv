module cache 
import cache_types::*;
(
    input   logic           clk,
    input   logic           rst,

    // cpu side signals, ufp -> upward facing port
    input   logic   [31:0]  ufp_addr,
    input   logic   [3:0]   ufp_rmask,
    input   logic   [3:0]   ufp_wmask,
    output  logic   [31:0]  ufp_rdata,
    input   logic   [31:0]  ufp_wdata,
    output  logic           ufp_resp,

    // memory side signals, dfp -> downward facing port
    output  logic   [31:0]  dfp_addr,
    output  logic           dfp_read,
    output  logic           dfp_write,
    input   logic   [255:0] dfp_rdata,
    output  logic   [255:0] dfp_wdata,
    input   logic           dfp_resp
);

            logic               valid_array_re  [4];
            logic   [255:0]     data_array_re   [4];
            logic   [255:0]     data_array_wr   [4];
            logic   [23:0]      tag_array_re    [4];
            logic   [23:0]      tag_array_wr    [4];
            logic   [31:0]      data_array_wmask[4];
            logic               web0            [4];

//////////////////////////////////////////////////////////
///                    DECODE STAGE                    ///
//////////////////////////////////////////////////////////

    logic               csb0;
    logic               read_data, update_array;
    logic   [3:0]       next_set;
    logic   [23:0]      next_tag;
    logic   [2:0]       next_offset;

    assign csb0 = (!read_data) && (!update_array); // active low

    always_comb begin
        next_set = ufp_addr[8:5];
        next_tag = {1'b0, ufp_addr[31:9]};
        next_offset = ufp_addr[4:2];

        read_data = '0;
        if( (ufp_rmask!=0) || (ufp_wmask!=0) ) read_data = '1;
    end


//////////////////////////////////////////////////////////
///                   STAGE TRANSITION                 ///
//////////////////////////////////////////////////////////

    stage_reg_t         stage_reg;
    
    always_ff @( posedge clk ) begin
        if( ufp_rmask ) begin
            stage_reg.curr_addr <= ufp_addr;
            stage_reg.curr_tag <= next_tag;
            stage_reg.curr_set <= next_set;
            stage_reg.curr_offset <= next_offset;
        end
    end

//////////////////////////////////////////////////////////
///                    PROCESS STAGE                   ///
//////////////////////////////////////////////////////////
    
    logic               hit, csb1;
    logic   [31:0]      curr_addr;
    logic   [23:0]      curr_tag;
    logic   [3:0]       curr_set;
    logic   [2:0]       curr_offset;
    logic   [1:0]       evict_idx;

    assign csb1 = !hit;  // active low, update lru when hit occurs

    always_comb begin
        curr_addr = stage_reg.curr_addr;
        curr_tag = stage_reg.curr_tag;
        curr_set = stage_reg.curr_set;
        curr_offset = stage_reg.curr_offset;
    end
            
    // hit detection
    always_comb begin
        ufp_resp = '0;
        ufp_rdata = 'x;
        hit = '0;
        for( int i = 0; i < 4; i++ ) begin
            if( valid_array_re[i] ) begin
                if( curr_tag == tag_array_re[i] ) begin
                    hit = '1;
                    ufp_resp = '1;
                    ufp_rdata = data_array_re[i][32*curr_offset+:32];
                end
            end
        end
    end

    // miss - access dfp and write back
    always_comb begin
        for( int i = 0; i < 4; i++ ) begin
            web0[i] = '1;
            tag_array_wr[i] = 'x;
            data_array_wr[i] = 'x;
        end
        dfp_addr = 'x;
        dfp_read = '0;
        dfp_write = '0;
        dfp_wdata = 'x;
        update_array = '0;
        
        if( !hit ) begin
            // miss control
            dfp_addr = curr_addr;
            dfp_read = '1;
            if( dfp_resp ) begin
                dfp_read = '0;
                // write back to cache arrays
                update_array = '1;
                web0[evict_idx] = '0;
                tag_array_wr[evict_idx] = curr_tag;
                data_array_wr[evict_idx] = dfp_rdata;
                data_array_wmask[evict_idx] = '1;       // replace the entire cache line
            end
        end
    end


//////////////////////////////////////////////////////////
///                    CACHE MEMORY                    ///
//////////////////////////////////////////////////////////

    logic   [3:0]   addr;

    always_comb begin
        if( update_array ) begin
            addr = curr_set;
        end else begin
            addr = next_set;
        end
    end

    generate for (genvar i = 0; i < 4; i++) begin : arrays
        mp_cache_data_array data_array (
            .clk0       (clk),
            .csb0       (csb0),
            .web0       (web0[i]),
            .wmask0     (data_array_wmask[i]),
            .addr0      (addr),
            .din0       (data_array_wr[i]),
            .dout0      (data_array_re[i])
        );
        mp_cache_tag_array tag_array (
            .clk0       (clk),
            .csb0       (csb0),
            .web0       (web0[i]),
            .addr0      (addr),
            .din0       (tag_array_wr[i]),
            .dout0      (tag_array_re[i])
        );
        valid_array valid_array (
            .clk0       (clk),
            .rst0       (rst),
            .csb0       (csb0),
            .web0       (web0[i]),
            .addr0      (addr),
            .din0       (1'b1),
            .dout0      (valid_array_re[i])
        );
    end endgenerate

//////////////////////////////////////////////////////////
///                    LRU CONTROL                     ///
//////////////////////////////////////////////////////////

    logic   [2:0]   lru_dout1;
    logic   [2:0]   curr_lru, next_lru;


    lru_ctrl lru_ctrl (
        .curr_lru(curr_lru),
        .next_lru(next_lru),
        .evict_idx(evict_idx)
    );

    lru_array lru_array (
        .clk0       (clk),
        .rst0       (rst),
        .csb0       (csb0), 
        .web0       (1'b1),     // read current lru
        .addr0      (next_set),
        .din0       ('0),
        .dout0      (curr_lru),
        .csb1       (csb1),
        .web1       (1'b0),     // update next lru
        .addr1      (curr_set),
        .din1       (next_lru),
        .dout1      (lru_dout1)
    );

endmodule
