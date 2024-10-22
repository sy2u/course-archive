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
    logic               new_req, update_array;
    logic   [3:0]       next_set;
    logic   [23:0]      next_tag;
    logic   [2:0]       next_offset;

    assign csb0 = !(new_req || update_array); // active low

    always_comb begin
        new_req = '0;

        if( ufp_rmask || ufp_wmask ) new_req = '1;
        next_set = ufp_addr[8:5];
        next_tag = {1'b0, ufp_addr[31:9]};
        next_offset = ufp_addr[4:2];
    end


//////////////////////////////////////////////////////////
///                   STAGE TRANSITION                 ///
//////////////////////////////////////////////////////////

    logic               curr_req;
    logic   [31:0]      curr_addr;
    logic   [23:0]      curr_tag;
    logic   [3:0]       curr_set, curr_ufp_rmask;
    logic   [2:0]       curr_offset;
    
    always_ff @( posedge clk ) begin
        if( rst ) begin
            curr_ufp_rmask <= '0;
            curr_req <= '0;
        end else if( ufp_resp || ufp_rmask ) begin
            curr_addr <= ufp_addr;
            curr_tag <= next_tag;
            curr_set <= next_set;
            curr_offset <= next_offset;
            curr_ufp_rmask <= ufp_rmask;
            curr_req <= new_req;
        end
    end

//////////////////////////////////////////////////////////
///                    PROCESS STAGE                   ///
//////////////////////////////////////////////////////////

    process_state_t     curr_state, next_state;
    logic               hit, csb1;
    logic   [1:0]       evict_way, access_way;

    assign csb1 = !ufp_resp;  // active low, update lru when new hit occurs

    // state transition
    always_ff @( posedge clk ) begin
        if( rst ) begin
            curr_state <= compare;
        end else begin
            curr_state <= next_state;
        end
    end

    always_comb begin
        next_state = curr_state;
        unique case (curr_state)
            compare: if( curr_req && (!hit) ) next_state = readmem;
            readmem: if( dfp_resp ) next_state = write;
            write:   if( hit ) next_state = compare;
            default:;
        endcase
    end

    // signal control
    always_comb begin
        for( int i = 0; i < 4; i++ ) begin
            web0[i] = '1;
            tag_array_wr[i] = 'x;
            data_array_wr[i] = 'x;
        end
        dfp_addr = 'x;
        dfp_write = '0;
        dfp_wdata = 'x;
        update_array = '0;
        ufp_resp = '0;
        ufp_rdata = 'x;
        dfp_read = '0;
        unique case (curr_state)
            compare: begin
                if( curr_req ) begin
                    if( hit ) begin
                        ufp_resp = '1; 
                        ufp_rdata = data_array_re[access_way][32*curr_offset+:32];
                    end else begin
                        dfp_addr = curr_addr;
                        dfp_read = '1;
                    end
                end
            end
            readmem: begin
                dfp_addr = curr_addr;
                dfp_read = '1;
                if( dfp_resp ) begin
                    update_array = '1;
                    web0[evict_way] = '0;
                    tag_array_wr[evict_way] = curr_tag;
                    data_array_wr[evict_way] = dfp_rdata;
                    data_array_wmask[evict_way] = '1;       // replace the entire cache line
                end
            end
            write: begin
                if( hit ) begin
                    ufp_resp = '1; 
                    ufp_rdata = data_array_re[access_way][32*curr_offset+:32];
                end
            end
            default:;
        endcase
    end
            
    // hit detection
    always_comb begin
        hit = '0;
        access_way = 'x;
        if( curr_req ) begin
            for( int i = 0; i < 4; i++ ) begin
                if( valid_array_re[i] ) begin
                    if( curr_tag == tag_array_re[i] ) begin
                        hit = '1;
                        access_way = 2'(i);        
                    end
                end
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
        .access_way(access_way),
        .evict_way(evict_way)
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
