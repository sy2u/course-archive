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

    req_t               new_req;
    logic               update_array, queued;
    logic   [3:0]       set_in;
    logic   [22:0]      tag_in;
    logic   [2:0]       offset_in;

    always_comb begin
        new_req = none;
        if(ufp_rmask != '0) new_req = read;
        if(ufp_wmask != '0) new_req = write;
        queued = (new_req!=none) && update_array;
        set_in = ufp_addr[8:5];
        tag_in = ufp_addr[31:9];
        offset_in = ufp_addr[4:2];
    end

//////////////////////////////////////////////////////////
///                   STAGE TRANSITION                 ///
//////////////////////////////////////////////////////////

    req_t               curr_req;
    logic               curr_queued;
    logic   [31:0]      curr_addr, curr_wdata;
    logic   [22:0]      curr_tag;
    logic   [3:0]       curr_set, curr_ufp_wmask;
    logic   [2:0]       curr_offset;
    logic               stall;
    
    always_ff @( posedge clk ) begin
        if( rst ) begin
            curr_req        <= none;
            curr_ufp_wmask  <= '0;
        end else if( ~stall ) begin
            curr_ufp_wmask  <= ufp_wmask;
            curr_wdata      <= ufp_wdata;
            curr_req        <= new_req;
            curr_addr       <= ufp_addr;
            curr_tag        <= tag_in;
            curr_set        <= set_in;
            curr_offset     <= offset_in;
            curr_queued     <= queued;
        end
    end


//////////////////////////////////////////////////////////
///                    PROCESS STAGE                   ///
//////////////////////////////////////////////////////////

    process_state_t     curr_state, next_state;
    logic               hit, evict;
    logic   [1:0]       evict_way, access_way;

    // state transition
    always_ff @( posedge clk ) begin
        if( rst ) begin
            curr_state <= idle;
        end else begin
            curr_state <= next_state;
        end
    end

    always_comb begin
        next_state = curr_state;
        unique case (curr_state)
            idle:                                               next_state = compare;
            compare:    if      ( evict )                       next_state = writemem;
                        else if ((curr_req!=none) && (~hit))    next_state = readmem;
                        else if ((curr_req==write) && hit)      next_state = sramstall;
            writemem:   if      ( dfp_resp )                    next_state = readmem;
            readmem:    if      ( dfp_resp )                    next_state = writeback;
            writeback:  if      ((curr_req==read) && hit)       next_state = compare;
                        else if ((curr_req==write) && hit)      next_state = sramstall;
            sramstall:                                          next_state = compare;
            default:;
        endcase
    end

    // signal control
    always_comb begin
        for( int i = 0; i < 4; i++ ) begin
            web0[i] = '1;
            tag_array_wr[i] = 'x;
            data_array_wr[i] = 'x;
            data_array_wmask[i] = '0;
        end
        dfp_addr = 'x;
        dfp_write = '0;
        dfp_wdata = 'x;
        update_array = '0;
        ufp_resp = '0;
        ufp_rdata = 'x;
        dfp_read = '0;
        evict = '0;
        stall = '0;
        unique case (curr_state)
            compare: begin
                if( curr_req!=none ) begin
                    stall = '1;
                    if( hit ) begin
                        stall = '0;
                        if( curr_req == read ) begin
                            ufp_resp = '1; 
                            ufp_rdata = data_array_re[access_way][32*curr_offset+:32];
                        end else if ( curr_req == write ) begin
                            ufp_resp = '1;
                            update_array = '1;
                            web0[access_way] = '0;
                            tag_array_wr[access_way] = {1'b1,curr_tag};
                            data_array_wr[access_way][32*curr_offset+:32] = curr_wdata;
                            data_array_wmask[access_way] = 32'(curr_ufp_wmask << curr_addr[4:0]);
                        end
                    end else begin
                        if( tag_array_re[evict_way][23] & valid_array_re[evict_way] ) begin
                            evict = '1;
                            dfp_write = '1;
                            dfp_addr = {tag_array_re[evict_way][22:0],curr_set,5'd0};
                            dfp_wdata = data_array_re[evict_way];
                        end else begin
                            dfp_addr = {curr_addr[31:5], 5'd0};;
                            dfp_read = '1;
                        end
                    end
                end
            end
            writemem: begin
                stall = '1;
                dfp_write = '1;
                dfp_addr = {tag_array_re[evict_way][22:0],curr_set,5'd0};
                dfp_wdata = data_array_re[evict_way];
            end
            readmem: begin
                stall = '1;
                dfp_addr = {curr_addr[31:5], 5'd0};
                dfp_read = '1;
                if( dfp_resp ) begin
                    update_array = '1;
                    web0[evict_way] = '0;
                    tag_array_wr[evict_way] = {1'b0,curr_tag};
                    data_array_wr[evict_way] = dfp_rdata;
                    data_array_wmask[evict_way] = '1;       // replace the entire cache line
                end
            end
            writeback: begin
                stall = '1;
                if( hit ) begin
                    stall = '0;
                    if( curr_req == read ) begin
                        ufp_resp = '1; 
                        ufp_rdata = data_array_re[access_way][32*curr_offset+:32];
                    end else if (curr_req == write) begin
                        ufp_resp = '1;
                        update_array = '1;
                        web0[access_way] = '0;
                        tag_array_wr[access_way] = {1'b1,curr_tag};
                        data_array_wr[access_way][32*curr_offset+:32] = curr_wdata;
                        data_array_wmask[access_way] = 32'(curr_ufp_wmask << curr_addr[4:0]);
                    end
                end
            end
            sramstall: if( curr_req!=none ) stall = '1;
            default:;
        endcase
    end
            
    // hit detection
    always_comb begin
        hit = '0;
        access_way = 'x;
        if( curr_req != none ) begin
            for( int i = 0; i < 4; i++ ) begin
                if( valid_array_re[i] ) begin
                    if( curr_tag == tag_array_re[i][22:0] ) begin
                        hit = '1;
                        access_way = 2'(unsigned'(i));   
                    end
                end
            end
        end
    end


//////////////////////////////////////////////////////////
///                    CACHE MEMORY                    ///
//////////////////////////////////////////////////////////

    logic   [3:0]   addr;
    logic           csb0, csb1;

    always_comb begin
        if( update_array | curr_queued ) begin
            addr = curr_set;
        end else begin
            addr = set_in;
        end
    end

    assign csb1 = ~ufp_resp;
    assign csb0 = ~((~stall)|update_array|curr_queued);

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
        .csb0       (stall), 
        .web0       (1'b1),     // read current lru
        .addr0      (set_in),
        .din0       ('0),
        .dout0      (curr_lru),
        .csb1       (csb1),
        .web1       (1'b0),     // update next lru
        .addr1      (curr_set),
        .din1       (next_lru),
        .dout1      (lru_dout1)
    );

endmodule
