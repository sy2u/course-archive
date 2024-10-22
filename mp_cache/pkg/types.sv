package cache_types;

    // typedef struct packed {
    //     logic   [31:0]      curr_addr;
    //     logic   [23:0]      curr_tag;
    //     logic   [3:0]       curr_set;
    //     logic   [2:0]       curr_offset;
    //     logic   [2:0]       curr_lru;
    //     logic   [3:0]       curr_ufp_rmask;
    //     logic               curr_req;
    // } stage_reg_t;

    typedef enum logic [1:0] {
        compare     = 2'b00,
        readmem     = 2'b01,
        writeback   = 2'b10,
        sramstall   = 2'b11
    } process_state_t;

    typedef enum logic [1:0] {
        none    = 2'b00,
        read    = 2'b01,
        write   = 2'b10
    } req_t;

    // typedef enum logic [1:0] {
    //     way_A = 2'b00,
    //     way_B = 2'b01,
    //     way_C = 2'b10,
    //     way_D = 2'b11
    // } evict_way_t;

endpackage