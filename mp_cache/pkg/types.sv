package cache_types;

    typedef struct packed {
        logic   [31:0]      curr_addr;
        logic   [23:0]      curr_tag;
        logic   [3:0]       curr_set;
        logic   [2:0]       curr_offset;
        logic   [2:0]       curr_lru;
    } stage_reg_t;

    // typedef enum logic [1:0] {
    //     way_A = 2'b00,
    //     way_B = 2'b01,
    //     way_C = 2'b10,
    //     way_D = 2'b11
    // } evict_way_t;

endpackage