package cache_types;

    typedef enum logic [2:0] {
        compare     = 3'b000,
        readmem     = 3'b001,
        writeback   = 3'b010,
        sramstall   = 3'b011,
        writemem    = 3'b100,
        idle        = 3'b101
    } process_state_t;

    typedef enum logic [1:0] {
        none    = 2'b00,
        read    = 2'b01,
        write   = 2'b10
    } req_t;

endpackage