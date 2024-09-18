package rv32i_types;

//////////////////////////////////////////////////////
// Merge what is in mp_verif/pkg/types.sv over here //
//////////////////////////////////////////////////////

    typedef enum logic [6:0] {
        op_b_lui       = 7'b0110111, // load upper immediate (U type)
        op_b_auipc     = 7'b0010111, // add upper immediate PC (U type)
        op_b_jal       = 7'b1101111, // jump and link (J type)
        op_b_jalr      = 7'b1100111, // jump and link register (I type)
        op_b_br        = 7'b1100011, // branch (B type)
        op_b_load      = 7'b0000011, // load (I type)
        op_b_store     = 7'b0100011, // store (S type)
        op_b_imm       = 7'b0010011, // arith ops with register/immediate operands (I type)
        op_b_reg       = 7'b0110011  // arith ops with register operands (R type)
    } rv32i_opcode;

    // typedef enum logic [2:0] {
    //     arith_f3_add   = 3'b000, // check logic 30 for sub if op_reg op
    //     arith_f3_sll   = 3'b001,
    //     arith_f3_slt   = 3'b010,
    //     arith_f3_sltu  = 3'b011,
    //     arith_f3_xor   = 3'b100,
    //     arith_f3_sr    = 3'b101, // check logic 30 for logical/arithmetic
    //     arith_f3_or    = 3'b110,
    //     arith_f3_and   = 3'b111
    // } arith_f3_t;

    typedef enum logic [2:0] {
        load_f3_lb     = 3'b000,
        load_f3_lh     = 3'b001,
        load_f3_lw     = 3'b010,
        load_f3_lbu    = 3'b100,
        load_f3_lhu    = 3'b101
    } load_f3_t;

    typedef enum logic [2:0] {
        store_f3_sb    = 3'b000,
        store_f3_sh    = 3'b001,
        store_f3_sw    = 3'b010
    } store_f3_t;

    // typedef enum logic [2:0] {
    //     branch_f3_beq  = 3'b000,
    //     branch_f3_bne  = 3'b001,
    //     branch_f3_blt  = 3'b100,
    //     branch_f3_bge  = 3'b101,
    //     branch_f3_bltu = 3'b110,
    //     branch_f3_bgeu = 3'b111
    // } branch_f3_t;

    typedef enum logic [2:0] {
        cmp_op_beq  = 3'b000,
        cmp_op_bne  = 3'b001,
        cmp_op_blt  = 3'b100,
        cmp_op_bge  = 3'b101,
        cmp_op_bltu = 3'b110,
        cmp_op_bgeu = 3'b111
    } cmp_ops_t;

    typedef enum logic [2:0] {
        alu_op_add     = 3'b000,
        alu_op_sll     = 3'b001,
        alu_op_sra     = 3'b010,
        alu_op_sub     = 3'b011,
        alu_op_xor     = 3'b100,
        alu_op_srl     = 3'b101,
        alu_op_or      = 3'b110,
        alu_op_and     = 3'b111
    } alu_ops_t;

///////////
// Muxes //
///////////
    typedef enum logic {
        rs1_out = 1'b0,
        pc_out  = 1'b1
    } alu_m1_sel_t;

    typedef enum logic {
        rs2_out = 2'b00,
        u_imm_m = 2'b01,
        i_imm_m = 2'b10, 
        const4  = 2'b11
    } alu_m2_sel_t;

    typedef enum logic {
        rs2_out = 1'b0,
        i_imm_m = 1'b1
    } cmp_m_sel_t;

    typedef enum logic {
        u_imm_m     = 3'd0, 
        alu_out     = 3'd1,
        ext_br      = 3'd2,
        lb          = 3'd3,
        lbu         = 3'd4,
        lh          = 3'd5,
        lhu         = 3'd6,
        lw          = 3'd7
    } rd_m_sel_t;

////////////////////
// Control Words //
//////////////////
    typedef struct packed {
        alu_m1_sel_t    alu_m1_sel;
        alu_m2_sel_t    alu_m2_sel;
        alu_ops_t       aluop;
        cmp_m_sel_t     cmp_sel;
        cmp_ops_t       cmpop;
    } ex_ctrl_t;

    typedef struct packed {
        logic       [2:0]   funct3;
        logic               mem_re;
        logic               mem_we;
    } mem_ctrl_t;

    typedef struct packed {
        logic               regf_we;
        rd_m_sel_t          rd_m_sel;
    } wb_ctrl_t;

/////////////////////
// Stage registers //
/////////////////////
    typedef struct packed {
        logic   [31:0]      pc;
        logic   [63:0]      order;
    } if_id_stage_reg_t;

    typedef struct packed {
        // data
        logic   [31:0]      inst_s;
        logic   [31:0]      pc_s;
        logic   [63:0]      order_s;
        logic   [31:0]      u_imm_s;
        logic   [31:0]      s_imm_s;
        logic   [31:0]      i_imm_s;
        logic   [31:0]      rs1_v_s;
        logic   [31:0]      rs2_v_s;
        logic   [31:0]      rd_s_s;
        // control
        ex_ctrl_t           ex_ctrl_s;
        mem_ctrl_t          mem_ctrl_s;
        wb_ctrl_t           wb_ctrl_s;
    } id_ex_stage_reg_t;

    typedef struct packed {
        // data
        logic   [31:0]      inst_s;
        logic   [31:0]      pc_s;
        logic   [63:0]      order_s;
        logic               br_en_s;
        logic   [31:0]      u_imm_s;
        logic   [31:0]      alu_out_s;
        logic   [31:0]      rs2_v_s;
        logic   [31:0]      rd_s;
        // control
        mem_ctrl_t          mem_ctrl_s;
        wb_ctrl_t           wb_ctrl_s;
    } ex_mem_stage_reg_t;

    typedef struct packed {
        // data: monitor
        logic   [31:0]      inst_s;
        logic   [31:0]      pc_s;
        logic   [63:0]      order_s;
        // data for reg
        logic               br_en_s;
        logic   [31:0]      u_imm_s;
        logic   [31:0]      alu_out_s;
        logic   [31:0]      rd_s_s;
        logic   [31:0]      dmem_addr_s;
        wb_ctrl_t           wb_ctrl_s;
    } mem_wb_stage_reg_t;

endpackage
