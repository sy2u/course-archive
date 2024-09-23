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

    typedef enum logic [2:0] {
        arith_f3_add   = 3'b000, // check logic 30 for sub if op_reg op
        arith_f3_sll   = 3'b001,
        arith_f3_slt   = 3'b010,
        arith_f3_sltu  = 3'b011,
        arith_f3_xor   = 3'b100,
        arith_f3_sr    = 3'b101, // check logic 30 for logical/arithmetic
        arith_f3_or    = 3'b110,
        arith_f3_and   = 3'b111
    } arith_f3_t;

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

    typedef enum logic [2:0] {
        branch_f3_beq  = 3'b000,
        branch_f3_bne  = 3'b001,
        branch_f3_blt  = 3'b100,
        branch_f3_bge  = 3'b101,
        branch_f3_bltu = 3'b110,
        branch_f3_bgeu = 3'b111
    } branch_f3_t;

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
    typedef enum logic [1:0] {
        rs1_out = 2'b00,
        pc_out  = 2'b01,
        invalid_alu_m1 = 2'b10
    } alu_m1_sel_t;

    typedef enum logic [2:0] {
        rs2_out = 3'b000,
        u_imm_m = 3'b001,
        i_imm_m = 3'b010, 
        s_imm_m = 3'b011,
        const4  = 3'b100,
        invalid_alu_m2 = 3'b101
    } alu_m2_sel_t;

    typedef enum logic [1:0] {
        rs2_out_cmp = 2'b00,
        i_imm_m_cmp = 2'b01,
        invalid_cmp = 2'b10
    } cmp_m_sel_t;

    typedef enum logic [3:0] {
        u_imm_m_rd  = 4'b0000, 
        alu_out_rd  = 4'b0001,
        ext_br      = 4'b0010,
        lb          = 4'b0011,
        lbu         = 4'b0100,
        lh          = 4'b0101,
        lhu         = 4'b0110,
        lw          = 4'b0111,
        invalid_rd  = 4'b1000
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
        logic   [31:0]      pc_s;
        logic   [31:0]      pc_next_s;
        logic               valid_s;
        logic   [63:0]      order_s;
    } if_id_stage_reg_t;

    typedef struct packed {
        // data: monitor
        logic   [31:0]      inst_s;
        logic   [31:0]      pc_s;
        logic   [31:0]      pc_next_s;
        logic   [63:0]      order_s;
        logic               valid_s;
        logic   [4:0]       rs1_s_s;
        logic   [4:0]       rs2_s_s;
        // data
        logic   [31:0]      u_imm_s;
        logic   [31:0]      s_imm_s;
        logic   [31:0]      i_imm_s;
        logic   [4:0]       rd_s_s;
        // control
        ex_ctrl_t           ex_ctrl_s;
        mem_ctrl_t          mem_ctrl_s;
        wb_ctrl_t           wb_ctrl_s;
    } id_ex_stage_reg_t;

    typedef struct packed {
        // data: monitor
        logic   [31:0]      inst_s;
        logic   [31:0]      pc_s;
        logic   [31:0]      pc_next_s;
        logic   [63:0]      order_s;
        logic               valid_s;
        logic   [4:0]       rs1_s_s;
        logic   [4:0]       rs2_s_s;
        logic   [31:0]      rs1_v_s;
        // data
        logic               br_en_s;
        logic   [31:0]      u_imm_s;
        logic   [31:0]      alu_out_s;
        logic   [31:0]      rs2_v_s;
        logic   [4:0]       rd_s_s;
        // control
        mem_ctrl_t          mem_ctrl_s;
        wb_ctrl_t           wb_ctrl_s;
    } ex_mem_stage_reg_t;

    typedef struct packed {
        // data: monitor
        logic   [31:0]      inst_s;
        logic   [31:0]      pc_s;
        logic   [31:0]      pc_next_s;
        logic   [63:0]      order_s;
        logic               valid_s;
        logic   [4:0]       rs1_s_s;
        logic   [4:0]       rs2_s_s;
        logic   [31:0]      rs1_v_s;
        logic   [31:0]      rs2_v_s;
        logic   [3:0]       mem_rmask_s;
        logic   [3:0]       mem_wmask_s;
        logic   [31:0]      mem_wdata_s;
        // data for reg
        logic               br_en_s;
        logic   [31:0]      u_imm_s;
        logic   [31:0]      alu_out_s;
        logic   [4:0]       rd_s_s;
        logic   [31:0]      dmem_addr_s;
        logic   [31:0]      mem_addr_s;
        wb_ctrl_t           wb_ctrl_s;
    } mem_wb_stage_reg_t;

////////////////////////////
// imported for random tb //
////////////////////////////
    typedef union packed {
        logic [31:0] word;

        struct packed {
            logic [11:0] i_imm;
            logic [4:0]  rs1;
            logic [2:0]  funct3;
            logic [4:0]  rd;
            rv32i_opcode opcode;
        } i_type;

        struct packed {
            logic [6:0]  funct7;
            logic [4:0]  rs2;
            logic [4:0]  rs1;
            logic [2:0]  funct3;
            logic [4:0]  rd;
            rv32i_opcode opcode;
        } r_type;

        struct packed {
            logic [11:5] imm_s_top;
            logic [4:0]  rs2;
            logic [4:0]  rs1;
            logic [2:0]  funct3;
            logic [4:0]  imm_s_bot;
            rv32i_opcode opcode;
        } s_type;


        struct packed {
            logic           imm_12;
            logic [10:5]    imm_10_5;
            logic [4:0]     rs2;
            logic [4:0]     rs1;
            logic [2:0]     funct3;
            logic [4:1]     imm_4_1;
            logic           imm_11;
            rv32i_opcode    opcode;
        } b_type;

        struct packed {
            logic [31:12] imm;
            logic [4:0]   rd;
            rv32i_opcode  opcode;
        } j_type;

    } instr_t;

    typedef enum logic [6:0] {
        base           = 7'b0000000,
        variant        = 7'b0100000
    } funct7_t;

endpackage
