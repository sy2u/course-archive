// Blank randinst: if you want to use random stimulus, copy over your part 3
// files randinst.svh and instr_cg.svh.
class RandInst;

    covergroup instr_cg;
    endgroup : instr_cg

    rand instr_t instr;

    function new();
        instr_cg = new();
    endfunction : new

endclass : RandInst
