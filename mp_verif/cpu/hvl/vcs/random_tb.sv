//-----------------------------------------------------------------------------
// Title                 : random_tb
// Project               : ECE 411 mp_verif
//-----------------------------------------------------------------------------
// File                  : random_tb.sv
// Author                : ECE 411 Course Staff
//-----------------------------------------------------------------------------
// IMPORTANT: If you don't change the random seed, every time you do a `make run`
// you will run the /same/ random test. SystemVerilog calls this "random stability",
// and it's to ensure you can reproduce errors as you try to fix the DUT. Make sure
// to change the random seed or run more instructions if you want more extensive
// coverage.
//------------------------------------------------------------------------------
module random_tb
import rv32i_types::*;
(
    mem_itf_w_mask.mem itf
);

    `include "../../hvl/vcs/randinst.svh"

    RandInst gen = new();

    // Do a bunch of LUIs to get useful register state.
    task init_register_state();
        for (int i = 0; i < 32; ++i) begin
            @(posedge itf.clk iff |itf.rmask[0]);
            gen.randomize() with {
                instr.j_type.opcode == op_b_lui;
                instr.j_type.rd == i[4:0];
            };

            // Your code here: package these memory interactions into a task.
            itf.rdata[0] <= gen.instr.word;
            itf.resp[0] <= 1'b1;
            @(posedge itf.clk) itf.resp[0] <= 1'b0;
        end
    endtask : init_register_state

    // Note that this memory model is not consistent! It ignores
    // writes and always reads out a random, valid instruction.
    task run_random_instrs();
        repeat (5000) begin
            @(posedge itf.clk iff (|itf.rmask[0] || |itf.wmask[0]));

            // Always read out a valid instruction.
            if (|itf.rmask[0]) begin
                gen.randomize();
                itf.rdata[0] <= gen.instr.word;
            end

            // If it's a write, do nothing and just respond.
            itf.resp[0] <= 1'b1;
            @(posedge itf.clk) itf.resp[0] <= 1'b0;
        end
    endtask : run_random_instrs

    always @(posedge itf.clk iff !itf.rst) begin
        if ($isunknown(itf.rmask[0]) || $isunknown(itf.wmask[0])) begin
            $error("Memory Error: mask containes 1'bx");
            itf.error <= 1'b1;
        end
        if ((|itf.rmask[0]) && (|itf.wmask[0])) begin
            $error("Memory Error: Simultaneous memory read and write");
            itf.error <= 1'b1;
        end
        if ((|itf.rmask[0]) || (|itf.wmask[0])) begin
            if ($isunknown(itf.addr[0])) begin
                $error("Memory Error: Address contained 'x");
                itf.error <= 1'b1;
            end
            // Only check for 16-bit alignment since instructions are
            // allowed to be at 16-bit boundaries due to JALR.
            if (itf.addr[0][0] != 1'b0) begin
                $error("Memory Error: Address is not 16-bit aligned");
                itf.error <= 1'b1;
            end
        end
    end

    // A single initial block ensures random stability.
    initial begin

        // Wait for reset.
        @(posedge itf.clk iff itf.rst == 1'b0);

        // Get some useful state into the processor by loading in a bunch of state.
        init_register_state();

        // Run!
        run_random_instrs();

        // Finish up
        $display("Random testbench finished!");
        $finish;
    end

endmodule : random_tb
