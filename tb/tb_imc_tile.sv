// ---------------------------------------------------------------------------
// tb_imc_tile.sv
//
// Self-checking testbench for imc_core. Loads the same FROZEN artifacts the
// hardware uses, runs the FSM, sweeps the readback port, and compares the
// 8 x 16 INT8 result against `golden_y.hex` byte-for-byte.
//
// PASS condition: ZERO mismatches.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_imc_tile;

    // ARTIFACTS_DIR can be overridden at compile time:
    //   xelab -d ARTIFACTS_DIR='"/abs/path/artifacts"' ...
    // Default is repository-relative (works when xsim runs from repo root).
`ifndef ARTIFACTS_DIR
  `define ARTIFACTS_DIR "/proj/dsv_xhd/tdudyala/personal/imc_assignment_2/artifacts"
`endif

    localparam string W_PATH      = `ARTIFACTS_DIR;
    localparam string W_FILE      = {W_PATH, "/weights.mem"};
    localparam string X_FILE      = {W_PATH, "/x_vectors.hex"};
    localparam string GOLDEN_FILE = {W_PATH, "/golden_y.hex"};

    localparam int S = 8;
    localparam int N = 16;

    // Clock / reset
    reg clk = 0;
    always #5 clk = ~clk;       // 100 MHz

    reg rst_n = 0;
    reg start = 0;
    wire done;

    reg  [2:0]  y_rd_addr = 3'd0;
    wire [127:0] y_rd_data;

    wire [3:0]  dbg_state;
    wire [2:0]  dbg_i;
    wire [3:0]  dbg_j;
    wire [127:0] dbg_w_col;
    wire [127:0] dbg_x_row;
    wire signed [31:0] dbg_acc;
    wire signed [7:0]  dbg_y_byte;

    imc_core #(
        .W_FILE       (W_FILE),
        .X_FILE       (X_FILE),
        .REQUANT_M    (32'sd91078),
        .REQUANT_SHIFT(24)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start),
        .done      (done),
        .y_rd_addr (y_rd_addr),
        .y_rd_data (y_rd_data),
        .dbg_state (dbg_state),
        .dbg_i     (dbg_i),
        .dbg_j     (dbg_j),
        .dbg_w_col (dbg_w_col),
        .dbg_x_row (dbg_x_row),
        .dbg_acc   (dbg_acc),
        .dbg_y_byte(dbg_y_byte)
    );

    // Golden expected outputs: S rows of 128 bits.
    reg [127:0] golden [0:S-1];

    integer mismatches;
    integer total_bytes;
    integer i, b;
    reg [127:0] got_row, exp_row;
    reg [7:0]   got_byte, exp_byte;

    initial begin
        $display("============================================================");
        $display("  TB_IMC_TILE -- bit-exact verification");
        $display("  W_FILE      = %s", W_FILE);
        $display("  X_FILE      = %s", X_FILE);
        $display("  GOLDEN_FILE = %s", GOLDEN_FILE);
        $display("============================================================");

        $readmemh(GOLDEN_FILE, golden);

        // Reset
        rst_n  = 1'b0;
        start  = 1'b0;
        repeat (4) @(posedge clk);
        rst_n  = 1'b1;
        repeat (2) @(posedge clk);

        // Pulse start
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        // Wait for done with timeout
        begin : wait_done
            integer cycles; cycles = 0;
            while (!done) begin
                @(posedge clk);
                cycles = cycles + 1;
                if (cycles > 5000) begin
                    $display("[TB] TIMEOUT waiting for done (%0d cycles)", cycles);
                    $fatal(1, "timeout");
                end
            end
            $display("[TB] core asserted done after %0d cycles", cycles);
        end

        // Read back each row, compare byte-by-byte
        mismatches  = 0;
        total_bytes = 0;
        for (i = 0; i < S; i = i + 1) begin
            y_rd_addr = i[2:0];
            @(posedge clk);     // address into readback path
            @(posedge clk);     // settle
            got_row = y_rd_data;
            exp_row = golden[i];
            $display("[TB] row %0d:", i);
            $display("       got = %032h", got_row);
            $display("       exp = %032h", exp_row);
            for (b = 0; b < N; b = b + 1) begin
                got_byte = got_row[8*b +: 8];
                exp_byte = exp_row[8*b +: 8];
                total_bytes = total_bytes + 1;
                if (got_byte !== exp_byte) begin
                    mismatches = mismatches + 1;
                    $display("       MISMATCH at i=%0d j=%0d  got=%0d  exp=%0d",
                             i, b, $signed(got_byte), $signed(exp_byte));
                end
            end
        end

        $display("============================================================");
        if (mismatches == 0) begin
            $display("  RESULT: PASS  (%0d / %0d bytes match)",
                     total_bytes, total_bytes);
        end else begin
            $display("  RESULT: FAIL  (%0d / %0d bytes mismatch)",
                     mismatches, total_bytes);
        end
        $display("============================================================");

        $finish;
    end

    // Optional waveform dump
    initial begin
        if ($test$plusargs("dump")) begin
            $dumpfile("tb_imc_tile.vcd");
            $dumpvars(0, tb_imc_tile);
        end
    end

endmodule
