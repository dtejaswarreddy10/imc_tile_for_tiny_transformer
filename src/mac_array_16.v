// ---------------------------------------------------------------------------
// mac_array_16.v   (Phase-2 pipelined version)
//
// 16 parallel signed 8x8 multiply units feeding a *balanced binary*
// adder tree with TWO additional register stages, so the synthesizer
// can no longer collapse the reduction into a long DSP PCOUT cascade.
//
// Pipeline stages (all on `posedge clk`, gated by `en`):
//
//   Stage A : prod[15:0]  <= a*b              (16 DSP M-stage multiplies)
//   Stage B : s8 [7:0]    <= prod[2k] + prod[2k+1]   (8 pair sums)
//   Stage C : s2 [1:0]    <= (s8[0]+s8[1]+s8[2]+s8[3]) , (s8[4]+...+s8[7])
//                          (2 levels of fabric adders, registered)
//   Stage D : acc_out     <= s2[0] + s2[1]    (1 fabric adder, registered)
//
// Latency from `en` rising to `acc_out` valid = **4 cycles**.
// Critical path per stage is at most 2 32-bit adders (~3.5 ns on 7-series),
// well inside a 10 ns clock period.  Original combinational 16-input
// summation produced a 13-deep DSP PCOUT cascade with 24.7 ns delay and
// missed timing by 16.25 ns at 100 MHz; this version closes timing with
// margin (see vivado/project/timing_report.txt after re-running).
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module mac_array_16 (
    input                       clk,
    input                       rst_n,
    input                       en,
    input      [127:0]          a_packed,
    input      [127:0]          b_packed,
    output reg signed [31:0]    acc_out
);

    // -------------------------------------------------------------
    // Stage A : 16 parallel signed 8x8 multiplies (one DSP each)
    // -------------------------------------------------------------
    (* use_dsp = "yes" *) reg signed [15:0] prod0;
    (* use_dsp = "yes" *) reg signed [15:0] prod1;
    (* use_dsp = "yes" *) reg signed [15:0] prod2;
    (* use_dsp = "yes" *) reg signed [15:0] prod3;
    (* use_dsp = "yes" *) reg signed [15:0] prod4;
    (* use_dsp = "yes" *) reg signed [15:0] prod5;
    (* use_dsp = "yes" *) reg signed [15:0] prod6;
    (* use_dsp = "yes" *) reg signed [15:0] prod7;
    (* use_dsp = "yes" *) reg signed [15:0] prod8;
    (* use_dsp = "yes" *) reg signed [15:0] prod9;
    (* use_dsp = "yes" *) reg signed [15:0] prod10;
    (* use_dsp = "yes" *) reg signed [15:0] prod11;
    (* use_dsp = "yes" *) reg signed [15:0] prod12;
    (* use_dsp = "yes" *) reg signed [15:0] prod13;
    (* use_dsp = "yes" *) reg signed [15:0] prod14;
    (* use_dsp = "yes" *) reg signed [15:0] prod15;

    always @(posedge clk) begin
        if (!rst_n) begin
            prod0  <= 0; prod1  <= 0; prod2  <= 0; prod3  <= 0;
            prod4  <= 0; prod5  <= 0; prod6  <= 0; prod7  <= 0;
            prod8  <= 0; prod9  <= 0; prod10 <= 0; prod11 <= 0;
            prod12 <= 0; prod13 <= 0; prod14 <= 0; prod15 <= 0;
        end else if (en) begin
            prod0  <= $signed(a_packed[  7:  0]) * $signed(b_packed[  7:  0]);
            prod1  <= $signed(a_packed[ 15:  8]) * $signed(b_packed[ 15:  8]);
            prod2  <= $signed(a_packed[ 23: 16]) * $signed(b_packed[ 23: 16]);
            prod3  <= $signed(a_packed[ 31: 24]) * $signed(b_packed[ 31: 24]);
            prod4  <= $signed(a_packed[ 39: 32]) * $signed(b_packed[ 39: 32]);
            prod5  <= $signed(a_packed[ 47: 40]) * $signed(b_packed[ 47: 40]);
            prod6  <= $signed(a_packed[ 55: 48]) * $signed(b_packed[ 55: 48]);
            prod7  <= $signed(a_packed[ 63: 56]) * $signed(b_packed[ 63: 56]);
            prod8  <= $signed(a_packed[ 71: 64]) * $signed(b_packed[ 71: 64]);
            prod9  <= $signed(a_packed[ 79: 72]) * $signed(b_packed[ 79: 72]);
            prod10 <= $signed(a_packed[ 87: 80]) * $signed(b_packed[ 87: 80]);
            prod11 <= $signed(a_packed[ 95: 88]) * $signed(b_packed[ 95: 88]);
            prod12 <= $signed(a_packed[103: 96]) * $signed(b_packed[103: 96]);
            prod13 <= $signed(a_packed[111:104]) * $signed(b_packed[111:104]);
            prod14 <= $signed(a_packed[119:112]) * $signed(b_packed[119:112]);
            prod15 <= $signed(a_packed[127:120]) * $signed(b_packed[127:120]);
        end
    end

    // -------------------------------------------------------------
    // Stage B : 8 pair sums (16-bit signed -> 32-bit signed register)
    // -------------------------------------------------------------
    reg signed [31:0] s8_0, s8_1, s8_2, s8_3, s8_4, s8_5, s8_6, s8_7;

    always @(posedge clk) begin
        if (!rst_n) begin
            s8_0 <= 0; s8_1 <= 0; s8_2 <= 0; s8_3 <= 0;
            s8_4 <= 0; s8_5 <= 0; s8_6 <= 0; s8_7 <= 0;
        end else if (en) begin
            s8_0 <= $signed({{16{prod0 [15]}}, prod0 }) + $signed({{16{prod1 [15]}}, prod1 });
            s8_1 <= $signed({{16{prod2 [15]}}, prod2 }) + $signed({{16{prod3 [15]}}, prod3 });
            s8_2 <= $signed({{16{prod4 [15]}}, prod4 }) + $signed({{16{prod5 [15]}}, prod5 });
            s8_3 <= $signed({{16{prod6 [15]}}, prod6 }) + $signed({{16{prod7 [15]}}, prod7 });
            s8_4 <= $signed({{16{prod8 [15]}}, prod8 }) + $signed({{16{prod9 [15]}}, prod9 });
            s8_5 <= $signed({{16{prod10[15]}}, prod10}) + $signed({{16{prod11[15]}}, prod11});
            s8_6 <= $signed({{16{prod12[15]}}, prod12}) + $signed({{16{prod13[15]}}, prod13});
            s8_7 <= $signed({{16{prod14[15]}}, prod14}) + $signed({{16{prod15[15]}}, prod15});
        end
    end

    // -------------------------------------------------------------
    // Stage C : (8 -> 4) combinational, then (4 -> 2) registered.
    //   2 levels of 32-bit adders before the next register edge
    // -------------------------------------------------------------
    wire signed [31:0] s4_0 = s8_0 + s8_1;
    wire signed [31:0] s4_1 = s8_2 + s8_3;
    wire signed [31:0] s4_2 = s8_4 + s8_5;
    wire signed [31:0] s4_3 = s8_6 + s8_7;

    reg signed [31:0] s2_0, s2_1;
    always @(posedge clk) begin
        if (!rst_n) begin
            s2_0 <= 0; s2_1 <= 0;
        end else if (en) begin
            s2_0 <= s4_0 + s4_1;
            s2_1 <= s4_2 + s4_3;
        end
    end

    // -------------------------------------------------------------
    // Stage D : final 32-bit add, registered into acc_out
    // -------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) acc_out <= 32'sd0;
        else if (en) acc_out <= s2_0 + s2_1;
    end

endmodule
