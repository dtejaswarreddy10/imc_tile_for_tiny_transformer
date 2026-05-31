// ---------------------------------------------------------------------------
// requantizer.v   (Phase-3 -- pipelined to break the acc -> y_mem path)
//
// Combinational requantization stage was the new critical path after the
// MAC adder tree was pipelined (Phase-2 WNS was -2.12 ns @ 100 MHz):
//
//     acc_out_reg --> 32x32 mult (2-DSP cascade, 5.55 ns combo)
//                 --> 64-bit shift
//                 --> 8-deep CARRY4 saturation chain
//                 --> dbg_y_byte (ILA fanout = 130, +1.25 ns route)
//                 --> y_mem_reg                              total = 11.97 ns
//
// Phase-3 fix: register the 32x32 product (lands on the DSP48 P-stage),
// breaking the path into two halves that each fit comfortably in 10 ns:
//
//     Stage 1: prod_reg <= acc * M                ~5.5 ns (DSP cascade)
//     Stage 2: y_q       = sat(prod_reg >>> SHIFT) ~3.5 ns (carry+sat+route)
//
// Latency from `en` rising to `y_q` valid = 1 cycle.
// FSM holds `en` high in S_RQ1 so the multiply is captured one cycle
// before y_we asserts in S_REQUANT.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module requantizer #(
    parameter signed [31:0] M     = 32'sd91078,
    parameter integer       SHIFT = 24
) (
    input                   clk,
    input                   rst_n,
    input                   en,
    input  signed [31:0]    acc,
    output reg signed [7:0] y_q
);

    // Stage 1: registered 32x32 signed multiply.
    // (* use_dsp = "yes" *) lets Vivado pull the register into the DSP48
    // P-stage, so the long PCIN/PCOUT cascade is no longer combinational
    // with the downstream saturation logic.
    (* use_dsp = "yes" *) reg signed [63:0] prod_reg;

    always @(posedge clk) begin
        if (!rst_n)      prod_reg <= 64'sd0;
        else if (en)     prod_reg <= $signed(acc) * M;
    end

    // Stage 2: combinational shift + saturate from the registered product.
    wire signed [63:0] shifted_w = prod_reg >>> SHIFT;

    always @(*) begin
        if      (shifted_w >  64'sd127)   y_q =  8'sd127;
        else if (shifted_w < -64'sd128)   y_q = -8'sd128;
        else                              y_q =  shifted_w[7:0];
    end

endmodule
