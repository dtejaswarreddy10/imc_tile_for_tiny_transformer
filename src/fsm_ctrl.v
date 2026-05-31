// ---------------------------------------------------------------------------
// fsm_ctrl.v   (Phase-3 -- pipelined MAC + pipelined requantizer)
//
// Controller for the IMC tile. Sweeps output column j (outer) and row i
// (inner) -- weight-stationary dataflow: we load Wo column j once and
// then stream all 8 X rows through the MAC array.
//
// 12 named states (assignment requires >=5):
//
//   S_IDLE     : wait for `start`
//   S_LOAD_W   : weight_bram[w_addr] read in flight
//   S_LATCH_W  : w_col_reg <= w_data; queue x_addr
//   S_LOAD_X   : x_bram[x_addr] read in flight
//   S_LATCH_X  : x_row_reg <= x_data
//   S_MAC1     : mac_en pulse 1 -> Stage A (products) registered
//   S_MAC2     : mac_en pulse 2 -> Stage B (s8 pair-sums) registered
//   S_MAC3     : mac_en pulse 3 -> Stage C (s2 quad-sums) registered
//   S_MAC4     : mac_en pulse 4 -> Stage D (acc_out) registered
//   S_RQ1      : rq_en pulse    -> requantizer prod_reg <= acc * M
//   S_REQUANT  : y[i][j] <= sat(prod_reg >>> SHIFT); advance counters
//   S_DONE     : done=1, return to IDLE on !start
//
// MAC pipeline latency = 4 cycles, requantizer multiply latency = 1 cycle.
// Phase-2 still failed timing by -2.12 ns at 100 MHz because the
// requantizer's 32x32 multiply (2-DSP cascade) + 8-deep saturation chain
// + ILA fanout-130 routing all stacked combinationally between
// `acc_out_reg` and `y_mem`. Phase-3 inserts S_RQ1 so the multiply lands
// on a registered DSP P-port; both halves of the path now fit in 10 ns.
//
// Cycle budget per output: LOAD_X + LATCH_X + 4*MAC + RQ1 + REQUANT = 8 cycles.
// Outer per-column overhead: LOAD_W + LATCH_W = 2 cycles.
// Total: N*(2 + S*8) + 1 = 16*(2+8*8) + 1 = 1057 cycles end-to-end.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module fsm_ctrl #(
    parameter integer S = 8,    // X rows
    parameter integer N = 16    // Wo columns
) (
    input             clk,
    input             rst_n,
    input             start,

    // BRAM addresses
    output reg [3:0]  w_addr,
    output reg [2:0]  x_addr,

    // Datapath strobes
    output reg        w_latch,    // latch w_col_reg from weight_bram.data
    output reg        x_latch,    // latch x_row_reg from x_bram.data
    output reg        mac_en,     // advance MAC array pipeline
    output reg        rq_en,      // advance requantizer pipeline (prod_reg)
    output reg        y_we,       // write y[i][j] from requantizer output

    // Loop indices (also used to address the y register file)
    output reg [3:0]  j_idx,
    output reg [2:0]  i_idx,

    // Status / observability
    output reg        done,
    output reg [3:0]  state_o
);

    localparam [3:0]
        S_IDLE    = 4'd0,
        S_LOAD_W  = 4'd1,
        S_LATCH_W = 4'd2,
        S_LOAD_X  = 4'd3,
        S_LATCH_X = 4'd4,
        S_MAC1    = 4'd5,
        S_MAC2    = 4'd6,
        S_MAC3    = 4'd7,
        S_MAC4    = 4'd8,
        S_RQ1     = 4'd9,
        S_REQUANT = 4'd10,
        S_DONE    = 4'd11;

    reg [3:0] state, next_state;

    // -------------------------------------------------------------
    // Sequential: state register + indices + BRAM addresses
    // -------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            i_idx  <= 3'd0;
            j_idx  <= 4'd0;
            w_addr <= 4'd0;
            x_addr <= 3'd0;
        end else begin
            state <= next_state;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        i_idx  <= 3'd0;
                        j_idx  <= 4'd0;
                        w_addr <= 4'd0;
                    end
                end

                S_LATCH_W: begin
                    // queue X address so it is on the BRAM port during S_LOAD_X
                    x_addr <= i_idx;
                end

                S_REQUANT: begin
                    // y[i][j] just written. Decide next iteration and pre-load
                    // BRAM address one cycle ahead of the LOAD_* state.
                    if (i_idx == S-1 && j_idx == N-1) begin
                        // last output -- nothing to pre-load
                    end else if (i_idx < S-1) begin
                        i_idx  <= i_idx + 3'd1;
                        x_addr <= i_idx + 3'd1;
                    end else begin
                        i_idx  <= 3'd0;
                        j_idx  <= j_idx + 4'd1;
                        w_addr <= j_idx + 4'd1;
                    end
                end

                default: ;
            endcase
        end
    end

    // -------------------------------------------------------------
    // Combinational next-state
    // -------------------------------------------------------------
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE   : if (start)                 next_state = S_LOAD_W;
            S_LOAD_W :                            next_state = S_LATCH_W;
            S_LATCH_W:                            next_state = S_LOAD_X;
            S_LOAD_X :                            next_state = S_LATCH_X;
            S_LATCH_X:                            next_state = S_MAC1;
            S_MAC1   :                            next_state = S_MAC2;
            S_MAC2   :                            next_state = S_MAC3;
            S_MAC3   :                            next_state = S_MAC4;
            S_MAC4   :                            next_state = S_RQ1;
            S_RQ1    :                            next_state = S_REQUANT;
            S_REQUANT: begin
                if (i_idx == S-1 && j_idx == N-1) next_state = S_DONE;
                else if (i_idx < S-1)             next_state = S_LOAD_X;
                else                              next_state = S_LOAD_W;
            end
            S_DONE   : if (!start)                next_state = S_IDLE;
            default  :                            next_state = S_IDLE;
        endcase
    end

    // -------------------------------------------------------------
    // Moore-style output strobes
    // -------------------------------------------------------------
    always @(*) begin
        w_latch = 1'b0;
        x_latch = 1'b0;
        mac_en  = 1'b0;
        rq_en   = 1'b0;
        y_we    = 1'b0;
        done    = 1'b0;
        state_o = state;
        case (state)
            S_LATCH_W: w_latch = 1'b1;
            S_LATCH_X: x_latch = 1'b1;
            S_MAC1   : mac_en  = 1'b1;
            S_MAC2   : mac_en  = 1'b1;
            S_MAC3   : mac_en  = 1'b1;
            S_MAC4   : mac_en  = 1'b1;
            S_RQ1    : rq_en   = 1'b1;
            S_REQUANT: y_we    = 1'b1;
            S_DONE   : done    = 1'b1;
            default  : ;
        endcase
    end

endmodule
