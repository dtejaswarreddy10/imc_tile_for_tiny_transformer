// ---------------------------------------------------------------------------
// imc_core.v
//
// Core IMC tile: weight_bram + x_bram + MAC array + requantizer + Y register
// file + FSM controller. Pure synchronous PL logic. No AXI, no PS hooks.
//
// The simulation testbench instantiates this directly. The on-chip top
// (imc_tile_top.v) wraps this with VIO + ILA debug cores.
//
// Parameters of interest:
//   REQUANT_M      - 32-bit signed multiplier (default 91078, from requant.txt)
//   REQUANT_SHIFT  - arithmetic right shift amount (default 24)
//   W_FILE         - hex file initialising the weight BRAM
//   X_FILE         - hex file initialising the activation BRAM
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module imc_core #(
    parameter integer S             = 8,
    parameter integer K             = 16,
    parameter integer N             = 16,
    parameter signed [31:0] REQUANT_M     = 32'sd91078,
    parameter integer       REQUANT_SHIFT = 24,
    parameter W_FILE = "weights.mem",
    parameter X_FILE = "x_vectors.hex"
) (
    input              clk,
    input              rst_n,
    input              start,
    output             done,

    // Result readback port (used by VIO on the board, by TB in simulation)
    input      [2:0]   y_rd_addr,
    output     [127:0] y_rd_data,

    // Observability for ILA / waveform
    output     [3:0]   dbg_state,
    output     [2:0]   dbg_i,
    output     [3:0]   dbg_j,
    output     [127:0] dbg_w_col,
    output     [127:0] dbg_x_row,
    output signed [31:0] dbg_acc,
    output signed [7:0]  dbg_y_byte
);

    // -----------------------------------------------------------------
    // Wires from FSM
    // -----------------------------------------------------------------
    wire [3:0] w_addr;
    wire [2:0] x_addr;
    wire       w_latch, x_latch, mac_en, rq_en, y_we;
    wire [3:0] j_idx;
    wire [2:0] i_idx;
    wire [3:0] state_o;

    // -----------------------------------------------------------------
    // Weight + activation BRAMs (synchronous read, 1-cycle latency)
    // -----------------------------------------------------------------
    wire [127:0] w_bram_dout;
    wire [127:0] x_bram_dout;

    weight_bram #(.MEM_INIT_FILE(W_FILE)) u_wbram (
        .clk (clk),
        .addr(w_addr),
        .data(w_bram_dout)
    );

    x_bram #(.MEM_INIT_FILE(X_FILE)) u_xbram (
        .clk (clk),
        .addr(x_addr),
        .data(x_bram_dout)
    );

    // -----------------------------------------------------------------
    // Datapath registers
    // -----------------------------------------------------------------
    reg [127:0] w_col_reg;
    reg [127:0] x_row_reg;

    always @(posedge clk) begin
        if (!rst_n) begin
            w_col_reg <= 128'd0;
            x_row_reg <= 128'd0;
        end else begin
            if (w_latch) w_col_reg <= w_bram_dout;
            if (x_latch) x_row_reg <= x_bram_dout;
        end
    end

    // -----------------------------------------------------------------
    // 16-wide MAC array (DSP48-targeted, balanced binary adder tree).
    // **4-cycle latency** from `mac_en` assertion to acc_out valid.
    // FSM holds `mac_en` for four states (S_MAC1..S_MAC4) so all four
    // pipeline stages (prod -> s8 -> s2 -> acc) advance.
    // -----------------------------------------------------------------
    wire signed [31:0] acc_w;

    mac_array_16 u_mac (
        .clk     (clk),
        .rst_n   (rst_n),
        .en      (mac_en),
        .a_packed(w_col_reg),
        .b_packed(x_row_reg),
        .acc_out (acc_w)
    );

    // -----------------------------------------------------------------
    // Requantizer (Phase-3: 1-cycle pipelined multiply, then combo sat)
    // -----------------------------------------------------------------
    wire signed [7:0] y_byte_w;
    requantizer #(
        .M    (REQUANT_M),
        .SHIFT(REQUANT_SHIFT)
    ) u_rq (
        .clk  (clk),
        .rst_n(rst_n),
        .en   (rq_en),
        .acc  (acc_w),
        .y_q  (y_byte_w)
    );

    // -----------------------------------------------------------------
    // Y register file: S rows x N bytes, flat addressed.
    //   index = i*N + j        ->  y_mem[i*16 + j] = y[i][j]
    // -----------------------------------------------------------------
    reg [7:0] y_mem [0:S*N-1];
    integer ii;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (ii = 0; ii < S*N; ii = ii + 1) y_mem[ii] <= 8'd0;
        end else if (y_we) begin
            y_mem[i_idx * N + j_idx] <= y_byte_w;
        end
    end

    // -----------------------------------------------------------------
    // Readback: assemble row `y_rd_addr` from 16 byte cells.
    // bit [8*j +: 8] = y[y_rd_addr][j]   (matches packing convention).
    // -----------------------------------------------------------------
    genvar g;
    generate
        for (g = 0; g < N; g = g + 1) begin : g_yrd
            assign y_rd_data[8*g +: 8] = y_mem[y_rd_addr * N + g];
        end
    endgenerate

    // -----------------------------------------------------------------
    // Controller
    // -----------------------------------------------------------------
    fsm_ctrl #(.S(S), .N(N)) u_fsm (
        .clk   (clk),
        .rst_n (rst_n),
        .start (start),
        .w_addr(w_addr),
        .x_addr(x_addr),
        .w_latch(w_latch),
        .x_latch(x_latch),
        .mac_en (mac_en),
        .rq_en  (rq_en),
        .y_we   (y_we),
        .j_idx  (j_idx),
        .i_idx  (i_idx),
        .done   (done),
        .state_o(state_o)
    );

    // -----------------------------------------------------------------
    // Observability
    // -----------------------------------------------------------------
    assign dbg_state  = state_o;
    assign dbg_i      = i_idx;
    assign dbg_j      = j_idx;
    assign dbg_w_col  = w_col_reg;
    assign dbg_x_row  = x_row_reg;
    assign dbg_acc    = acc_w;
    assign dbg_y_byte = y_byte_w;

endmodule
