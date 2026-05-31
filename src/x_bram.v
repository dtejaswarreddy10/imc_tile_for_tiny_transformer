// ---------------------------------------------------------------------------
// x_bram.v
//
// 8 deep x 128 wide Block RAM, initialized from `x_vectors.hex`.
// One read returns one full row of X (16 INT8 values) packed as:
//     bits [8*k +: 8]  =  X_q[addr, k]
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module x_bram #(
    parameter MEM_INIT_FILE = "x_vectors.hex",
    parameter ADDR_W        = 3,            // 8 rows -> 3 bits
    parameter DATA_W        = 128
) (
    input                          clk,
    input      [ADDR_W-1:0]        addr,
    output reg [DATA_W-1:0]        data
);

    (* ram_style = "block" *)
    reg [DATA_W-1:0] mem [0:(1<<ADDR_W)-1];

    initial begin
        $readmemh(MEM_INIT_FILE, mem);
    end

    always @(posedge clk) begin
        data <= mem[addr];
    end

endmodule
