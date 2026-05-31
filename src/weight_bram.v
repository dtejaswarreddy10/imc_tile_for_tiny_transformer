// ---------------------------------------------------------------------------
// weight_bram.v
//
// 16 deep x 128 wide Block RAM, initialized from `weights.mem`.
// One read returns one full column of Wo (16 INT8 values) packed as:
//     bits [8*i +: 8]  =  Wo_q[i, addr]
//
// Synchronous read (1 cycle latency): registered output, infers BRAM in
// Vivado.
// ---------------------------------------------------------------------------
`timescale 1ns / 1ps

module weight_bram #(
    parameter MEM_INIT_FILE = "weights.mem",
    parameter ADDR_W        = 4,            // 16 columns -> 4 bits
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
