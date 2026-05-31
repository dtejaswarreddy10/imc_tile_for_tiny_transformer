"""
sim_imc_core.py
---------------
Cycle-accurate Python model of `rtl/imc_core.v`. Consumes the SAME hex
files the RTL does (artifacts/weights.mem, x_vectors.hex) and produces a
byte-stream comparable to artifacts/golden_y.hex.

Purpose: in environments without an HDL simulator (xsim/iverilog/etc),
this proves the IMC tile algorithm is correct *as designed* before the
RTL ever runs in Vivado. Once you have Vivado, sim/tb_imc_tile.sv runs
the same comparison on the actual RTL.

Modeled fidelity:
  - 1-cycle synchronous BRAM read latency
  - 4-cycle pipelined MAC array (prod -> s8 pair-sums -> s2 quad-sums -> acc)
  - 1-cycle pipelined requantizer multiply (Phase-3)
  - 12-state FSM identical to fsm_ctrl.v (Phase-3)
  - Requantizer (acc * M) >>> SHIFT, INT8 saturation
  - Y register file, indexed y_mem[i*N + j]
  - Read-port assembly y_rd_data[8*j +: 8] = y_mem[row*N + j]
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
ART  = os.path.join(REPO, "artifacts")

S = 8
N = 16
K = 16
REQUANT_M     = 91078
REQUANT_SHIFT = 24

# FSM state codes (must match fsm_ctrl.v Phase-3)
S_IDLE, S_LOAD_W, S_LATCH_W, S_LOAD_X, S_LATCH_X, \
S_MAC1, S_MAC2, S_MAC3, S_MAC4, S_RQ1, S_REQUANT, S_DONE = range(12)

STATE_NAMES = {
    S_IDLE: "IDLE", S_LOAD_W: "LOAD_W", S_LATCH_W: "LATCH_W",
    S_LOAD_X: "LOAD_X", S_LATCH_X: "LATCH_X",
    S_MAC1: "MAC1", S_MAC2: "MAC2", S_MAC3: "MAC3", S_MAC4: "MAC4",
    S_RQ1: "RQ1", S_REQUANT: "REQUANT", S_DONE: "DONE",
}


def load_hex_lines(path: str) -> List[int]:
    """Read a $readmemh-compatible file; return list of integers (one per line).
    Skips blank lines and `// ...` / `# ...` comments."""
    out: List[int] = []
    with open(path, "r", encoding="ascii") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//") or s.startswith("#"):
                continue
            # strip inline // comments if any
            s = re.split(r"//", s, maxsplit=1)[0].strip()
            if not s:
                continue
            out.append(int(s, 16))
    return out


def s8(v: int) -> int:
    """Reinterpret an 8-bit unsigned int as signed."""
    v &= 0xFF
    return v - 0x100 if v & 0x80 else v


def sat_int8(v: int) -> int:
    if v > 127:  return 127
    if v < -128: return -128
    return v


def arith_rshift(v: int, n: int) -> int:
    """Arithmetic right shift on a Python signed int (Python `>>` already is)."""
    return v >> n


def unpack_lane(word: int, idx: int) -> int:
    """Extract signed 8-bit lane `idx` from a 128-bit packed word."""
    return s8((word >> (8 * idx)) & 0xFF)


@dataclass
class CoreState:
    state: int = S_IDLE
    i: int = 0
    j: int = 0
    w_addr: int = 0
    x_addr: int = 0
    w_col: int = 0          # 128-bit
    x_row: int = 0          # 128-bit
    w_bram_dout: int = 0    # registered BRAM output
    x_bram_dout: int = 0
    # MAC pipeline registers (Phase-2 balanced binary tree)
    prod: List[int] = field(default_factory=lambda: [0]*16)   # Stage A
    s8:   List[int] = field(default_factory=lambda: [0]*8)    # Stage B
    s2:   List[int] = field(default_factory=lambda: [0]*2)    # Stage C
    acc: int = 0                                              # Stage D
    # Requantizer pipeline register (Phase-3)
    prod_rq: int = 0                                          # acc * M
    y_mem: List[int] = field(default_factory=lambda: [0]*(S*N))
    cycle: int = 0
    done: bool = False


def step(c: CoreState, w_mem: List[int], x_mem: List[int], start: int) -> None:
    """Advance the model by one clock edge -- mirrors RTL semantics."""

    # ---- Combinational outputs from current state (Moore strobes) ----
    w_latch = (c.state == S_LATCH_W)
    x_latch = (c.state == S_LATCH_X)
    mac_en  = (c.state in (S_MAC1, S_MAC2, S_MAC3, S_MAC4))
    rq_en   = (c.state == S_RQ1)
    y_we    = (c.state == S_REQUANT)

    # ---- Combinational requantizer stage 2 from registered prod_rq ----
    y_byte = sat_int8(arith_rshift(c.prod_rq, REQUANT_SHIFT))

    # ---- Compute next state (next-state logic) ----
    ns = c.state
    if c.state == S_IDLE:
        if start: ns = S_LOAD_W
    elif c.state == S_LOAD_W:   ns = S_LATCH_W
    elif c.state == S_LATCH_W:  ns = S_LOAD_X
    elif c.state == S_LOAD_X:   ns = S_LATCH_X
    elif c.state == S_LATCH_X:  ns = S_MAC1
    elif c.state == S_MAC1:     ns = S_MAC2
    elif c.state == S_MAC2:     ns = S_MAC3
    elif c.state == S_MAC3:     ns = S_MAC4
    elif c.state == S_MAC4:     ns = S_RQ1
    elif c.state == S_RQ1:      ns = S_REQUANT
    elif c.state == S_REQUANT:
        if c.i == S-1 and c.j == N-1: ns = S_DONE
        elif c.i < S-1:               ns = S_LOAD_X
        else:                         ns = S_LOAD_W
    elif c.state == S_DONE:
        if not start: ns = S_IDLE

    # ---- Sequential updates (registered at clock edge) ----

    # BRAM synchronous read: data updates from current address
    new_w_dout = w_mem[c.w_addr]
    new_x_dout = x_mem[c.x_addr]

    # Datapath registers
    new_w_col = c.w_col
    new_x_row = c.x_row
    if w_latch: new_w_col = c.w_bram_dout
    if x_latch: new_x_row = c.x_bram_dout

    # MAC pipeline -- ALL stages capture from CURRENT registers (not new ones),
    # mirroring RTL non-blocking semantics.
    new_prod = list(c.prod)
    new_s8   = list(c.s8)
    new_s2   = list(c.s2)
    new_acc  = c.acc
    if mac_en:
        # Stage A: products from current x/w registers
        new_prod = [
            unpack_lane(c.w_col, k) * unpack_lane(c.x_row, k)
            for k in range(16)
        ]
        # Stage B: pair-sums of current prod registers
        new_s8 = [
            c.prod[2*k] + c.prod[2*k + 1] for k in range(8)
        ]
        # Stage C: 4-then-2 reduction of current s8 registers
        s4 = [c.s8[2*k] + c.s8[2*k + 1] for k in range(4)]
        new_s2 = [s4[0] + s4[1], s4[2] + s4[3]]
        # Stage D: final add of current s2 registers
        new_acc = c.s2[0] + c.s2[1]

    # Requantizer Stage 1: register acc * M while rq_en is high
    new_prod_rq = c.prod_rq
    if rq_en:
        new_prod_rq = c.acc * REQUANT_M

    # Y register file write
    new_y_mem = list(c.y_mem)
    if y_we:
        new_y_mem[c.i * N + c.j] = y_byte & 0xFF

    # Indices + BRAM addresses (mirror RTL conditional updates)
    new_i, new_j = c.i, c.j
    new_w_addr, new_x_addr = c.w_addr, c.x_addr

    if c.state == S_IDLE and start:
        new_i, new_j, new_w_addr = 0, 0, 0
    elif c.state == S_LATCH_W:
        new_x_addr = c.i
    elif c.state == S_REQUANT:
        if c.i == S-1 and c.j == N-1:
            pass
        elif c.i < S-1:
            new_i = c.i + 1
            new_x_addr = c.i + 1
        else:
            new_i = 0
            new_j = c.j + 1
            new_w_addr = c.j + 1

    # Apply
    c.state = ns
    c.i = new_i
    c.j = new_j
    c.w_addr = new_w_addr
    c.x_addr = new_x_addr
    c.w_bram_dout = new_w_dout
    c.x_bram_dout = new_x_dout
    c.w_col = new_w_col
    c.x_row = new_x_row
    c.prod = new_prod
    c.s8   = new_s8
    c.s2   = new_s2
    c.acc  = new_acc
    c.prod_rq = new_prod_rq
    c.y_mem = new_y_mem
    c.done = (c.state == S_DONE)
    c.cycle += 1


def y_rd_data(c: CoreState, row: int) -> int:
    """Mirror of the RTL readback assembly."""
    word = 0
    for j in range(N):
        word |= (c.y_mem[row * N + j] & 0xFF) << (8 * j)
    return word


def main() -> int:
    w_mem = load_hex_lines(os.path.join(ART, "weights.mem"))
    x_mem = load_hex_lines(os.path.join(ART, "x_vectors.hex"))
    golden = load_hex_lines(os.path.join(ART, "golden_y.hex"))

    assert len(w_mem) == 16, f"weights.mem must have 16 lines, got {len(w_mem)}"
    assert len(x_mem) == 8,  f"x_vectors.hex must have 8 lines, got {len(x_mem)}"
    assert len(golden) == 8, f"golden_y.hex must have 8 lines, got {len(golden)}"

    c = CoreState()

    # Reset complete (modeled by initial defaults). Hold start low for a few
    # cycles, pulse start for one cycle, then wait for done.
    for _ in range(2):
        step(c, w_mem, x_mem, start=0)

    step(c, w_mem, x_mem, start=1)
    while not c.done:
        step(c, w_mem, x_mem, start=0)
        if c.cycle > 5000:
            print(f"[SIM] TIMEOUT at cycle {c.cycle}")
            return 2

    print(f"[SIM] core asserted done at cycle {c.cycle}")

    # Compare each row to golden
    mismatches = 0
    total = 0
    for i in range(S):
        got = y_rd_data(c, i)
        exp = golden[i]
        print(f"  row {i}:")
        print(f"    got = {got:032x}")
        print(f"    exp = {exp:032x}")
        for b in range(N):
            g = s8((got >> (8 * b)) & 0xFF)
            e = s8((exp >> (8 * b)) & 0xFF)
            total += 1
            if g != e:
                mismatches += 1
                print(f"    MISMATCH i={i} j={b}  got={g} exp={e}")

    print("=" * 60)
    if mismatches == 0:
        print(f"  RESULT: PASS  ({total}/{total} bytes match)")
        return 0
    else:
        print(f"  RESULT: FAIL  ({mismatches}/{total} bytes mismatch)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
