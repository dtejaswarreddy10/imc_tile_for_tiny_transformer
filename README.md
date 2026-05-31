# IMC Tile — INT8 Transformer Attention on Zedboard

> **One-line pitch.** A weight-stationary INT8 GEMM tile (16-MAC DSP48 array,
> BRAM-resident weights, 12-state FSM, ILA + VIO only) that bit-exactly
> executes the output projection of one attention head from
> `prajjwal1/bert-tiny` on a Zynq-7020 — a scaled-down, hand-built version
> of the same primitive that runs inside NVIDIA tensor cores, AMD MI300,
> and d-Matrix DIMC chips.

[![Board](https://img.shields.io/badge/board-Zedboard%20%7C%20xc7z020-1f6feb)](https://digilent.com/reference/programmable-logic/zedboard/)
[![Tool](https://img.shields.io/badge/Vivado-2023.x-1f6feb)](https://www.xilinx.com/products/design-tools/vivado.html)
[![Lang](https://img.shields.io/badge/HDL-Verilog%20%7C%20SystemVerilog-1f6feb)]()
[![Status](https://img.shields.io/badge/bit--exact-128%2F128%20PASS-2ea043)]()

---

## At a glance

| Aspect            | Value                                                              |
|-------------------|--------------------------------------------------------------------|
| Workload          | `prajjwal1/bert-tiny`, layer-0 / head-0 attention output projection |
| Tile dimensions   | `Y[8×16] = X[8×16] · Wo[16×16]`, INT8                              |
| Storage           | `Wo` in BRAM, **128-bit wide word per column** (the IMC idea)      |
| Compute           | **16 parallel DSP48E1** MACs, INT32 accumulate                     |
| Quantisation      | symmetric per-tensor INT8, requant `(acc·M)>>>shift`, M=91078 s=24 |
| Controller        | 12-state Moore FSM (LOAD/LATCH split + 4 MAC + 1 RQ pipeline cycle) |
| Runtime control   | Vivado **VIO** (start, soft-reset, readback address)               |
| Observability     | Vivado **ILA** (state, indices, MAC bus, acc, requant byte, done)  |
| **Verification**  | **128/128 bytes match `golden_y.hex`** — Python sim, xsim, on-board |
| End-to-end cycles | 1 059 cycles ≈ 10.59 µs @ 100 MHz (Phase-3 pipelined)               |

> Per assignment requirements: **no PS, no AXI** — the design is pure PL.

---

## Architecture

```
                +---------------------+
                |  weight_bram (BRAM) |
                |  16 x 128b          |
                |  init=weights.mem   |
                +-----+---------------+
                      | 128b column
                      v
                +-----+---------+        +---------------------+
                | w_col_reg     |        |  x_bram (BRAM)      |
                +-----+---------+        |  8 x 128b           |
                      |                  |  init=x_vectors.hex |
                      |   +--------------+                     |
                      v   v              +---------------------+
                +-----+---+----+
                | mac_array_16 |  16 x DSP48E1, 2-stage pipe
                +------+-------+
                       |
                       v
                +------+-------+
                | acc (INT32)  |
                +------+-------+
                       |
                       v
                +------+-------+
                | requantizer  |  (acc * M) >>> SHIFT, sat INT8
                +------+-------+
                       |
                       v
                +------+--------+
                |  y_mem 8x16B  | <--- y_rd_addr  ---> y_rd_data[127:0]
                +---------------+

          fsm_ctrl (12 states) drives every strobe + address
          ILA observes the dataflow ; VIO drives start/readback
```

See [docs/report/REPORT.md](docs/report/REPORT.md) for the full block
diagram, FSM bubble diagram, dataflow timing, and verification log.

---

## Repository layout

```
sw/                    Python: layer extraction, INT8 quant, golden, cycle-acc model
  golden_int8_gemm.py     hand-rolled NumPy quant + integer GEMM + requant
  extract_and_quantize.py driver -- emits the frozen artifacts
  sim_imc_core.py         cycle-accurate model of imc_core.v (no HDL sim needed)

artifacts/             FROZEN reference -- single source of truth
  weights.mem             16 x 128-bit hex (one column of Wo per line)
  x_vectors.hex           8  x 128-bit hex (one row of X  per line)
  golden_y.hex            8  x 128-bit hex (reference Y_q outputs)
  requant.txt             scales, M, shift, error stats

src/                   Verilog source (Vivado-targeted)
  weight_bram.v           16 x 128b synchronous BRAM (one read = full column)
  x_bram.v                8  x 128b synchronous BRAM
  mac_array_16.v          16 parallel DSP48-targeted MACs + adder tree
  requantizer.v           (acc*M)>>>shift, INT8 saturation
  fsm_ctrl.v              12-state Moore FSM (Phase-3)
  imc_core.v              datapath top + Y register file
  imc_tile_top.v          chip top: VIO + ILA + core

tb/
  tb_imc_tile.sv          self-checking testbench vs golden_y.hex

constraints/
  zedboard.xdc            clk Y9 (100 MHz), LD0 T22

vivado/
  build.tcl               project create + IP gen (vio_0, ila_0) + synth + impl + bit
  program_and_verify.tcl  program board, drive VIO, sweep readback, diff vs golden
  run_sim.sh              one-shot xsim wrapper

docs/
  00_PRIMER_before_rtl.md   from-scratch primer (read this first)
  devlog.md                 per-day build log
  report/REPORT.md          full project report
  site/index.html           one-page site (LinkedIn target)
  INTERVIEW.md              interview cheat-sheet
  img/                      diagrams + ILA / VIO screenshots
```

---

## Quickstart

```
# 1. Frozen artifacts (Python only)
python sw/extract_and_quantize.py

# 2. Pre-RTL design-intent check (no Vivado required, validates 128/128 bytes)
python sw/sim_imc_core.py

# 3. RTL simulation (Vivado xsim)
bash vivado/run_sim.sh

# 4. Synthesis + implementation + bitstream (Vivado)
vivado -mode batch -source vivado/build.tcl

# 5. On-board verification (Zedboard over USB-JTAG)
vivado -mode batch -source vivado/program_and_verify.tcl
```

---

## Results

| Stage                       | Outcome                                       |
|-----------------------------|-----------------------------------------------|
| Quantisation error          | max 1.43 LSB / mean 0.54 LSB vs float          |
| Python cycle-accurate sim   | **PASS** 128/128 bytes match (cycle 1059, Phase-3) |
| Vivado xsim TB (Phase-1)    | **PASS** 128/128 bytes match (673 cycles)      |
| BRAM (RAMB36E1)             | 13 / 140 (9.29 %, incl. ILA/VIO debug)         |
| LUT / FF (Phase-3 build)    | 2 798 / 5 633 (5.3 % / 5.3 %)                  |
| DSP48E1 utilisation         | **23 / 220** (10.45 %) — 16 MAC + 7 RQ multiply |
| WNS @ 100 MHz — Phase-1     | **−16.252 ns** (13-deep DSP cascade)           |
| WNS @ 100 MHz — Phase-2     | **−2.121 ns** (requantizer became critical)    |
| WNS @ 100 MHz — Phase-3     | **+1.070 ns** (MET, 0 failing endpoints)       |
| Wall-clock per tile         | 17.7 µs → **10.59 µs (1.67× faster)**         |
| On-board (ILA/VIO)          | **PASS** 8/8 rows match                        |

ILA waveform and VIO readback screenshots: `docs/img/`.

---

## Why this project

- **In-Memory Computing** — one BRAM read delivers a full 16-element kernel.
  This is the architectural lever d-Matrix, Mythic, and Samsung HBM-PIM are
  built on.
- **Transformer attention** — same primitive (INT8 GEMM, weight-stationary,
  requantize) that powers every modern LLM accelerator.
- **Bit-exact verification** — no "close enough"; every byte equals the
  Python reference. The discipline that real silicon teams require.
- **Pure PL, no AXI** — proves the design works as a *standalone IMC tile*,
  not as an AXI peripheral
