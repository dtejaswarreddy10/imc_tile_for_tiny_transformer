"""
golden_int8_gemm.py
--------------------
Pure-NumPy reference for the IMC tile. Provides:

  * symmetric per-tensor INT8 quantization (weights and activations)
  * integer-only matmul accumulating in INT32
  * requantization via (acc * M) >>> shift  with INT8 saturation
  * derivation of (M, shift) from a target float multiplier

This module is the SINGLE SOURCE OF TRUTH for what the hardware must
compute. Once the artifacts (.mem / .hex) are emitted, this code is
frozen and never modified during RTL debug.

Bit-packing convention used throughout the project:
  - 16 INT8 values are packed into a 128-bit word.
  - Index `i` (0..15) lives at bit positions [8*i +: 8] of the word.
  - Therefore in the .mem/.hex file (MSB-first hex), index 15 is the
    LEFT-most byte and index 0 is the RIGHT-most byte.

  Example, indices 0..3 = (0x01, 0x02, 0x03, 0x04), rest 0:
       128-bit word value = 0x00000000_00000000_00000000_04030201
       hex line written   = "00000000000000000000000004030201"
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

INT8_MIN = -128
INT8_MAX = 127


@dataclass(frozen=True)
class QParams:
    """Symmetric per-tensor INT8 quantization parameters."""
    scale: float        # float -> int  via  q = round(x / scale)


def derive_qparams(x: np.ndarray) -> QParams:
    """Symmetric per-tensor scale: max(|x|) / 127. Avoid div-by-zero."""
    amax = float(np.max(np.abs(x))) if x.size else 0.0
    scale = amax / 127.0 if amax > 0 else 1.0
    return QParams(scale=scale)


def quantize_int8(x: np.ndarray, qp: QParams) -> np.ndarray:
    """Float -> INT8 (clipped, banker's-rounded via numpy round)."""
    q = np.round(x / qp.scale).astype(np.int32)
    q = np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)
    return q


def dequantize(q: np.ndarray, qp: QParams) -> np.ndarray:
    """INT8 -> float (for sanity comparisons only)."""
    return q.astype(np.float32) * qp.scale


# ---------------------------------------------------------------------------
# Requantization multiplier derivation
# ---------------------------------------------------------------------------

def derive_requant_mshift(real_multiplier: float,
                          shift_bits: int = 24) -> Tuple[int, int]:
    """
    Approximate `real_multiplier` (a positive float, typically <1) as an
    integer pair (M, shift) such that  real ~= M / 2**shift.

    We fix `shift_bits` (default 24). M is then chosen as round(real * 2^shift)
    and clipped to fit in INT32.
    """
    if real_multiplier <= 0:
        raise ValueError(f"real_multiplier must be positive, got {real_multiplier}")
    M = int(round(real_multiplier * (1 << shift_bits)))
    if M >= (1 << 31):
        # Reduce shift until M fits in INT32 (signed). Should rarely trigger
        # for our dimensions but kept for safety.
        while M >= (1 << 31) and shift_bits > 0:
            shift_bits -= 1
            M = int(round(real_multiplier * (1 << shift_bits)))
    return M, shift_bits


# ---------------------------------------------------------------------------
# Integer GEMM + requantization (the FPGA's exact math)
# ---------------------------------------------------------------------------

def int_matmul_int32(x_q: np.ndarray, w_q: np.ndarray) -> np.ndarray:
    """
    Pure integer matmul. x_q is [S, K] INT8, w_q is [K, N] INT8.
    Returns [S, N] INT32.
    """
    if x_q.dtype != np.int8 or w_q.dtype != np.int8:
        raise TypeError("inputs must be INT8")
    return x_q.astype(np.int32) @ w_q.astype(np.int32)


def requantize_int32_to_int8(acc: np.ndarray, M: int, shift: int) -> np.ndarray:
    """
    Apply (acc * M) >>> shift, saturate to INT8. Uses arithmetic right shift
    (equivalent to floor-division by 2^shift on signed integers in NumPy).
    """
    prod = acc.astype(np.int64) * np.int64(M)
    # arithmetic right shift on signed int64 (numpy >> on signed is arith)
    shifted = prod >> np.int64(shift)
    sat = np.clip(shifted, INT8_MIN, INT8_MAX).astype(np.int8)
    return sat


def gemm_pipeline(x_f: np.ndarray,
                  w_f: np.ndarray,
                  shift_bits: int = 24
                  ) -> dict:
    """
    Full software pipeline matching the hardware:
      1. derive QParams for X and W from their float ranges
      2. derive QParams for Y from float Y = x_f @ w_f range
      3. compute M, shift such that  s_x * s_w / s_y  ~=  M / 2^shift
      4. quantize X, W, run integer matmul, requantize -> Y_q
      5. also compute float Y for error analysis

    Returns dict with all artifacts.
    """
    qp_x = derive_qparams(x_f)
    qp_w = derive_qparams(w_f)
    y_f = x_f.astype(np.float32) @ w_f.astype(np.float32)
    qp_y = derive_qparams(y_f)

    real_mult = (qp_x.scale * qp_w.scale) / qp_y.scale
    M, shift = derive_requant_mshift(real_mult, shift_bits=shift_bits)

    x_q = quantize_int8(x_f, qp_x)
    w_q = quantize_int8(w_f, qp_w)
    acc = int_matmul_int32(x_q, w_q)
    y_q = requantize_int32_to_int8(acc, M, shift)

    # Error: dequantized hardware output vs float reference, in units of LSB.
    y_q_dq = y_q.astype(np.float32) * qp_y.scale
    err = y_q_dq - y_f
    err_lsb = err / qp_y.scale
    return {
        "qp_x": qp_x, "qp_w": qp_w, "qp_y": qp_y,
        "M": M, "shift": shift, "real_mult": real_mult,
        "x_q": x_q, "w_q": w_q,
        "acc": acc, "y_q": y_q,
        "y_f": y_f, "y_q_dq": y_q_dq,
        "err_lsb_max": float(np.max(np.abs(err_lsb))),
        "err_lsb_mean": float(np.mean(np.abs(err_lsb))),
    }


# ---------------------------------------------------------------------------
# Bit-packing to .mem / .hex
# ---------------------------------------------------------------------------

def pack_int8_lane_to_word(int8_vec: np.ndarray) -> int:
    """
    Pack a length-16 INT8 vector into a single 128-bit unsigned integer.
    Index i lives at bit positions [8*i +: 8].
    Two's-complement encoding.
    """
    if int8_vec.shape != (16,):
        raise ValueError(f"expected shape (16,), got {int8_vec.shape}")
    word = 0
    for i in range(16):
        v = int(int8_vec[i]) & 0xFF  # two's-complement byte
        word |= v << (8 * i)
    return word


def word_to_hex32(word: int) -> str:
    """128-bit unsigned int -> 32-char lowercase hex (no prefix)."""
    if word < 0 or word >= (1 << 128):
        raise ValueError("word out of range for 128 bits")
    return f"{word:032x}"


def write_mem_lines(path: str, lines: list[str], header: str = "") -> None:
    """Write one hex word per line, with an optional `// ` comment header."""
    with open(path, "w", encoding="ascii") as f:
        if header:
            for hl in header.splitlines():
                f.write(f"// {hl}\n")
        for ln in lines:
            f.write(ln + "\n")


def emit_weights_mem(w_q: np.ndarray, path: str) -> None:
    """
    Emit weight matrix Wo (16x16 INT8) as 16 lines of 128-bit hex.
    Line j = column j of Wo. Index i (row) at bit [8*i +: 8].
    """
    if w_q.shape != (16, 16) or w_q.dtype != np.int8:
        raise ValueError(f"expected (16,16) INT8, got {w_q.shape} {w_q.dtype}")
    lines = [word_to_hex32(pack_int8_lane_to_word(w_q[:, j])) for j in range(16)]
    write_mem_lines(path, lines,
                    header="weights.mem - 16 lines x 128-bit hex; "
                           "line j = column j of Wo (16x16 INT8). "
                           "Row i lives at bit [8*i +: 8].")


def emit_x_hex(x_q: np.ndarray, path: str) -> None:
    """
    Emit activations X (S x 16 INT8) as S lines of 128-bit hex.
    Line i = row i of X. Index k (col) at bit [8*k +: 8].
    """
    S, K = x_q.shape
    if K != 16 or x_q.dtype != np.int8:
        raise ValueError(f"expected (*,16) INT8, got {x_q.shape} {x_q.dtype}")
    lines = [word_to_hex32(pack_int8_lane_to_word(x_q[i, :])) for i in range(S)]
    write_mem_lines(path, lines,
                    header="x_vectors.hex - S lines x 128-bit hex; "
                           "line i = row i of X. Col k at bit [8*k +: 8].")


def emit_golden_y_hex(y_q: np.ndarray, path: str) -> None:
    """Same layout as X: line i = row i of Y_q."""
    S, N = y_q.shape
    if N != 16 or y_q.dtype != np.int8:
        raise ValueError(f"expected (*,16) INT8, got {y_q.shape} {y_q.dtype}")
    lines = [word_to_hex32(pack_int8_lane_to_word(y_q[i, :])) for i in range(S)]
    write_mem_lines(path, lines,
                    header="golden_y.hex - reference INT8 outputs. "
                           "Line i = row i of Y_q. Col j at bit [8*j +: 8].")


def emit_requant_txt(path: str, info: dict) -> None:
    """Human + machine-readable record of quant parameters."""
    qp_x = info["qp_x"]; qp_w = info["qp_w"]; qp_y = info["qp_y"]
    txt = (
        "# Requantization parameters (frozen)\n"
        f"s_x          = {qp_x.scale:.10e}\n"
        f"s_w          = {qp_w.scale:.10e}\n"
        f"s_y          = {qp_y.scale:.10e}\n"
        f"real_mult    = {info['real_mult']:.10e}\n"
        f"M            = {info['M']}\n"
        f"shift        = {info['shift']}\n"
        f"approx_mult  = {info['M'] / (1 << info['shift']):.10e}\n"
        f"err_lsb_max  = {info['err_lsb_max']:.6f}\n"
        f"err_lsb_mean = {info['err_lsb_mean']:.6f}\n"
    )
    with open(path, "w", encoding="ascii") as f:
        f.write(txt)
