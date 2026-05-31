"""
extract_and_quantize.py
-----------------------
Phase A driver. Produces the FROZEN reference artifacts consumed by the
RTL testbench and the on-board verification flow:

    artifacts/weights.mem
    artifacts/x_vectors.hex
    artifacts/golden_y.hex
    artifacts/requant.txt

Workload: one attention head's output projection Wo from `prajjwal1/bert-tiny`,
sliced to a 16x16 INT8 tile. Activation X is an 8x16 tensor.

Real bert-tiny weights are used IF the `transformers` + `torch` stack is
available locally. Otherwise we fall back to a deterministic seeded random
matrix that mimics the statistical properties of the real layer (zero-mean,
small std). Either way, results are reproducible (fixed seed).

The X tensor is generated as a deterministic seeded Gaussian (this is fine:
the assignment requires real *trained weights*; the activation just needs
to be a sensible test vector that exercises the datapath).

Usage:
    python sw/extract_and_quantize.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

# Make sw/ importable when run from repo root
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from golden_int8_gemm import (
    gemm_pipeline,
    emit_weights_mem, emit_x_hex, emit_golden_y_hex, emit_requant_txt,
)

REPO = os.path.abspath(os.path.join(HERE, os.pardir))
ART = os.path.join(REPO, "artifacts")
os.makedirs(ART, exist_ok=True)

SEED = 0xBEEF
TILE_K = 16   # d_head (and Wo rows)
TILE_N = 16   # d_out  (and Wo cols)
TILE_S = 8    # sequence length (X rows)


# ---------------------------------------------------------------------------
# Weight extraction: real bert-tiny if available, else seeded fallback
# ---------------------------------------------------------------------------

def try_load_bert_tiny_wo() -> tuple[np.ndarray, str]:
    """
    Try to load `prajjwal1/bert-tiny` and return a 16x16 slice of the
    output-projection weight of layer-0, head-0.

    bert-tiny: hidden=128, num_heads=2, d_head=64, num_layers=2.
    `attention.output.dense.weight` shape is [128, 128] (out, in).
    PyTorch nn.Linear stores weight as [out_features, in_features], and
    its forward computes  y = x @ W^T. So in our convention Wo (mathematical
    [in, out]) equals  W^T,  shape [128, 128] still.

    Head 0's contribution lives in input-feature columns [0:64) of the
    forward weight, which after transpose is rows [0:64) of Wo. We then
    take the top-left 16x16 sub-tile.
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoModel
    except Exception as e:
        return None, f"transformers/torch not available ({e.__class__.__name__})"

    try:
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    except Exception as e:
        return None, f"could not download bert-tiny ({e.__class__.__name__}: {e})"

    # Layer 0, attention output dense
    dense = model.encoder.layer[0].attention.output.dense
    W = dense.weight.detach().cpu().numpy()  # [out=128, in=128]
    # Wo (math) = W^T, so [in, out]. Head 0 occupies in-feature rows [0:64).
    Wo_full = W.T
    Wo = Wo_full[:TILE_K, :TILE_N].astype(np.float32).copy()
    return Wo, "loaded prajjwal1/bert-tiny attention.output.dense (layer 0, head 0 slice)"


def synthetic_wo() -> np.ndarray:
    """Deterministic fallback: small zero-mean Gaussian, std~=0.05."""
    rng = np.random.default_rng(SEED)
    return rng.normal(0.0, 0.05, size=(TILE_K, TILE_N)).astype(np.float32)


def get_wo() -> tuple[np.ndarray, str]:
    Wo, msg = try_load_bert_tiny_wo()
    if Wo is None:
        Wo = synthetic_wo()
        return Wo, f"WARN: using synthetic Wo ({msg})"
    return Wo, msg


# ---------------------------------------------------------------------------
# Activation X: deterministic seeded Gaussian
# ---------------------------------------------------------------------------

def make_x() -> np.ndarray:
    rng = np.random.default_rng(SEED ^ 0xA5A5)
    return rng.normal(0.0, 0.5, size=(TILE_S, TILE_K)).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("== Phase A: extract + quantize + golden ==")
    Wo, src = get_wo()
    print(f"[Wo] source: {src}")
    print(f"[Wo] shape={Wo.shape}, dtype={Wo.dtype}, "
          f"min={Wo.min():+.4f}, max={Wo.max():+.4f}, std={Wo.std():.4f}")

    X = make_x()
    print(f"[X ] shape={X.shape}, dtype={X.dtype}, "
          f"min={X.min():+.4f}, max={X.max():+.4f}, std={X.std():.4f}")

    info = gemm_pipeline(X, Wo, shift_bits=24)
    print(f"[Q ] s_x={info['qp_x'].scale:.4e}  "
          f"s_w={info['qp_w'].scale:.4e}  "
          f"s_y={info['qp_y'].scale:.4e}")
    print(f"[Q ] real_mult={info['real_mult']:.6e}  "
          f"M={info['M']}  shift={info['shift']}  "
          f"approx={info['M']/(1<<info['shift']):.6e}")
    print(f"[Q ] err_lsb_max={info['err_lsb_max']:.4f}  "
          f"err_lsb_mean={info['err_lsb_mean']:.4f}")

    # Sanity guards
    assert info["err_lsb_max"] <= 1.5, (
        f"max requant error {info['err_lsb_max']:.3f} LSB > 1.5; "
        "tighten s_y or increase shift_bits."
    )
    assert -(1 << 31) <= info["M"] < (1 << 31), "M out of INT32 range"

    # Emit artifacts
    p_w = os.path.join(ART, "weights.mem")
    p_x = os.path.join(ART, "x_vectors.hex")
    p_y = os.path.join(ART, "golden_y.hex")
    p_r = os.path.join(ART, "requant.txt")
    emit_weights_mem(info["w_q"], p_w)
    emit_x_hex(info["x_q"], p_x)
    emit_golden_y_hex(info["y_q"], p_y)
    emit_requant_txt(p_r, info)
    print(f"[OUT] {p_w}")
    print(f"[OUT] {p_x}")
    print(f"[OUT] {p_y}")
    print(f"[OUT] {p_r}")

    # Print golden Y_q for quick visual sanity
    print("[Y_q INT8] (rows = sequence positions):")
    for i, row in enumerate(info["y_q"]):
        print(f"  i={i}: {list(int(v) for v in row)}")

    print("== Phase A complete ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
