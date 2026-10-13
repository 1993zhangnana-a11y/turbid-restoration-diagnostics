"""Model checks (needs torch): identity at init, parameter counts reported in the paper,
strict loading of the finalised-baseline checkpoints (if TDIAG_WEIGHTS points to them)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from models import build_core, build_final_model, PriorWrapper, count_params, GEO_C

EXPECTED = {  # wrapped parameter counts (core + geometry heads), Table II of the paper
    ("nafnet", (2, 2, 4)): 904_901, ("nafnet", (1, 2, 4)): 888_453, ("nafnet", (2, 2, 7)): 1_261_637,
    ("restormer", (1, 3, 4)): 1_257_251, ("restormer", (2, 3, 4)): 1_281_893, ("restormer", (4, 3, 4)): 1_331_177,
    ("resunet", (2, 2, 1)): 937_797,
}


def test_counts_and_identity():
    for (fam, blk), n in EXPECTED.items():
        m = PriorWrapper(build_core(fam, 3, 32, blk)).eval()
        assert count_params(m) == n, (fam, blk, count_params(m), n)
        I = torch.rand(1, 3, 64, 64)
        with torch.inference_mode():
            o = m(I, torch.rand(1, GEO_C, 64, 64), torch.rand(1, 4))
        assert float((o - I).abs().max()) == 0.0
    assert count_params(build_final_model()) == 828_549


def test_load_final_weights():
    d = os.environ.get("TDIAG_WEIGHTS")
    if not d:
        return
    from models import load_final_model
    for s in (0, 1, 2):
        p = Path(d) / f"final_rgb_e60_s{s}.pt"
        if p.exists():
            load_final_model(str(p))


if __name__ == "__main__":
    test_counts_and_identity(); test_load_final_weights(); print("models: all checks passed")
