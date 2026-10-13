"""Command-line tool: decompose a set of paired images.

  python -m tdiag.cli --degraded DIR --clear DIR [--restored DIR] --out OUT
  python -m tdiag.cli --manifest pairs.csv --out OUT          # columns degraded,clear[,restored][,group][,cluster]

Outputs per_image.csv, summary.csv (means + cluster-bootstrap 95% CI, share as ratio of means), bands.csv.
By default PSNR is computed on Rec.601 luma at a fixed size; use --rgb --size 0 for the literature protocol
(native resolution, RGB).
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from . import decomp, lfr, stats
from .io import load, luma, pair_dirs, pair_manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--degraded"); ap.add_argument("--clear"); ap.add_argument("--restored")
    ap.add_argument("--manifest"); ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256, help="resize to size x size; 0 = native resolution")
    ap.add_argument("--rgb", action="store_true", help="PSNR / affine on RGB instead of luma")
    ap.add_argument("--mask-thr", type=float, default=-1.0, help="valid pixels: clear luma > thr (e.g. 0.02); <0 = all")
    ap.add_argument("--boot", type=int, default=4000)
    a = ap.parse_args(argv)
    units = pair_manifest(a.manifest) if a.manifest else pair_dirs(a.degraded, a.clear, a.restored)
    assert units, "no pairs found"
    rows, bands = [], []
    for u in units:
        X = load(u["degraded"], a.size); Y = load(u["clear"], a.size)
        O = load(u["restored"], a.size) if u.get("restored") else None
        x, y = (X, Y) if a.rgb else (luma(X), luma(Y))
        o = None if O is None else (O if a.rgb else luma(O))
        m2 = luma(Y) > a.mask_thr if a.mask_thr >= 0 else np.ones(Y.shape[:2], bool)
        r = dict(name=u["name"], group=u.get("group", "all"), cluster=u.get("cluster") or u["name"])
        r.update(decomp.decompose(x, y, o, m2))
        rows.append(r)
        xl, yl = luma(X), luma(Y)
        bc = lfr.band_ceiling(xl, yl, m2)
        bg = lfr.band_gain(xl, luma(O), yl, m2) if O is not None else [np.nan] * len(bc)
        for k, b in enumerate(bc):
            bands.append(dict(name=u["name"], group=r["group"], band=b["band"], rho=b["rho"],
                              lfr_db=b["ceiling_db"], gain_db=bg[k], headroom_db=b["ceiling_db"] - bg[k]))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    _write(out / "per_image.csv", rows); _write(out / "bands.csv", bands)
    summ = []
    for g in sorted({r["group"] for r in rows}):
        s = [r for r in rows if r["group"] == g]; cl = [r["cluster"] for r in s]
        d = dict(group=g, n=len(s))
        for k in ("psnr_in", "gain_affine", "gain_total", "gain_struct"):
            if k in s[0]:
                d[k], d[k + "_lo"], d[k + "_hi"], _ = stats.cluster_bootstrap_mean([r[k] for r in s], cl, a.boot)
        if "gain_total" in s[0]:
            d["share"], d["share_lo"], d["share_hi"] = stats.cluster_bootstrap_ratio(
                [r["gain_affine"] for r in s], [r["gain_total"] for r in s], cl, a.boot)
        summ.append(d)
    _write(out / "summary.csv", summ)
    print(json.dumps(summ, indent=1))


def _write(p, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
