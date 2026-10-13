"""Pair discovery and image loading."""
from __future__ import annotations
import csv, re
from pathlib import Path
import numpy as np
from PIL import Image

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LUMA = np.array([.299, .587, .114], np.float32)


def load(path, size=0, long_side=0, mode="RGB"):
    im = Image.open(path).convert(mode)
    if size:
        im = im.resize((size, size), Image.BILINEAR)
    elif long_side and max(im.size) > long_side:
        s = long_side / max(im.size)
        im = im.resize((max(4, round(im.size[0] * s / 4) * 4), max(4, round(im.size[1] * s / 4) * 4)), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255


def luma(a):
    a = np.asarray(a, np.float32)
    return (a * LUMA).sum(-1) if a.ndim == 3 and a.shape[-1] == 3 else a


def pair_dirs(degraded, clear, restored=None):
    """Same relative file stem in each directory tree -> one pair."""
    def index(root):
        root = Path(root)
        return {str(p.relative_to(root).with_suffix("")).replace("\\", "/"): p
                for p in sorted(root.rglob("*")) if p.suffix.lower() in EXT}
    ds, cs = index(degraded), index(clear); rs = index(restored) if restored else {}
    return [dict(name=k, group="all", degraded=str(ds[k]), clear=str(cs[k]), restored=str(rs.get(k, "")))
            for k in sorted(ds) if k in cs]


def pair_manifest(path):
    """CSV with columns degraded,clear[,restored][,group][,cluster][,name]."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return [dict(name=r.get("name") or f"row{i}", group=r.get("group", "all"), cluster=r.get("cluster", ""),
                     degraded=r["degraded"], clear=r["clear"], restored=r.get("restored", ""))
                for i, r in enumerate(csv.DictReader(fh))]


def pair_hazy_gt(raw):
    """NTIRE-style naming: '<idx>_*hazy*.<ext>' <-> '<idx>_*GT*.<ext>'. Returns [(idx, hazy, gt)]."""
    hz, gt = {}, {}
    for p in Path(raw).rglob("*"):
        if p.suffix.lower() not in EXT or "__MACOSX" in str(p):
            continue
        m = re.match(r"\D*(\d+)", p.stem)
        if not m:
            continue
        s = p.stem.lower(); k = int(m.group(1))
        if "hazy" in s:
            hz[k] = p
        elif "gt" in s or "clean" in s or "clear" in s:
            gt[k] = p
    return [(k, hz[k], gt[k]) for k in sorted(set(hz) & set(gt))]
