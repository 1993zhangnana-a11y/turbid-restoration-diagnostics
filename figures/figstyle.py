# -*- coding: utf-8 -*-
r"""Common style and export for all paper figures.

* Formats: PNG / PDF / EPS in one call. EPS is converted from PDF with `pdftops` (poppler) or Ghostscript,
  because matplotlib's PS backend does not support transparency and would turn the semi-transparent
  confidence bands into opaque blocks.
* Fonts: Times-like serif by default (Times New Roman, Nimbus Roman, TeX Gyre Termes, ...), TrueType
  embedding (fonttype 42) as required by IEEE; all text is rescaled uniformly by FIG_SIZE / 9.

Usage in a plotting script:
    import figstyle
    figstyle.savefig(fig, 'out/prefix')

Environment variables: FIG_FONT=times|helvetica|arial|dejavu, FIG_SIZE=9 (pt), FIG_DPI=200,
FIG_FORMATS=png,pdf,eps (svg also supported).
"""
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text

# ---------------------------------------------------------------- font presets
FAMILIES = {
    # name: (font.family, candidate fonts, mathtext set)
    'times':     ('serif', ['Times New Roman', 'Nimbus Roman', 'TeX Gyre Termes', 'Liberation Serif',
                            'DejaVu Serif'], 'stix'),
    'helvetica': ('sans-serif', ['Helvetica', 'Arial', 'Nimbus Sans',
                                 'Liberation Sans', 'DejaVu Sans'], 'stixsans'),
    'arial':     ('sans-serif', ['Arial', 'Liberation Sans', 'Nimbus Sans',
                                 'DejaVu Sans'], 'stixsans'),
    'dejavu':    ('sans-serif', ['DejaVu Sans'], 'dejavusans'),
}

# register TeX Gyre fonts if present (hosts without Times New Roman)
try:
    import glob as _glob
    from matplotlib import font_manager as _fm
    for _f in _glob.glob('/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyre*.otf'):
        _fm.fontManager.addfont(_f)
except Exception:
    pass

BASE_SIZE = 9.0          # reference font size used in the plotting scripts
_state = {'scale': 1.0, 'family': None}


def apply(font=None, size=None, dpi=None):
    """Set global rcParams (called once on import, from environment variables)."""
    font = (font or os.environ.get('FIG_FONT', 'times')).lower()
    size = float(size or os.environ.get('FIG_SIZE', BASE_SIZE))
    dpi = int(dpi or os.environ.get('FIG_DPI', 200))
    fam, cands, mathset = FAMILIES.get(font, FAMILIES['times'])

    plt.rcParams.update({
        'font.family': fam,
        ('font.serif' if fam == 'serif' else 'font.sans-serif'): cands,
        'font.size': size,
        'mathtext.fontset': mathset,
        'axes.titlesize': size * 1.10,
        'axes.labelsize': size * 1.02,
        'xtick.labelsize': size * 0.95,
        'ytick.labelsize': size * 0.95,
        'legend.fontsize': size * 0.92,
        'figure.titlesize': size * 1.30,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        # embed TrueType fonts (Type 42); avoid Type 3
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.unicode_minus': False,
        'axes.grid': False,
    })
    _state['scale'] = size / BASE_SIZE
    _state['family'] = cands
    return _state['scale']


def finalize(fig, scale=None, family=None):
    """Rescale all text uniformly and set the font family."""
    scale = _state['scale'] if scale is None else scale
    family = _state['family'] if family is None else family
    for t in fig.findobj(Text):
        try:
            if scale != 1.0:
                t.set_fontsize(t.get_fontsize() * scale)
            if family:
                t.set_fontfamily(family)
        except Exception:
            pass
    return fig


def _pdf_to_eps(pdf, eps):
    """pdf -> eps via pdftops (poppler) or Ghostscript."""
    for cmd in (['pdftops', '-eps', '-level3', str(pdf), str(eps)],
                ['gs', '-q', '-dNOPAUSE', '-dBATCH', '-dSAFER', '-dEPSCrop',
                 '-sDEVICE=eps2write', f'-sOutputFile={eps}', str(pdf)]):
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=120)
            if r.returncode == 0 and Path(eps).exists() and Path(eps).stat().st_size > 1024:
                return cmd[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def savefig(fig, out, formats=None, tight=True, scale=None, family=None):
    """Save in several formats; `out` is the path prefix without extension."""
    formats = formats or os.environ.get('FIG_FORMATS', 'png,pdf,eps').split(',')
    formats = [f.strip().lower() for f in formats if f.strip()]
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    finalize(fig, scale, family)
    kw = dict(bbox_inches='tight') if tight else {}

    made = []
    for f in ('png', 'pdf'):
        if f in formats or ('eps' in formats and f == 'pdf'):
            fig.savefig(f'{out}.{f}', **kw)
            if f in formats:
                made.append(f)
    if 'svg' in formats:
        fig.savefig(f'{out}.svg', **kw); made.append('svg')
    if 'eps' in formats:
        tool = _pdf_to_eps(f'{out}.pdf', f'{out}.eps')
        if tool:
            made.append(f'eps({tool})')
        else:
            # fallback: PS backend directly; transparency may be lost
            fig.savefig(f'{out}.eps', **kw)
            made.append('eps(matplotlib, transparency may be lost)')
        if 'pdf' not in formats:
            Path(f'{out}.pdf').unlink(missing_ok=True)
    print(f'[figstyle] {out.name}: {", ".join(made)}'
          f'  font={_state["family"][0] if _state["family"] else "default"}'
          f'  scale={_state["scale"]:.2f}')
    return made


apply()   # effective on import
