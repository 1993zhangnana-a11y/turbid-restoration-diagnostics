"""U-shaped restoration core shared by the three backbone families studied in the paper.

The skeleton (attributes inp, e1, sp1, d1, e2, sp2, d2, mid, fl, u2, f2, u1, f1, out, gh) is identical for
every family; only the basic block differs:
  * 'nafnet'    gated convolution   (NAFBlock: SimpleGate + simplified channel attention)  [Chen et al., 2022]
  * 'restormer' channel attention   (MDTA + GDFN, bias-free LayerNorm)                      [Zamir et al., 2022]
  * 'resunet'   plain residual conv (conv3x3-ReLU-conv3x3, residual scale 0.1)            [Lim et al., 2017]
The zero-initialised output convolution makes every model an exact identity map at step 0.

build_final_model() is the paper's finalised baseline: NAFNet-Lite, RGB, base=32, blocks=(2,2,4),
828,549 parameters; parameter names match the released checkpoints (strict loading).
In the finalised configuration sp1/sp2 (geometry SPADE), fl (state FiLM) and gh (physics head) receive
zeros, but they are part of the weights and must not be removed.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _cp

GEO_C = 5          # geometry channels: tool_mask, tool_edge, depth_norm, path_norm, confidence(FOV)
PROMPT_C = 4       # state prompt: [tau/4, b/32, 1, 0]
USE_CKPT = os.environ.get('TDIAG_CKPT', '1') == '1'   # gradient checkpointing for transformer blocks


class SPADE(nn.Module):
    """geometry -> per-pixel (gamma, beta); zero-initialised => identity at init."""
    def __init__(self, c, geo_c=GEO_C, hidden=48):
        super().__init__()
        self.sh = nn.Sequential(nn.Conv2d(geo_c, hidden, 3, padding=1), nn.ReLU(True))
        self.g = nn.Conv2d(hidden, c, 3, padding=1)
        self.b = nn.Conv2d(hidden, c, 3, padding=1)
        self.n = nn.GroupNorm(1, c, affine=False)
        for m in (self.g, self.b):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, geo):
        g = F.interpolate(geo, size=x.shape[-2:], mode="bilinear", align_corners=False)
        h = self.sh(g)
        return self.n(x) * (1 + self.g(h)) + self.b(h)


class PromptFiLM(nn.Module):
    """global scalars (tau, b) -> channel-wise modulation; zero-initialised."""
    def __init__(self, c, p=PROMPT_C):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(p, 64), nn.ReLU(True), nn.Linear(64, 2 * c))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias); self.c = c

    def forward(self, x, p):
        gb = self.f(p)
        return x * (1 + gb[:, :self.c, None, None]) + gb[:, self.c:, None, None]


# ----------------------------------------------------------------------------- blocks
class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = nn.GroupNorm(1, c); self.n2 = nn.GroupNorm(1, c)
        self.c1 = nn.Conv2d(c, 2 * c, 1)
        self.dw = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.c2 = nn.Conv2d(c, c, 1)
        self.c3 = nn.Conv2d(c, 2 * c, 1); self.c4 = nn.Conv2d(c, c, 1)
        self.b1 = nn.Parameter(torch.zeros(1, c, 1, 1)); self.b2 = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.dw(self.c1(self.n1(x)))
        a, b = y.chunk(2, 1); y = a * b                      # SimpleGate
        y = self.c2(y * self.sca(y))
        x = x + y * self.b1
        y = self.c3(self.n2(x)); a, b = y.chunk(2, 1)
        return x + self.c4(a * b) * self.b2


class LayerNorm2d(nn.Module):
    """Restormer's bias-free LayerNorm over channels."""
    def __init__(self, c):
        super().__init__(); self.w = nn.Parameter(torch.ones(c))

    def forward(self, x):
        s = x.var(1, keepdim=True, unbiased=False)
        return x / torch.sqrt(s + 1e-5) * self.w[None, :, None, None]


class MDTA(nn.Module):
    """Multi-Dconv head transposed attention: the attention map is C x C, not HW x HW."""
    def __init__(self, c, heads):
        super().__init__()
        self.h = heads; self.t = nn.Parameter(torch.ones(heads, 1, 1)); self.n = LayerNorm2d(c)
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.dw = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3, bias=False)
        self.pj = nn.Conv2d(c, c, 1, bias=False); nn.init.zeros_(self.pj.weight)

    def forward(self, x):
        b, c, hh, ww = x.shape
        q, k, v = self.dw(self.qkv(self.n(x))).chunk(3, 1)
        r = lambda z: z.reshape(b, self.h, c // self.h, hh * ww)
        q, k, v = r(q), r(k), r(v)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        a = (q @ k.transpose(-2, -1)) * self.t
        return x + self.pj((a.softmax(-1) @ v).reshape(b, c, hh, ww))


class GDFN(nn.Module):
    def __init__(self, c, expand=2.0):
        super().__init__()
        hid = int(c * expand); self.n = LayerNorm2d(c)
        self.pw = nn.Conv2d(c, hid * 2, 1, bias=False)
        self.dw = nn.Conv2d(hid * 2, hid * 2, 3, padding=1, groups=hid * 2, bias=False)
        self.pj = nn.Conv2d(hid, c, 1, bias=False); nn.init.zeros_(self.pj.weight)

    def forward(self, x):
        a, b = self.dw(self.pw(self.n(x))).chunk(2, 1)
        return x + self.pj(F.gelu(a) * b)


class TBlock(nn.Module):
    def __init__(self, c, heads):
        super().__init__(); self.a = MDTA(c, heads); self.f = GDFN(c)

    def _run(self, x):
        return self.f(self.a(x))

    def forward(self, x):   # gradient checkpointing is mathematically identical, saves memory
        if USE_CKPT and self.training and x.requires_grad:
            return _cp.checkpoint(self._run, x, use_reentrant=False)
        return self._run(x)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c1 = nn.Conv2d(c, c, 3, padding=1); self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + 0.1 * self.c2(F.relu(self.c1(x)))


# ----------------------------------------------------------------------------- core
class UNetCore(nn.Module):
    def __init__(self, family="nafnet", in_ch=3, base=32, blocks=(2, 2, 4), geo_c=GEO_C, prompt_c=PROMPT_C,
                 head="residual", heads=(1, 2, 4)):
        super().__init__()
        self.head_kind = head; self.family = family
        c1, c2, c3 = base, base * 2, base * 4
        if family == "nafnet":
            mk = [lambda c: NAFBlock(c)] * 3
        elif family == "restormer":
            mk = [lambda c, h=h: TBlock(c, h) for h in heads]
        elif family == "resunet":
            mk = [lambda c: ResBlock(c)] * 3
        else:
            raise ValueError(family)
        self.inp = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.e1 = nn.Sequential(*[mk[0](c1) for _ in range(blocks[0])])
        self.d1 = nn.Conv2d(c1, c2, 2, stride=2)
        self.e2 = nn.Sequential(*[mk[1](c2) for _ in range(blocks[1])])
        self.d2 = nn.Conv2d(c2, c3, 2, stride=2)
        self.mid = nn.Sequential(*[mk[2](c3) for _ in range(blocks[2])])
        self.u2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.f2 = nn.Sequential(*[mk[1](c2) for _ in range(blocks[1])])
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.f1 = nn.Sequential(*[mk[0](c1) for _ in range(blocks[0])])
        self.out = nn.Conv2d(c1, in_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.sp1 = SPADE(c1, geo_c); self.sp2 = SPADE(c2, geo_c); self.fl = PromptFiLM(c3, prompt_c)
        self.gh = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c3, 64), nn.ReLU(True),
                                nn.Linear(64, 2))
        nn.init.zeros_(self.gh[-1].weight); nn.init.zeros_(self.gh[-1].bias)

    def forward(self, I, geo=None, prompt=None):
        B = I.shape[0]
        if geo is None:
            geo = torch.zeros(B, GEO_C, 8, 8, device=I.device, dtype=I.dtype)
        if prompt is None:
            prompt = torch.zeros(B, PROMPT_C, device=I.device, dtype=I.dtype)
        x = self.sp1(self.e1(self.inp(I)), geo)
        y = self.sp2(self.e2(self.d1(x)), geo)
        z = self.fl(self.mid(self.d2(y)), prompt)
        y = self.f2(self.u2(z) + y); x = self.f1(self.u1(y) + x)
        o = I + self.out(x)
        return o if self.training else o.clamp(0, 1)


def build_core(family="nafnet", in_ch=3, base=32, blocks=None, prompt_c=PROMPT_C):
    blocks = blocks or {"nafnet": (2, 2, 4), "restormer": (1, 3, 4), "resunet": (2, 2, 1)}[family]
    return UNetCore(family, in_ch, base, tuple(blocks), GEO_C, prompt_c)


def build_final_model(in_ch=3):
    return UNetCore("nafnet", in_ch, 32, (2, 2, 4))


def load_final_model(checkpoint, in_ch=3, device="cpu"):
    m = build_final_model(in_ch)
    sd = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def count_params(m):
    return sum(p.numel() for p in m.parameters())
