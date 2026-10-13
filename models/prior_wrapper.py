"""Injection of the spatial-localisation prior (tool geometry) at the bottleneck.

Geometry channels: [tool_mask, tool_edge, depth_norm, path_norm, confidence]. The SPADE layers of the core
receive zeros; the geometry enters only through two zero-initialised heads that produce multiplicative
modulations 1 + 0.1 tanh(.) at 1/4 and 1/8 resolution. The tool mask is Gaussian-blurred (sigma = 4 px).
With zero geometry input the wrapper is the plain core (used for the 'none', Wiener and state arms).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorWrapper(nn.Module):
    def __init__(self, core, content="full"):
        super().__init__()
        self.core = core; self.content = content
        c3 = core.fl.c                      # bottleneck width
        self.g4 = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, c3, 3, padding=1))
        self.g8 = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, c3, 3, padding=1))
        for g in (self.g4, self.g8):
            nn.init.zeros_(g[-1].weight); nn.init.zeros_(g[-1].bias)
        t = torch.arange(-12, 13, dtype=torch.float32); k = torch.exp(-.5 * (t / 4) ** 2); k = k / k.sum()
        self.register_buffer("kernel", (k[:, None] * k[None, :])[None, None])

    def forward(self, I, geo, prompt):
        c = self.core; blank = torch.zeros_like(geo)
        x = c.sp1(c.e1(c.inp(I)), blank); y = c.sp2(c.e2(c.d1(x)), blank)
        z = c.fl(c.mid(c.d2(y)), prompt)
        soft = F.conv2d(F.pad(geo[:, :1], (12, 12, 12, 12), mode="replicate"), self.kernel)
        if self.content == "coarse":
            g = torch.cat([soft, soft * geo[:, 2:3], geo[:, 3:4], torch.zeros_like(soft)], 1)
        else:
            g = torch.cat([soft, geo[:, 2:5]], 1)
        g4 = F.interpolate(g, size=z.shape[-2:], mode="area")
        g8 = F.interpolate(g, size=(z.shape[-2] // 2, z.shape[-1] // 2), mode="area")
        gamma4 = .1 * torch.tanh(self.g4(g4))
        gamma8 = F.interpolate(.1 * torch.tanh(self.g8(g8)), size=z.shape[-2:], mode="bilinear", align_corners=False)
        z = z * (1 + gamma4) * (1 + gamma8)
        y = c.f2(c.u2(z) + y); x = c.f1(c.u1(y) + x); out = I + c.out(x)
        return out if self.training else out.clamp(0, 1)
