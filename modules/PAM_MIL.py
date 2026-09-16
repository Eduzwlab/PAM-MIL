from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba.mamba_ssm import SRMamba, BiMamba, Mamba
except Exception as exc:
    raise ImportError(
        "Cannot import SRMamba/BiMamba/Mamba from mamba.mamba_ssm. "
        "Please check your project-specific mamba package path."
    ) from exc

try:
    from .spatial_mamba.spatialmamba_bld import AGSSMambaLayer
except Exception:
    from spatial_mamba.spatialmamba_bld import AGSSMambaLayer

CLASS_NAMES = ["Basal", "LumA", "LumB"]


# ════════════════════════════════════════════════════════════════════════════
# Initialization
# ════════════════════════════════════════════════════════════════════════════


def initialize_weights(module: nn.Module) -> None:
    """Stable initialization for small WSI soft-label datasets."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# ════════════════════════════════════════════════════════════════════════════
# Slide-level preprocessing
# ════════════════════════════════════════════════════════════════════════════


class SlidePreprocessor(nn.Module):
    """
    Per-slide feature standardization + Nmax token selection.

    The original code mixed regression optimization with random WSI token
    selection during validation/test in some versions.  For a ranking metric such
    as OvR-Macro AUC, this can create artificial epoch-to-epoch noise.  This
    module makes validation/test deterministic and allows deterministic training
    views as a reproducible alternative to random token augmentation.
    """

    def __init__(
            self,
            nmax: int = 4096,
            coverage_bins: int = 32,
            coverage_per_bin: int = 8,
            saliency_ratio: float = 0.25,
            norm_eps: float = 1e-6,
            clip_std: float = 6.0,
            train_sampling_mode: str = "deterministic",
            keep_order: bool = True,
    ) -> None:
        super().__init__()
        self.nmax = int(nmax)
        self.coverage_bins = int(coverage_bins)
        self.coverage_per_bin = int(coverage_per_bin)
        self.saliency_ratio = float(saliency_ratio)
        self.norm_eps = float(norm_eps)
        self.clip_std = float(clip_std)
        self.keep_order = bool(keep_order)
        if train_sampling_mode not in {"random", "deterministic", "off"}:
            raise ValueError(f"Unsupported train_sampling_mode: {train_sampling_mode}")
        self.train_sampling_mode = train_sampling_mode
        self.view_index = 0
        self.num_views = 1

    def set_eval_view(self, view_index: int = 0, num_views: int = 1) -> None:
        """Backward-compatible name used by old main scripts."""
        self.view_index = int(max(0, view_index))
        self.num_views = int(max(1, num_views))

    @torch.no_grad()
    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.norm_eps)
        if self.clip_std > 0:
            x = x.clamp(-self.clip_std, self.clip_std)
        return x

    @staticmethod
    def _unique_limited(idx: torch.Tensor, limit: int, sorted_order: bool) -> torch.Tensor:
        if idx.numel() == 0:
            return idx
        idx = torch.unique(idx, sorted=sorted_order)
        return idx[:limit]

    @torch.no_grad()
    def _coverage_indices_random(self, n: int, device: torch.device) -> torch.Tensor:
        bins = max(1, min(self.coverage_bins, n))
        per_bin = max(1, self.coverage_per_bin)
        chunks = []
        for i in range(bins):
            start = (i * n) // bins
            end = ((i + 1) * n) // bins
            width = end - start
            if width <= 0:
                continue
            k = min(per_bin, width)
            chunks.append(torch.randperm(width, device=device)[:k] + start)
        if not chunks:
            return torch.arange(min(n, self.nmax), device=device)
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def _coverage_indices_deterministic(self, n: int, device: torch.device) -> torch.Tensor:
        bins = max(1, min(self.coverage_bins, n))
        per_bin = max(1, self.coverage_per_bin)
        chunks = []
        for i in range(bins):
            start = (i * n) // bins
            end = ((i + 1) * n) // bins
            width = end - start
            if width <= 0:
                continue
            k = min(per_bin, width)
            if k == 1:
                local = torch.tensor([width // 2], device=device, dtype=torch.long)
            else:
                local = torch.linspace(0, width - 1, steps=k, device=device).long()

            # Deterministic shifted views. view 0 is the canonical deterministic view.
            if self.num_views > 1 and self.view_index > 0:
                stride = max(1, width // max(k, 1))
                shift = int(round((self.view_index % self.num_views) * stride / self.num_views))
                if shift > 0:
                    local = torch.remainder(local + shift, width)
                    local, _ = torch.sort(local)
            chunks.append(local + start)
        if not chunks:
            return torch.arange(min(n, self.nmax), device=device)
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def _fill_indices(
            self,
            base_idx: torch.Tensor,
            energy: torch.Tensor,
            n: int,
            deterministic: bool,
    ) -> torch.Tensor:
        device = energy.device
        selected = torch.zeros(n, dtype=torch.bool, device=device)
        selected[base_idx] = True
        idx = base_idx
        remain = self.nmax - int(selected.sum().item())
        if remain <= 0:
            return self._unique_limited(idx, self.nmax, self.keep_order)

        # Keep a modest high-energy quota, but avoid turning token selection into
        # top-k-only selection, which often overfits small molecular-subtype data.
        sal_k = min(int(round(remain * self.saliency_ratio)), n - int(selected.sum().item()))
        if sal_k > 0:
            e_masked = energy.masked_fill(selected, float("-inf"))
            top_idx = torch.topk(e_masked, k=sal_k, largest=True).indices
            selected[top_idx] = True
            idx = torch.cat([idx, top_idx], dim=0)
            remain = self.nmax - int(selected.sum().item())

        if remain > 0:
            pool = (~selected).nonzero(as_tuple=False).flatten()
            if pool.numel() > 0:
                k = min(remain, pool.numel())
                if deterministic:
                    pos = torch.linspace(0, pool.numel() - 1, steps=k, device=device).long()
                    if self.num_views > 1 and self.view_index > 0:
                        stride = max(1, pool.numel() // max(k, 1))
                        shift = int(round((self.view_index % self.num_views) * stride / self.num_views))
                        if shift > 0:
                            pos = torch.remainder(pos + shift, pool.numel())
                            pos, _ = torch.sort(pos)
                    fill = pool[pos]
                else:
                    fill = pool[torch.randperm(pool.numel(), device=device)[:k]]
                idx = torch.cat([idx, fill], dim=0)

        return self._unique_limited(idx, self.nmax, self.keep_order)

    @torch.no_grad()
    def _sample_one(self, x: torch.Tensor, deterministic: bool) -> torch.Tensor:
        n = int(x.shape[0])
        if self.nmax <= 0 or n <= self.nmax:
            return x
        energy = torch.norm(x, p=2, dim=-1)
        if deterministic:
            base = self._coverage_indices_deterministic(n, x.device)
        else:
            base = self._coverage_indices_random(n, x.device)
        idx = self._fill_indices(base, energy, n, deterministic=deterministic)
        return x[idx, :]

    @torch.no_grad()
    def _sample(self, x: torch.Tensor, deterministic: bool) -> torch.Tensor:
        if self.nmax <= 0 or x.shape[1] <= self.nmax:
            return x
        sampled = [self._sample_one(x[b], deterministic).unsqueeze(0) for b in range(x.shape[0])]
        return torch.cat(sampled, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self._standardize(x)

        if self.training:
            if self.train_sampling_mode == "off":
                return x
            return self._sample(x, deterministic=(self.train_sampling_mode == "deterministic"))

        # Validation/test must not call torch.randperm.
        return self._sample(x, deterministic=True)


# ════════════════════════════════════════════════════════════════════════════
# Attention pooling
# ════════════════════════════════════════════════════════════════════════════


class GatedAttentionPooling(nn.Module):
    """Subtype-specific gated attention pooling."""

    def __init__(self, dim: int = 512, hidden_dim: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout if dropout and dropout > 0 else 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_v = self.attention_v(x)
        a_u = self.attention_u(x)
        scores = self.attention_w(self.dropout(a_v * a_u))  # [B,N,1]
        attn = F.softmax(scores.transpose(1, 2), dim=-1)  # [B,1,N]
        pooled = torch.bmm(attn, x).squeeze(1)  # [B,C]
        return pooled, attn.squeeze(1)  # [B,C], [B,N]


# ════════════════════════════════════════════════════════════════════════════
# Main model
# ════════════════════════════════════════════════════════════════════════════


class PAM_MIL(nn.Module):
    """PAM_MIL for PAM50 soft-label regression."""

    CLASS_NAMES = CLASS_NAMES

    def __init__(
            self,
            in_dim: int = 1536,
            dropout: float = 0.25,
            act: str = "gelu",
            layer: int = 2,
            rate: int = 10,
            type: str = "spatial_mamba",
            train_sampling_mode: str = "deterministic",
            max_patches: int = 4096,
            coverage_bins: int = 32,
            coverage_per_bin: int = 8,
            saliency_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        self.type = str(type)
        self.rate = int(rate)
        self.layer = int(layer)

        proj = [nn.Linear(in_dim, 512)]
        act_l = act.lower()
        if act_l == "relu":
            proj.append(nn.ReLU())
        elif act_l == "gelu":
            proj.append(nn.GELU())
        else:
            raise ValueError(f"Unsupported activation: {act}")
        if dropout and dropout > 0:
            proj.append(nn.Dropout(dropout))
        self.patch_proj = nn.Sequential(*proj)

        self.layers = nn.ModuleList()
        self.final_norm = nn.LayerNorm(512)

        if self.type in {"SRMamba", "Mamba", "BiMamba"}:
            block_cls = {"SRMamba": SRMamba, "Mamba": Mamba, "BiMamba": BiMamba}[self.type]
            for _ in range(self.layer):
                self.layers.append(nn.ModuleDict({
                    "norm": nn.LayerNorm(512),
                    "block": block_cls(d_model=512, d_state=16, d_conv=4, expand=2),
                }))
        elif self.type == "PAM_MIL":
            depths = [2 for _ in range(self.layer)]
            drop_path_rate = 0.03
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
            for i in range(self.layer):
                self.layers.append(nn.ModuleDict({
                    "norm": nn.LayerNorm(512),
                    "block": AGSSMambaLayer(
                        dim=512,
                        depth=depths[i],
                        d_state=16,
                        dt_init="constant",
                        mlp_ratio=4.0,
                        drop=0.0,
                        attn_drop=0.0,
                        drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                        norm_layer=nn.LayerNorm,
                        downsample=None,
                        use_checkpoint=False,
                        current_block_type="StructureAwareSSM1d",
                        is_last_layer=(i == self.layer - 1),
                    ),
                }))
        else:
            raise NotImplementedError(f"Mamba type [{self.type}] is not implemented")

        self.attn_Basal = GatedAttentionPooling(dim=512, hidden_dim=256, dropout=dropout)
        self.attn_LumA = GatedAttentionPooling(dim=512, hidden_dim=256, dropout=dropout)
        self.attn_LumB = GatedAttentionPooling(dim=512, hidden_dim=256, dropout=dropout)

        self.head_Basal = nn.Linear(512, 1)
        self.head_LumA = nn.Linear(512, 1)
        self.head_LumB = nn.Linear(512, 1)

        self.preproc = SlidePreprocessor(
            nmax=max_patches,
            coverage_bins=coverage_bins,
            coverage_per_bin=coverage_per_bin,
            saliency_ratio=saliency_ratio,
            train_sampling_mode=train_sampling_mode,
            keep_order=True,
        )

        self.apply(initialize_weights)

    def _mamba_forward(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            residual = h
            h_norm = layer["norm"](h)
            block = layer["block"]
            if self.type == "SRMamba":
                h = block(h_norm, rate=self.rate)
            else:
                h = block(h_norm)
            h = h + residual
        return self.final_norm(h)

    def _forward_single(self, x: torch.Tensor):
        x = self.preproc(x)  # [B,N,D]
        h = self.patch_proj(x)  # [B,N,512]
        h = self._mamba_forward(h)  # [B,N,512]

        z_b, a_b = self.attn_Basal(h)
        z_la, a_la = self.attn_LumA(h)
        z_lb, a_lb = self.attn_LumB(h)

        pred_b = torch.sigmoid(self.head_Basal(z_b)).squeeze(-1)
        pred_la = torch.sigmoid(self.head_LumA(z_la)).squeeze(-1)
        pred_lb = torch.sigmoid(self.head_LumB(z_lb)).squeeze(-1)
        preds = torch.stack([pred_b, pred_la, pred_lb], dim=1)

        attn_dict: Dict[str, torch.Tensor] = {
            "Basal": a_b,
            "LumA": a_la,
            "LumB": a_lb,
        }
        return preds, attn_dict, {}

    def forward(self, x: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return self._forward_single(x.float())

    def forward_test(self, x: torch.Tensor) -> torch.Tensor:
        preds, _, _ = self.forward(x)
        return preds

    def get_attention_weights(self, x: torch.Tensor):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            preds, attn_dict, _ = self.forward(x)
        if was_training:
            self.train()
        return preds.squeeze(0), {key: value.squeeze(0) for key, value in attn_dict.items()}


if __name__ == "__main__":
    torch.manual_seed(42)
    model = MambaMILv2(
        in_dim=1536,
        dropout=0.25,
        act="gelu",
        layer=2,
        rate=10,
        type="SRMamba",
        train_sampling_mode="deterministic",
        max_patches=4096,
    )
    model.eval()
    x = torch.rand(1, 5000, 1536)
    p1, a1, _ = model(x)
    p2, a2, _ = model(x)
    assert p1.shape == (1, 3)
    assert torch.allclose(p1, p2), "Evaluation sampling must be deterministic."
    for k, v in a1.items():
        assert v.shape == (1, 4096), (k, v.shape)
        assert torch.allclose(v, a2[k]), k
        assert torch.allclose(v.sum(dim=1), torch.ones(1), atol=1e-4), k
    model.train()
    y = torch.tensor([[0.3, 0.2, 0.6]])
    p, _, _ = model(torch.rand(1, 300, 1536))
    loss = ((p - y) ** 2).mean()
    loss.backward()
    assert model.head_Basal.weight.grad is not None
    print("All smoke tests passed.")