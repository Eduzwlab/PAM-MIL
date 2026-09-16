import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Callable
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint


try:
    from .utils import selective_scan_fn
except:
    try:
        from utils import selective_scan_fn
    except ImportError:
        def selective_scan_fn(x):
            return x * torch.sigmoid(x)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Input: (B, L, D)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StateFusion1d(nn.Module):
    def __init__(self, dim):
        super(StateFusion1d, self).__init__()

        self.dim = dim
        self.kernel_3 = nn.Parameter(torch.ones(dim, 1, 3))
        self.kernel_3_1 = nn.Parameter(torch.ones(dim, 1, 3))
        self.kernel_3_2 = nn.Parameter(torch.ones(dim, 1, 3))
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    @staticmethod
    def padding(input_tensor, padding):
        return torch.nn.functional.pad(input_tensor, padding, mode='replicate')

    def forward(self, h):

        h_pad1 = self.padding(h, (1, 1))
        h1 = F.conv1d(h_pad1, self.kernel_3, padding=0, dilation=1, groups=self.dim)


        h_pad2 = self.padding(h, (4, 4))
        h2 = F.conv1d(h_pad2, self.kernel_3_1, padding=0, dilation=4, groups=self.dim)

        h_pad3 = self.padding(h, (8, 8))
        h3 = F.conv1d(h_pad3, self.kernel_3_2, padding=0, dilation=8, groups=self.dim)

        out = self.alpha[0] * h1 + self.alpha[1] * h2 + self.alpha[2] * h3
        return out


class CrossGateSSA1d(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                     **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn

        self.state_fusion = StateFusion1d(self.d_inner)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                bias=True, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)
        if bias:
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init == "random" or "constant":
            A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n",
                       d=d_inner).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init == "simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init == "zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init == "random" or "constant":
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D)
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm(self, x: torch.Tensor):
        B, D, L = x.shape

        xs = x.transpose(1, 2).contiguous()
        x_dbl = torch.matmul(xs, self.x_proj_weight.t())
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dts = torch.matmul(dts, self.dt_projs_weight.t())

        dts = dts.transpose(1, 2).contiguous()  # (B, D, L)
        Bs = Bs.transpose(1, 2).contiguous()  # (B, N, L)
        Cs = Cs.transpose(1, 2).contiguous()  # (B, N, L)
        xs = xs.transpose(1, 2).contiguous()  # (B, D, L)

        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dt_projs_bias = self.dt_projs_bias

        y_raw = self.selective_scan(
            xs, dts,
            As, Bs, Cs,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        y = self.state_fusion(y_raw)

        if Ds is not None:
            y = y + xs * Ds.view(1, -1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):

        B, L, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)


        x = x.transpose(1, 2).contiguous()
        x = self.act(self.conv1d(x))

        y = self.ssm(x)

        y = y.transpose(1, 2).contiguous()
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)

        if self.dropout is not None:
            y = self.dropout(y)
        return y

class ASSM(nn.Module):

    def __init__(self,
                 dim=512,
                 d_state=32,
                 num_tokens=64,
                 inner_rank=128,
                 mlp_ratio=4.0,
                 dt_init="random",
                 dt_rank="auto",
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank
        self.d_state = d_state

        self.embeddingA = nn.Parameter(torch.empty(self.inner_rank, d_state))
        self.embeddingB = nn.Parameter(torch.empty(num_tokens, inner_rank))
        nn.init.uniform_(self.embeddingA, -1 / inner_rank, 1 / inner_rank)
        nn.init.uniform_(self.embeddingB, -1 / num_tokens, 1 / num_tokens)

        hidden_dim = int(self.dim * mlp_ratio)
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(self.dim, hidden_dim)

        self.dt_rank = math.ceil(self.dim / 16) if dt_rank == "auto" else dt_rank
        self.x_proj = nn.Linear(hidden_dim, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, hidden_dim, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(hidden_dim, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(hidden_dim))

        dt_min, dt_max = 0.001, 0.1
        if dt_init == "random":
            dt = torch.exp(torch.rand(hidden_dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        elif dt_init == "constant":
            dt = torch.tensor([0.1] * hidden_dim)  # 示例常量
        else:
            dt = torch.exp(torch.rand(hidden_dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))

        dt = dt.clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias.data.copy_(inv_dt)


        self.selectiveScan = selective_scan_fn
        self.cpe_1d = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.SiLU()
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.dim)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 4),
            nn.GELU(),
            nn.Linear(self.dim // 4, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )


    def _semantic_neighbor_internal(self, x, index):
        dim = index.dim()
        for _ in range(x.dim() - index.dim()):
            index = index.unsqueeze(-1)
        index = index.expand(x.shape)
        return torch.gather(x, dim=dim - 1, index=index)

    def forward(self, x):
        B, N, C = x.shape

        full_embedding = self.embeddingB @ self.embeddingA  # [64, d_state]
        pred_route = self.route(x)
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, N, 64]
        prompt = torch.matmul(cls_policy, full_embedding)  # [B, N, d_state]

        detached_index = torch.argmax(cls_policy.detach(), dim=-1)
        _, x_sort_indices = torch.sort(detached_index, dim=-1, stable=True)
        x_sort_indices_reverse = torch.argsort(x_sort_indices, dim=-1)

        x_hid = self.in_proj(x)
        x_res = x_hid.transpose(1, 2)
        x_hid = (x_res + self.cpe_1d(x_res)).transpose(1, 2)

        semantic_x = self._semantic_neighbor_internal(x_hid, x_sort_indices)

        x_dbl = self.x_proj(semantic_x)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dts = self.dt_proj(dts)

        Bs = Bs + prompt
        Cs = Cs + prompt

        u = semantic_x.transpose(1, 2).contiguous()
        delta = dts.transpose(1, 2).contiguous()
        A = -torch.exp(self.A_log.float())
        B_ssm = Bs.transpose(1, 2).contiguous()
        C_ssm = Cs.transpose(1, 2).contiguous()

        y_semantic = self.selectiveScan(
            u, delta, A, B_ssm, C_ssm,
            D=self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False
        )

        y_restored = self._semantic_neighbor_internal(y_semantic.transpose(1, 2), x_sort_indices_reverse)

        out = self.out_proj(self.out_norm(y_restored))
        return out


class GatedMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim * 2)
        self.gate_act = nn.SiLU()
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.norm(x)
        x_proj = self.proj(x)
        x_main, x_gate = x_proj.chunk(2, dim=-1)

        x_filtered = x_main * self.gate_act(x_gate)

        return self.out_proj(x_filtered)

class CrossGatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_s = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_a = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.proj_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, f_struct, f_seman):
        g_s = self.gate_s(f_struct)
        g_a = self.gate_a(f_seman)
        fused = (f_struct * g_a) + (f_seman * g_s)
        return self.proj_out(self.norm(fused))

class PSSMambaModule(nn.Module):

    def __init__(self, dim, d_state=32, dt_init="random", mlp_ratio=4.0,
                 num_tokens=64, attn_drop_rate=0.0, use_gated=True, **kwargs):
        super().__init__()
        self.use_gated = use_gated

        dt_rank = math.ceil(dim / 16)
        mamba_kwargs = {
            "d_state": d_state,
            "dt_init": dt_init,
            "dt_rank": dt_rank,
            "dropout": attn_drop_rate,
        }

        self.branch_struct = CrossGateSSA1d(
            d_model=dim,
            d_conv=3,
            expand=mlp_ratio,
            **mamba_kwargs,
            **kwargs
        )

        self.branch_seman = ASSM(
            dim=dim,
            num_tokens=num_tokens,
            inner_rank=128,
            mlp_ratio=mlp_ratio,
            **mamba_kwargs,
            **kwargs
        )

        self.gamma_s = nn.Parameter(1e-5 * torch.ones(dim), requires_grad=True)
        self.gamma_a = nn.Parameter(1e-5 * torch.ones(dim), requires_grad=True)

        if self.use_gated:
            self.ln_gated = nn.LayerNorm(dim)
            self.gated_mixer = GatedMixer(dim)

        self.fusion = CrossGatedFusion(dim)

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):

        f_s = self.branch_struct(x)
        f_a = self.branch_seman(x)

        if self.use_gated:
            f_a = f_a + self.gated_mixer(self.ln_gated(f_a))

        f_s = f_s * self.gamma_s
        f_a = f_a * self.gamma_a

        fused_x = self.fusion(f_s, f_a)

        return self.out_proj(fused_x)

from functools import partial
class LayerScale(nn.Module):
    """LayerScale for [B, L, C]"""
    def __init__(self, dim: int, init_value: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        return x * self.gamma.view(1, 1, -1)


class CCDGateScalar(nn.Module):

    def __init__(self, dim, hidden=128, temp=1.0, init_bias=-4.0, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.temp = float(temp)
        self.eps = eps

        # Conservative init: gate starts near 0
        with torch.no_grad():
            self.mlp[-1].bias.fill_(init_bias)  # sigmoid(-4) ~ 0.018

    @torch.no_grad()
    def _local_reliability(self, y):

        dy = torch.abs(y[:, :, 1:] - y[:, :, :-1])         # (B,D,L-1)
        dy = F.pad(dy, (1, 0), mode="replicate")           # (B,D,L)
        r = torch.exp(-dy).mean(dim=(1, 2), keepdim=False) # (B,)
        return r.unsqueeze(-1)                              # (B,1)

    def forward(self, y_raw):
        # slide-level channel summary
        g = y_raw.mean(dim=-1)            # (B,D)
        g = self.norm(g)
        s = self.mlp(g) / self.temp       # (B,1)
        r = self._local_reliability(y_raw) # (B,1)
        gate = torch.sigmoid(s) * r       # (B,1)
        return gate.unsqueeze(-1)         # (B,1,1)

class StateFusion1dCCD(nn.Module):
    def __init__(self, dim, dilations=(1, 3, 5), padding_mode="replicate"):
        super().__init__()
        self.dim = dim
        self.dilations = tuple(dilations)
        assert len(self.dilations) == 3, "dilations must be a 3-tuple like (1,3,5)"
        self.padding_mode = padding_mode

        self.kernel_0 = nn.Parameter(torch.ones(dim, 1, 3))
        self.kernel_1 = nn.Parameter(torch.ones(dim, 1, 3))
        self.kernel_2 = nn.Parameter(torch.ones(dim, 1, 3))
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    def _pad(self, x, pad):
        return F.pad(x, (pad, pad), mode=self.padding_mode)

    def forward(self, h):
        d0, d1, d2 = self.dilations
        h0 = F.conv1d(self._pad(h, d0), self.kernel_0, padding=0, dilation=d0, groups=self.dim)
        h1 = F.conv1d(self._pad(h, d1), self.kernel_1, padding=0, dilation=d1, groups=self.dim)
        h2 = F.conv1d(self._pad(h, d2), self.kernel_2, padding=0, dilation=d2, groups=self.dim)
        out = self.alpha[0] * h0 + self.alpha[1] * h1 + self.alpha[2] * h2
        return out

class CrossGateSSA1d_CCDv2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="constant",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        ccd_hidden=128,
        ccd_temp=1.0,
        ccd_gate_init_bias=-4.0,
        fusion_dilations=(1, 3, 5),
        fusion_lambda=0.05,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # (store weight and delete module, as in your implementation)
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(
            self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs
        )
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init, device=device)
        self.Ds = self.D_init(self.d_inner, dt_init, device=device)

        # mamba CUDA op
        self.selective_scan = selective_scan_fn

        # post
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # fused modules
        self.state_fusion = StateFusion1dCCD(self.d_inner, dilations=fusion_dilations, padding_mode="replicate")
        self.ccd_gate = CCDGateScalar(
            dim=self.d_inner,
            hidden=ccd_hidden,
            temp=ccd_temp,
            init_bias=ccd_gate_init_bias
        )
        self.fusion_lambda = float(fusion_lambda)

    # ---------------- helper inits ----------------
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="constant",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)
        if bias:
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank), device=dt_proj.weight.device))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner,), device=dt_proj.weight.device))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank), device=dt_proj.weight.device))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner,), device=dt_proj.weight.device))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init in ["random", "constant"]:
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init == "simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state), device=device))
        elif init == "zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state), device=device))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="constant", device=None):
        if init in ["random", "constant"]:
            D = nn.Parameter(torch.ones((d_inner,), device=device))
            D._no_weight_decay = True
        elif init in ["simple", "zero"]:
            D = nn.Parameter(torch.ones((d_inner,), device=device))
        else:
            raise NotImplementedError
        return D

    def ssm(self, x: torch.Tensor):

        xs = x.transpose(1, 2).contiguous()               # (B,L,D)
        x_dbl = torch.matmul(xs, self.x_proj_weight.t())  # (B,L,dt_rank+2*d_state)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dts = torch.matmul(dts, self.dt_projs_weight.t()) # (B,L,D)

        dts = dts.transpose(1, 2).contiguous()            # (B,D,L)
        Bs  = Bs.transpose(1, 2).contiguous()             # (B,N,L)
        Cs  = Cs.transpose(1, 2).contiguous()             # (B,N,L)
        xs  = xs.transpose(1, 2).contiguous()             # (B,D,L)

        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dt_bias = self.dt_projs_bias

        y_raw = self.selective_scan(
            xs, dts, As, Bs, Cs,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        y_sf = self.state_fusion(y_raw)
        gate = self.ccd_gate(y_raw)
        y = y_raw + self.fusion_lambda * gate * (y_sf - y_raw)

        if Ds is not None:
            y = y + xs * Ds.view(1, -1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_inner = x_inner.transpose(1, 2).contiguous()
        x_inner = self.act(self.conv1d(x_inner))

        y = self.ssm(x_inner)
        y = y.transpose(1, 2).contiguous()

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)

        if self.dropout is not None:
            y = self.dropout(y)
        return y

class AGSSMambaBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            dt_init: str = "random",
            num_heads: int = 8,
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            current_block_type="CrossGateSSA1d",
            is_last_layer=False,
            **kwargs,
    ):
        super().__init__()

        self.cpe1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)
        self.is_last_layer = is_last_layer

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.ln_fourier = norm_layer(hidden_dim)

        self.mlp = MLP(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            act_layer=mlp_act_layer,
            drop=mlp_drop_rate
        )

        self.current_block_type = current_block_type
        if self.current_block_type == "CrossGateSSA1d":
            self.self_attention = CrossGateSSA1d_CCDv2(
                d_model=hidden_dim,
                dropout=attn_drop_rate,
                d_state=d_state,
                dt_init=dt_init,
                **kwargs
            )
        else:
            self.self_attention = PSSMambaModule(
                dim=hidden_dim,
                d_state=d_state,
                dt_init=dt_init,
                mlp_ratio=mlp_ratio,
                num_tokens=64,
                attn_drop_rate=attn_drop_rate,
                use_gated=is_last_layer,
                **kwargs
            )

    def forward(self, x: torch.Tensor):

        res = x
        x_cpe = x.transpose(1, 2).contiguous()
        x_cpe = self.cpe1(x_cpe).transpose(1, 2)
        x = res + x_cpe

        x = x + self.drop_path(self.self_attention(self.ln_1(x)))

        res = x
        x_cpe = x.transpose(1, 2).contiguous()
        x_cpe = self.cpe2(x_cpe).transpose(1, 2)
        x = res + x_cpe

        x = x + self.drop_path(self.mlp(self.ln_2(x)))

        return x

class AGSSMambaLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        d_state=16,
        dt_init="random",
        mlp_ratio=4.0,
        current_block_type="CrossGateSSA1d",
        is_last_layer=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        if isinstance(drop_path, (list, tuple)):
            assert len(drop_path) == depth, f"drop_path list length {len(drop_path)} != depth {depth}"
            dpr_list = list(drop_path)
        else:
            dpr_list = [float(drop_path)] * depth

        self.blocks = nn.ModuleList([
            AGSSMambaBlock(
                hidden_dim=dim,
                drop_path=dpr_list[i],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                dt_init=dt_init,
                mlp_ratio=mlp_ratio,
                current_block_type=current_block_type,
                is_last_layer=is_last_layer,
                **kwargs
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x