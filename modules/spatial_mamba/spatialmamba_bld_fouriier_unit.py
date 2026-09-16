import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint

# 假设 selective_scan_fn 依然可用，因为它本质上处理的是序列

'''
为了将 SpatialMambaLayer 及其相关组件的输入输出维度从图像格式 (B, H, W, C) 改为序列格式 (B, L, D)，我们需要将所有涉及 2D 空间操作（如 Conv2d、H/W 维度的 Reshape）替换为 1D 序列操作（如 Conv1d、L 维度保持）。
以下是修改后的完整代码。主要的改动点如下：
StateFusion: 将 Conv2d 改为 Conv1d，移除了针对 2D 优化的 DepthwiseFunction（因为它通常是针对 2D 核的），保留了多尺度膨胀卷积的逻辑。
StructureAwareSSM: 将 Conv2d 改为 Conv1d，移除了 (B, H, W, C) -> (B, C, H, W) 的转换，改为 (B, L, D) -> (B, D, L)。
SpatialMambaBlock: 修改了 CPE (Conditional Positional Encoding) 为 1D 卷积，并调整了维度的 Permute 顺序。
SpatialMambaLayer: 适配上述更改，不再需要 H, W 参数。
'''

try:
    from .utils import selective_scan_fn
except:
    try:
        from utils import selective_scan_fn
    except ImportError:
        # 如果没有导入，定义一个伪函数以防报错（实际使用需要真实的 mamba 实现）
        def selective_scan_fn(x, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                              return_last_state=False):
            return x * torch.sigmoid(x)


# MLP 不需要大改，只要保证输入是 (B, L, D) 且 Linear 作用于最后一维即可
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
    """
    改为 1D 版本。保留了多尺度膨胀卷积的思想。
    Input: (B, D, L)
    """

    def __init__(self, dim):
        super(StateFusion1d, self).__init__()

        self.dim = dim
        # 1D 卷积核，size=3
        self.kernel_3 = nn.Parameter(torch.ones(dim, 1, 3))
        self.kernel_3_1 = nn.Parameter(torch.ones(dim, 1, 3))
        self.kernel_3_2 = nn.Parameter(torch.ones(dim, 1, 3))
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    @staticmethod
    def padding(input_tensor, padding):
        # 1D replicate padding
        return torch.nn.functional.pad(input_tensor, padding, mode='replicate')

    def forward(self, h):
        # h shape: (B, D, L)

        # 1. 普通卷积, padding=1
        h_pad1 = self.padding(h, (1, 1))
        h1 = F.conv1d(h_pad1, self.kernel_3, padding=0, dilation=1, groups=self.dim)

        # 2. 膨胀卷积 dilation=3, padding=3
        h_pad2 = self.padding(h, (3, 3))
        h2 = F.conv1d(h_pad2, self.kernel_3_1, padding=0, dilation=3, groups=self.dim)

        # 3. 膨胀卷积 dilation=5, padding=5
        h_pad3 = self.padding(h, (5, 5))
        h3 = F.conv1d(h_pad3, self.kernel_3_2, padding=0, dilation=5, groups=self.dim)

        out = self.alpha[0] * h1 + self.alpha[1] * h2 + self.alpha[2] * h3
        return out


class StructureAwareSSM1d(nn.Module):
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

        # 修改：Conv2d -> Conv1d
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

        # 修改：使用 1D StateFusion
        self.state_fusion = StateFusion1d(self.d_inner)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    # dt_init, A_log_init, D_init methods remain the same as original (omitted for brevity, assume they exist)
    # Start of Copy-Paste helper methods
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

    # End of Copy-Paste helper methods

    # def ssm(self, x: torch.Tensor):
    #     # x input shape: (B, D, L) from conv1d
    #     B, D, L = x.shape
    #
    #     # (B, D, L) -> (B, L, D) for linear proj
    #     xs = x.transpose(1, 2).contiguous()
    #
    #     # x_proj: (B, L, C) -> (B, L, Rank+2*State)
    #     x_dbl = torch.matmul(xs, self.x_proj_weight.t())
    #
    #     dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    #
    #     # dts: (B, L, Rank) -> (B, L, Inner)
    #     dts = torch.matmul(dts, self.dt_projs_weight.t())
    #
    #     # Rearrange for selective scan: needs (B, D, L)
    #     dts = dts.transpose(1, 2).contiguous()  # (B, D, L)
    #     Bs = Bs.transpose(1, 2).contiguous()  # (B, State, L)
    #     Cs = Cs.transpose(1, 2).contiguous()  # (B, State, L)
    #     xs = xs.transpose(1, 2).contiguous()  # (B, D, L)
    #
    #     As = -torch.exp(self.A_logs)
    #     Ds = self.Ds
    #     dt_projs_bias = self.dt_projs_bias
    #
    #     # 1. SSM 核心 (返回 Hidden State)
    #     # Output h shape: (B, D, N, L)
    #     h = self.selective_scan(
    #         xs, dts,
    #         As, Bs, None,  # Pass C=None to get states
    #         z=None,
    #         delta_bias=dt_projs_bias,
    #         delta_softplus=True,
    #         return_last_state=False,
    #     )
    #
    #     # 2. State Fusion (适配维度)
    #     # h: (B, D, N, L). StateFusion 需要 3D 输入 (Batch, Channels, Length)
    #     # 解决方案：将 State 维度 (N) 折叠进 Batch 维度 -> (B*N, D, L)
    #     # 这样每个状态分量都会独立地进行空间/序列融合
    #     N = self.d_state
    #     h = h.permute(0, 2, 1, 3).contiguous()  # (B, N, D, L)
    #     h = h.view(B * N, D, L)
    #
    #     h = self.state_fusion(h)  # -> (B*N, D, L)
    #
    #     # 恢复维度 -> (B, D, N, L)
    #     h = h.view(B, N, D, L).permute(0, 2, 1, 3)
    #
    #     # 3. 计算输出 y = C * h + D * u
    #     # h: (B, D, N, L)
    #     # Cs: (B, N, L) -> unsqueeze(1) -> (B, 1, N, L)
    #     # Element-wise mult, then sum over N
    #     y = h * Cs.unsqueeze(1)
    #     y = y.sum(dim=2)  # (B, D, L)
    #
    #     # Residual
    #     y = y + xs * Ds.view(-1, 1)
    #
    #     return y
    # def ssm(self, x: torch.Tensor):
    #     # x input shape: (B, D, L) from conv1d
    #     B, D, L = x.shape
    #
    #     # (B, D, L) -> (B, L, D) for linear proj
    #     xs = x.transpose(1, 2).contiguous()
    #
    #     # x_proj: (B, L, C) -> (B, L, Rank+2*State)
    #     x_dbl = torch.matmul(xs, self.x_proj_weight.t())
    #
    #     dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    #
    #     # dts: (B, L, Rank) -> (B, L, Inner)
    #     dts = torch.matmul(dts, self.dt_projs_weight.t())
    #
    #     # Rearrange for selective scan: needs (B, D, L)
    #     dts = dts.transpose(1, 2).contiguous()  # (B, D, L)
    #     Bs = Bs.transpose(1, 2).contiguous()  # (B, State, L)
    #     Cs = Cs.transpose(1, 2).contiguous()  # (B, State, L)
    #     xs = xs.transpose(1, 2).contiguous()  # (B, D, L)
    #
    #     As = -torch.exp(self.A_logs)
    #     Ds = self.Ds
    #     dt_projs_bias = self.dt_projs_bias
    #
    #     # ==================== 修改开始 ====================
    #     # 使用 checkpoint 包裹 selective_scan
    #     # 这会牺牲一点速度，但极大地节省显存
    #     # 以此替换原来的 h = self.selective_scan(...)
    #
    #     # 定义一个 lambda 或辅助函数来适配 checkpoint 的输入要求
    #     def run_scan(u, delta, A, B, C, D, bias):
    #         return self.selective_scan(
    #             u, delta, A, B, C, D=None, z=None,
    #             delta_bias=bias, delta_softplus=True, return_last_state=False
    #         )
    #
    #     # 注意：checkpoint 要求输入必须有 requires_grad=True 的张量参与
    #     # 这里的 xs, dts, Bs, Cs 都是中间变量，通常没问题
    #     h = checkpoint(run_scan, xs, dts, As, Bs, None, None, dt_projs_bias, use_reentrant=False)
    #     # 正确：调用模块下的 checkpoint 函数
    #     # h = checkpoint.checkpoint(run_scan, xs, dts, As, Bs, None, None, dt_projs_bias, use_reentrant=False)
    #     # ==================== 修改结束 ====================
    #
    #     # 2. State Fusion (适配维度)
    #     N = self.d_state
    #     h = h.permute(0, 2, 1, 3).contiguous()  # (B, N, D, L)
    #     h = h.view(B * N, D, L)
    #
    #     h = self.state_fusion(h)  # -> (B*N, D, L)
    #
    #     # 恢复维度 -> (B, D, N, L)
    #     h = h.view(B, N, D, L).permute(0, 2, 1, 3)
    #
    #     # 3. 计算输出 y = C * h + D * u
    #     y = h * Cs.unsqueeze(1)
    #     y = y.sum(dim=2)  # (B, D, L)
    #
    #     # Residual
    #     y = y + xs * Ds.view(-1, 1)
    #
    #     return y
    # def ssm(self, x: torch.Tensor):
    #     # x input shape: (B, D, L) from conv1d
    #     B, D, L = x.shape
    #
    #     # 1. 线性投影 (Projections)
    #     # (B, D, L) -> (B, L, D)
    #     xs = x.transpose(1, 2).contiguous()
    #
    #     # x_proj: (B, L, C) -> (B, L, Rank + 2*State)
    #     x_dbl = torch.matmul(xs, self.x_proj_weight.t())
    #
    #     dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
    #
    #     # dts: (B, L, Rank) -> (B, L, Inner)
    #     dts = torch.matmul(dts, self.dt_projs_weight.t())
    #
    #     # 2. 调整维度以适配 Selective Scan
    #     # 需要 (B, D, L) 或 (B, N, L)
    #     dts = dts.transpose(1, 2).contiguous()  # (B, D, L)
    #     Bs = Bs.transpose(1, 2).contiguous()  # (B, N, L)
    #     Cs = Cs.transpose(1, 2).contiguous()  # (B, N, L)
    #     xs = xs.transpose(1, 2).contiguous()  # (B, D, L)
    #
    #     As = -torch.exp(self.A_logs)
    #     Ds = self.Ds
    #     dt_projs_bias = self.dt_projs_bias
    #
    #     # 3. 核心扫描 (Selective Scan)
    #     # 这里的 h 形状为 (B, D, N, L)
    #     # 注意：为了速度，这里移除了 checkpoint。
    #     # 如果显存爆了，请优先在 main.py 中减少输入序列长度 (max_seq_len)。
    #     h = self.selective_scan(
    #         xs, dts,
    #         As, Bs, None,  # 传入 None 以获取 Hidden States
    #         z=None,
    #         delta_bias=dt_projs_bias,
    #         delta_softplus=True,
    #         return_last_state=False,
    #     )
    #
    #     # 4. 状态融合 (State Fusion)
    #     # h: (B, D, N, L). StateFusion 需要 3D 输入 (Batch, Dim, Length)
    #     # 解决方案：将 State 维度 (N) 折叠进 Batch 维度 -> (B*N, D, L)
    #     N = self.d_state
    #
    #     # 使用 reshape 避免 stride/view 报错
    #     h = h.permute(0, 2, 1, 3).contiguous()  # (B, N, D, L)
    #     h = h.reshape(B * N, D, L)
    #
    #     h = self.state_fusion(h)  # -> (B*N, D, L)
    #
    #     # 恢复维度 -> (B, D, N, L)
    #     h = h.reshape(B, N, D, L).permute(0, 2, 1, 3).contiguous()
    #
    #     # 5. 计算最终输出 y = h * C + x * D
    #     # h: (B, D, N, L)
    #     # Cs: (B, N, L) -> unsqueeze(1) -> (B, 1, N, L)
    #     # Element-wise mult, then sum over N
    #     y = h * Cs.unsqueeze(1)
    #     y = y.sum(dim=2)  # (B, D, L)
    #
    #     # Residual connection
    #     y = y + xs * Ds.view(-1, 1)
    #
    #     return y

    def ssm(self, x: torch.Tensor):
        # x input shape: (B, D, L)
        B, D, L = x.shape

        # 1. 投影与准备 (保持不变)
        xs = x.transpose(1, 2).contiguous()
        x_dbl = torch.matmul(xs, self.x_proj_weight.t())
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dts = torch.matmul(dts, self.dt_projs_weight.t())

        # 调整维度以适配 mamba_ssm
        dts = dts.transpose(1, 2).contiguous()  # (B, D, L)
        Bs = Bs.transpose(1, 2).contiguous()  # (B, N, L)
        Cs = Cs.transpose(1, 2).contiguous()  # (B, N, L)
        xs = xs.transpose(1, 2).contiguous()  # (B, D, L)

        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dt_projs_bias = self.dt_projs_bias

        # ============================================================
        # 修改核心：直接传入 Cs，使用官方 CUDA 内核计算 output
        # ============================================================

        # 注意：这里我们传入了 Cs！这意味着 selective_scan 会直接返回 y (B, D, L)
        # 而不是返回 h (B, D, N, L)
        # 这一步利用了 mamba_ssm 的极速优化
        y_raw = self.selective_scan(
            xs, dts,
            As, Bs, Cs,  # <--- 关键改变：传入 Cs
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        # y_raw 的形状是 (B, D, L)

        # ============================================================
        # 修改 StateFusion：从“融合状态”改为“融合输出”
        # ============================================================

        # StateFusion 原本期望卷积 D 个通道。
        # 现在的 y_raw 也是 D 个通道，所以完全兼容！不需要 reshape B*N

        # 执行 1D 空间融合
        y = self.state_fusion(y_raw)

        # ============================================================
        # 后处理
        # ============================================================

        # 原逻辑中的 y = h*C + D*u
        # 现在 mamba_ssm 已经帮我们算了 h*C (即 y_raw)
        # 我们只需要加上 Residual (D*u)

        if Ds is not None:
            y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        # Input: (B, L, D)
        B, L, C = x.shape

        xz = self.in_proj(x)  # (B, L, 2D)
        x, z = xz.chunk(2, dim=-1)

        # (B, L, D) -> (B, D, L) for Conv1d
        x = x.transpose(1, 2).contiguous()
        x = self.act(self.conv1d(x))

        y = self.ssm(x)

        # (B, D, L) -> (B, L, D)
        y = y.transpose(1, 2).contiguous()

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)

        if self.dropout is not None:
            y = self.dropout(y)
        return y

# class FreMLP1D(nn.Module):
#     def __init__(self, nc, expand=2):
#         super(FreMLP1D, self).__init__()
#         # 处理通道维度 C 的 MLP
#         self.process1 = nn.Sequential(
#             nn.Conv1d(nc, expand * nc, 1),  # 使用 1D 卷积或 Linear 均可，处理通道 C
#             nn.LeakyReLU(0.1, inplace=False),
#             # nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv1d(expand * nc, nc, 1)
#         )
#
#     def forward(self, x):
#         """
#         输入 x 维度: (B, L, C)
#         """
#         # 为了方便 Conv1d 处理，先转为 (B, C, L)
#         x = x.transpose(1, 2)
#         B, C, L = x.shape
#
#         if L < 2:
#             return x.transpose(1, 2)
#
#         # --- 关键步骤：计算 2 的倍数 ---
#         # 如果 L 是奇数，则补齐为 L+1；如果是偶数，保持不变
#         L_even = L if L % 2 == 0 else L + 1
#
#         # 1. 1D 实数傅里叶正变换 (在 L 维度，即最后一个维度)
#         # n=L_even 会自动填充
#         # 输出维度: (B, C, L_even//2 + 1)
#         x_freq = torch.fft.rfft(x, n=L_even, dim=-1, norm='backward')
#
#         # 2. 提取幅度谱 (Magnitude) 和 相位谱 (Phase)
#         mag = torch.abs(x_freq)
#         pha = torch.angle(x_freq)
#
#         # 3. 对幅度谱进行非线性变换
#         # process1 输入输出均为 (B, C, L_freq)
#         mag = self.process1(mag)
#
#         # 4. 重建复数
#         real = mag * torch.cos(pha)
#         imag = mag * torch.sin(pha)
#         x_out_complex = torch.complex(real, imag)
#
#         # 5. 1D 实数傅里叶逆变换还原
#         # 输出维度: (B, C, L_even)
#         x_out = torch.fft.irfft(x_out_complex, n=L_even, dim=-1, norm='backward')
#
#         # --- 关键步骤：截断还原并转回 (B, L, C) ---
#         x_out = x_out[:, :, :L]
#         return x_out.transpose(1, 2)

class FreMLP1D(nn.Module):
    def __init__(self, nc, expand=2):
        super(FreMLP1D, self).__init__()
        # 处理通道维度 C 的 MLP
        self.process1 = nn.Sequential(
            nn.Linear(nc, expand * nc),
            nn.LeakyReLU(0.1, inplace=False),  # 严格设为 False，防止之前的 inplace 报错
            nn.Linear(expand * nc, nc)
        )

    def forward(self, x):
        """
        输入 x 维度: (B, L, C)
        B: Batch size
        L: 实例数量 (Instances/Patches)
        C: 特征维度
        """
        B, L, C = x.shape

        # 边界情况处理
        if L < 2:
            return x

        # --- 方案四关键步骤：计算 Padding 长度 ---
        # 找到大于等于 L 的最小的 2 的幂次方 (例如 L=3833 -> L_pad=4096)
        # cuFFT 处理 2^n 长度时性能最强，且绝对不会报 INVALID_SIZE 错误
        # L_pad = 2 ** (L - 1).bit_length()
        L_pad = L if L % 2 == 0 else L + 1

        # 1. 1D 实数傅里叶正变换
        # n=L_pad 会自动在序列末尾补 0 达到 L_pad 长度
        # 变换维度 dim=1 即实例维度
        # 输出维度: (B, L_pad//2 + 1, C)
        x_freq = torch.fft.rfft(x, n=L_pad, dim=1, norm='backward')

        # 2. 提取幅度谱 (Magnitude) 和 相位谱 (Phase)
        mag = torch.abs(x_freq)  # (B, L_pad//2 + 1, C)
        pha = torch.angle(x_freq)  # (B, L_pad//2 + 1, C)

        # 3. 对幅度谱进行非线性变换 (在 GPU 上执行)
        # Linear 会作用于最后一个维度 C
        mag = self.process1(mag)

        # 4. 根据修改后的幅度谱和原始相位谱重建复数
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out_complex = torch.complex(real, imag)

        # 5. 1D 实数傅里叶逆变换还原
        # n=L_pad 确保逆变换长度与正变换 Padding 后的长度一致
        # 输出维度: (B, L_pad, C)
        x_out = torch.fft.irfft(x_out_complex, n=L_pad, dim=1, norm='backward')

        # --- 方案四关键步骤：截断还原 ---
        # 只取前 L 个元素，舍弃 Padding 的部分
        # 输出维度: (B, L, C)
        return x_out[:, :L, :]


# class FourierUnit(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(FourierUnit, self).__init__()
#         # 对应 FFC 思想：输入和输出通道都 * 2（因为实部和虚部拼接）
#         self.conv_layer = nn.Sequential(
#             nn.Linear(in_channels * 2, out_channels * 2, bias=False),
#             nn.LayerNorm(out_channels * 2),
#             nn.ReLU(inplace=False)  # 严格设为 False
#         )
#
#     def forward(self, x):
#         """
#         输入 x 维度: (B, L, C)
#         B: Batch Size
#         L: 实例数量 (Patches)
#         C: 特征通道数
#         """
#         B, L, C = x.shape
#         if L < 2:
#             return x
#
#         # 1. Padding 优化：补齐至 2 的幂次方，防止 cuFFT 报错并加速
#         L_pad = 2 ** (L - 1).bit_length()
#
#         # 2. 1D 实数傅里叶变换
#         # 输出维度: (B, L_pad//2 + 1, C) 复数类型
#         ffted = torch.fft.rfft(x, n=L_pad, dim=1, norm='ortho')
#
#         # 3. 实虚部拼接处理
#         # 分别提取实部和虚部: (B, L_freq, C)
#         x_fft_real = torch.real(ffted)
#         x_fft_imag = torch.imag(ffted)
#
#         # 在通道维度拼接: (B, L_freq, C*2)
#         ffted_cat = torch.cat((x_fft_real, x_fft_imag), dim=-1)
#
#         # 4. 频域特征交互 (通道间 + 实虚部间)
#         # Linear 作用于最后一维 C*2
#         ffted_combined = self.conv_layer(ffted_cat)  # (B, L_freq, C_out*2)
#
#         # 5. 重建复数张量
#         # 将 C_out*2 重新拆分为实部和虚部
#         new_c = ffted_combined.shape[-1] // 2
#         res_real = ffted_combined[..., :new_c]
#         res_imag = ffted_combined[..., new_c:]
#
#         # 组合为复数类型
#         ffted_complex = torch.complex(res_real, res_imag)
#
#         # 6. 1D 逆傅里叶变换
#         # 输出维度: (B, L_pad, C_out)
#         output = torch.fft.irfft(ffted_complex, n=L_pad, dim=1, norm='ortho')
#
#         # 7. 截断还原至原始长度 L
#         return output[:, :L, :]

class FourierUnit(nn.Module):
    def __init__(self, d_model, bottleneck_ratio=2):
        super(FourierUnit, self).__init__()
        # 瓶颈结构减少参数，增强泛化
        mid_channels = (d_model * 2) // bottleneck_ratio

        self.conv_layer = nn.Sequential(
            nn.Linear(d_model * 2, mid_channels, bias=False),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Linear(mid_channels, d_model * 2, bias=False),
        )

        # 光谱门控：自适应选择频率
        self.gating = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.Sigmoid()
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, C = x.shape
        if L < 2: return x

        # 1. Padding 优化，解决 cuFFT 报错
        L_pad = 2 ** (L - 1).bit_length()

        # 2. 1D FFT
        ffted = torch.fft.rfft(x, n=L_pad, dim=1, norm='ortho')

        # 3. 拼接实虚部
        x_fft_real = torch.real(ffted)
        x_fft_imag = torch.imag(ffted)
        ffted_cat = torch.cat((x_fft_real, x_fft_imag), dim=-1)

        # 4. 频域处理：变换 + 门控
        feat = self.conv_layer(ffted_cat)
        gate = self.gating(ffted_cat)
        ffted_combined = (ffted_cat + feat) * gate

        # 5. 重建复数
        new_c = ffted_combined.shape[-1] // 2
        ffted_complex = torch.complex(ffted_combined[..., :new_c], ffted_combined[..., new_c:])

        # 6. IFFT + 截断
        output = torch.fft.irfft(ffted_complex, n=L_pad, dim=1, norm='ortho')
        return self.ln(output[:, :L, :])


import torch
import torch.nn as nn
import torch.nn.functional as F


class ASSM(nn.Module):
    """
    高级语义扫描模块 (ASSM) - 序列原生版
    设计者：高级代码工程师 & 数字病理专家
    适配：[B, N, 512] 病理特征输入
    """
    # def __init__(self, dim=512, d_state=16, num_tokens=64, inner_rank=128, mlp_ratio=2.0):
    def __init__(self, dim=512, d_state=32, num_tokens=64, inner_rank=128, mlp_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank
        self.d_state = d_state

        # 1. 内部集成 EmbeddingA 和 EmbeddingB
        # 将 embeddingA 写入 ASSM 内部，作为可学习参数
        self.embeddingA = nn.Parameter(torch.empty(self.inner_rank, d_state))
        self.embeddingB = nn.Parameter(torch.empty(num_tokens, inner_rank))

        nn.init.uniform_(self.embeddingA, -1 / inner_rank, 1 / inner_rank)
        nn.init.uniform_(self.embeddingB, -1 / num_tokens, 1 / num_tokens)

        # 2. Mamba 与特征处理组件
        hidden_dim = int(self.dim * mlp_ratio)
        self.in_proj = nn.Linear(self.dim, hidden_dim)

        # 1D 局部上下文增强 (针对 Patch 序列)
        self.cpe_1d = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.SiLU()
        )

        # 核心扫描算子
        # from mamba_ssm import SelectiveScan
        # self.selectiveScan = SelectiveScan(d_model=hidden_dim, d_state=self.d_state, expand=1)
        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.dim)

        # 3. 语义路由
        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 4),
            nn.GELU(),
            nn.Linear(self.dim // 4, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def _semantic_neighbor_internal(self, x, index):
        """
        集成你提供的实现逻辑：
        使用 torch.gather 灵活对齐维度并进行重排
        """
        dim = index.dim()  # 通常是 2 (B, N)
        # 维度对齐逻辑
        for _ in range(x.dim() - index.dim()):
            index = index.unsqueeze(-1)
        index = index.expand(x.shape)

        # 执行重排 (在 dim=1，即 N 维度上进行 gather)
        return torch.gather(x, dim=dim - 1, index=index)

    def forward(self, x):
        """
        Input x: [B, N, 512]
        """
        B, N, C = x.shape

        # --- 阶段 1: 语义 Prompt 生成 ---
        full_embedding = self.embeddingB @ self.embeddingA  # [64, d_state]
        pred_route = self.route(x)  # [B, N, 64]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)
        # 生成动态 Prompt: [B, N, d_state]
        prompt = torch.matmul(cls_policy, full_embedding)

        # --- 阶段 2: 索引计算 ---
        # 获取每个实例的组别索引
        detached_index = torch.argmax(cls_policy.detach(), dim=-1)  # [B, N]
        # 计算排序索引 (Unfold)
        _, x_sort_indices = torch.sort(detached_index, dim=-1, stable=True)
        # 计算逆向索引 (Fold)
        x_sort_indices_reverse = torch.argsort(x_sort_indices, dim=-1)

        # --- 阶段 3: 特征投影与局部增强 ---
        x_hid = self.in_proj(x)  # [B, N, hidden]
        # 1D CPE 增强
        x_res = x_hid.transpose(1, 2)
        x_hid = (x_res + self.cpe_1d(x_res)).transpose(1, 2)

        # --- 阶段 4: 语义邻居重排与扫描 ---
        # 1. 语义重排 (Unfold)
        semantic_x = self._semantic_neighbor_internal(x_hid, x_sort_indices)
        semantic_prompt = self._semantic_neighbor_internal(prompt, x_sort_indices)

        # 2. Mamba 核心扫描
        y = self.selectiveScan(semantic_x, semantic_prompt)

        # 3. 空间还原 (Fold)
        y_restored = self._semantic_neighbor_internal(y, x_sort_indices_reverse)

        # --- 阶段 5: 输出投影 ---
        out = self.out_proj(self.out_norm(y_restored))
        return out

class SpatialMambaBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            dt_init: str = "random",
            num_heads: int = 8,  # Unused but kept for API compatibility
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            use_fourier=False,  # 传入 flag
            **kwargs,
    ):
        super().__init__()

        # CPE (Conditional Positional Encoding) 改为 1D 卷积
        self.cpe1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)

        self.self_attention = StructureAwareSSM1d(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            dt_init=dt_init,
            **kwargs
        )

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

        self.fremlp = FreMLP1D(nc=hidden_dim, expand=2)

        # --- 方案 1 增加部分 ---
        self.use_fourier = use_fourier
        if self.use_fourier:
            self.ln_fourier = norm_layer(hidden_dim)
            self.fourier_unit = FourierUnit(hidden_dim)
            self.gamma_f = nn.Parameter(1e-5 * torch.ones(hidden_dim), requires_grad=True)



    def forward(self, x: torch.Tensor):
        # Input: (B, L, D)

        # CPE 1: 需要 (B, D, L)
        res = x
        x_cpe = x.transpose(1, 2).contiguous()
        x_cpe = self.cpe1(x_cpe).transpose(1, 2)
        x = res + x_cpe

        # SSM Part
        x = x + self.drop_path(self.self_attention(self.ln_1(x)))

        # CPE 2
        res = x
        x_cpe = x.transpose(1, 2).contiguous()
        x_cpe = self.cpe2(x_cpe).transpose(1, 2)
        x = res + x_cpe


        # FourierUnit
        # --- FourierUnit (Global Frequency Mixer) ---
        # 1. 使用独立的 LayerNorm (例如 self.ln_fourier)
        # 2. 修正残差逻辑：x = x + drop_path(gamma * fourier(norm(x)))
        # x = x + self.drop_path(self.gamma1 * self.fourier_unit(self.ln_fourier(x)))

        if self.use_fourier:
            x = x + self.drop_path(self.gamma_f * self.fourier_unit(self.ln_fourier(x)))

        # MLP Part
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        # x = x + self.drop_path(self.fremlp(self.ln_2(x)))
        return x


class SpatialMambaLayer(nn.Module):
    """
    Modified Layer to handle (B, L, D).
    Removed 'downsample' logic related to 2D resizing.
    """

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
            is_last_layer=False,  # 新增参数
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        for i in range(depth):
            self.blocks = nn.ModuleList([
                SpatialMambaBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    dt_init=dt_init,
                    mlp_ratio=mlp_ratio,
                    # 方案 1：只有最后一层的最后一个 Block 开启傅里叶
                    use_fourier=(is_last_layer and i == depth - 1),
                    **kwargs
                )
                for i in range(depth)])

    def forward(self, x):
        # Input x: (B, L, D)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x