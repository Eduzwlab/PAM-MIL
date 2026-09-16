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

class AdaptiveStateFusion1d(nn.Module):
    """
    强化版 1D 状态融合模块 (针对 3 分类分子分型优化)
    1. 动态尺度选择 (Selective Kernel)：根据特征自适应调整不同膨胀率的权重。
    2. 通道交互：引入 Pointwise 卷积，学习维度间的分子关联。
    3. 残差门控：通过残差连接防止深层特征退化。
    """

    def __init__(self, dim=512, reduction=8):
        super(AdaptiveStateFusion1d, self).__init__()
        self.dim = dim

        # 1. 多尺度空间聚合 (Depthwise)
        # 尺度 1: 捕获细胞级微观细节 (3x3, d=1)
        self.branch1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, dilation=1, groups=dim, bias=False)
        # 尺度 2: 捕获组织级局部架构 (3x3, d=3)
        self.branch2 = nn.Conv1d(dim, dim, kernel_size=3, padding=3, dilation=3, groups=dim, bias=False)
        # 尺度 3: 捕获更广泛的区域背景 (3x3, d=5)
        self.branch3 = nn.Conv1d(dim, dim, kernel_size=3, padding=5, dilation=5, groups=dim, bias=False)

        # 2. 动态权重生成器 (Selective Kernel Attention)
        # 融合三个分支的信息，决定哪个尺度对当前分型更重要
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(dim, dim // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // reduction, dim * 3, kernel_size=1, bias=False)  # 输出 3 个权重的通道
        )

        # 3. 通道交互投影 (Pointwise)
        # 学习不同特征维度之间的分子信号组合
        self.pointwise = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim),
            nn.SiLU()
        )

        # 学习型残差权重
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, h):
        # h: (B, D, L)
        res = h

        # --- 步骤 1: 并行多尺度卷积 ---
        f1 = self.branch1(h)
        f2 = self.branch2(h)
        f3 = self.branch3(h)

        # --- 步骤 2: 自适应尺度融合 (Selective Kernel) ---
        # 聚合所有分支信息
        f_sum = f1 + f2 + f3
        s = self.gap(f_sum)  # 全局信息 [B, D, 1]
        z = self.fc(s)  # [B, D*3, 1]

        # 拆分并生成 softmax 权重
        z = z.view(z.shape[0], 3, self.dim, 1)  # [B, 3, D, 1]
        z = F.softmax(z, dim=1)  # 在 3 个尺度维度上进行归一化

        # 动态加权
        h_fused = f1 * z[:, 0] + f2 * z[:, 1] + f3 * z[:, 2]

        # --- 步骤 3: 通道交互与提纯 ---
        # 原版缺乏通道间的混合，这里补齐
        h_fused = self.pointwise(h_fused)

        # --- 步骤 4: 门控残差连接 ---
        # 使用学习型 alpha 确保初始化时是恒等映射，训练平滑
        return res + self.alpha * h_fused


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
        # self.state_fusion = AdaptiveStateFusion1d(self.d_inner)

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


from einops import repeat


class StructureAwareSSM1d_v2(nn.Module):
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
            bidirectional=True,  # 新增：双向扫描开关
            **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.bidirectional = bidirectional

        # 1. 输入投影：一次性生成分支特征
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        # 2. 局部卷积：保持维度
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        # 3. SSM 参数投影
        # 如果是双向，我们需要为反向扫描准备一套参数，或者共享参数
        # 专家建议：为双向扫描分配独立的 dt, B, C 以捕捉不同方向的依赖
        ssm_out_dim = self.dt_rank + self.d_state * 2
        self.x_proj = nn.Linear(self.d_inner, ssm_out_dim, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A, D 初始化
        self.A_logs = self.A_log_init(self.d_state, self.d_inner)
        self.Ds = self.D_init(self.d_inner)

        # 官方内核
        self.selective_scan = selective_scan_fn

        # 4. 结构感知融合 (改进：增加残差结构)
        self.state_fusion = StateFusion1d(self.d_inner)

        # 5. 输出
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def A_log_init(d_state, d_inner):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        return nn.Parameter(A_log)

    @staticmethod
    def D_init(d_inner):
        return nn.Parameter(torch.ones(d_inner))

    def forward_ssm(self, x_inner):
        """核心扫描逻辑的封装，支持正反向调用"""
        B, L, D = x_inner.shape

        # 参数投影
        x_dbl = self.x_proj(x_inner)  # (B, L, Rank + 2*State)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt).transpose(1, 2).contiguous()  # (B, D, L)
        B_ssm = B_ssm.transpose(1, 2).contiguous()  # (B, N, L)
        C_ssm = C_ssm.transpose(1, 2).contiguous()  # (B, N, L)
        u = x_inner.transpose(1, 2).contiguous()  # (B, D, L)

        As = -torch.exp(self.A_logs.float())

        y = self.selective_scan(
            u, dt, As, B_ssm, C_ssm,
            z=None, delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True, return_last_state=False
        )
        return y  # (B, D, L)

    def forward(self, x):
        # x: (B, L, D)
        B, L, C = x.shape

        # 1. 输入投影与门控分支
        xz = self.in_proj(x)  # (B, L, 2*D_inner)
        x, z = xz.chunk(2, dim=-1)

        # 2. 局部卷积处理
        x_conv = x.transpose(1, 2).contiguous()
        x_conv = self.act(self.conv1d(x_conv))  # (B, D_inner, L)
        x_inner = x_conv.transpose(1, 2).contiguous()  # (B, L, D_inner)

        # 3. 双向 SSM 扫描
        # 正向扫描
        y_fwd = self.forward_ssm(x_inner)

        if self.bidirectional:
            # 反向扫描：翻转序列 -> 扫描 -> 翻转回
            x_bwd = x_inner.flip(dims=[1])
            y_bwd = self.forward_ssm(x_bwd)
            y_bwd = y_bwd.flip(dims=[2])
            y_combined = (y_fwd + y_bwd) * 0.5  # 均值融合
        else:
            y_combined = y_fwd

        # 4. 结构感知融合 (StateFusion)
        # 将 SSM 的扫描输出再次通过卷积进行空间精修
        y_fused = self.state_fusion(y_combined)

        # 5. D-路径残差连接与门控输出
        # y_fused: (B, D_inner, L)
        y = y_fused.transpose(1, 2).contiguous()  # (B, L, D_inner)

        # 结合 D 路径残差
        y = y + x_inner * self.Ds.view(1, 1, -1)

        # 门控输出：结合 z 分支并完成最后的线性投影
        y = self.out_norm(y)
        y = y * F.silu(z)  # Mamba 经典的门控结构
        y = self.out_proj(y)

        return self.dropout(y) if self.dropout else y


class StructureAwareSSM1d_v3(nn.Module):
    """
    高级结构感知双向 SSM 模块 (V2 优化版)
    设计：高级代码工程师 & 数字病理专家
    特性：
    1. 非因果性 (Non-causal)：通过双向扫描消除空间扫描偏见，实现全局视野。
    2. 参数独立化：正向与反向扫描拥有独立的 dt, B, C 参数，增强形态学建模能力。
    3. 内核优化：直接传入 Cs 使用 CUDA 内核，避免大内存占用。
    4. 结构融合：集成 StateFusion 强化 Patch 间的拓扑关联。
    """

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
            bidirectional=True,  # 默认开启双向以获得非因果属性
            **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.bidirectional = bidirectional

        # 1. 维度投影与局部卷积
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        # 2. SSM 参数投影 (针对双向扫描进行独立或对齐设计)
        # 为正向扫描准备的 x_proj
        self.x_proj_fwd = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj_fwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        if self.bidirectional:
            # 专家级优化：为反向扫描分配独立参数，捕捉反向空间依赖
            self.x_proj_bwd = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj_bwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 3. 核心算子参数 (A 矩阵与 D 路径)
        # A 矩阵初始化为 S4D 结构
        self.A_logs = self.A_log_init(self.d_state, self.d_inner)
        self.Ds = self.D_init(self.d_inner)

        # dt 初始化权重 (符合官方 Mamba 动力学)
        self._dt_init(self.dt_proj_fwd, dt_min, dt_max, dt_init_floor)
        if self.bidirectional:
            self._dt_init(self.dt_proj_bwd, dt_min, dt_max, dt_init_floor)

        # 4. 算子与融合
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        self.selective_scan = selective_scan_fn
        self.state_fusion = StateFusion1d(self.d_inner)  # 需确保外部定义

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def _dt_init(dt_proj, dt_min, dt_max, dt_init_floor):
        d_inner = dt_proj.out_features
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

    @staticmethod
    def A_log_init(d_state, d_inner):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=d_inner).contiguous()
        return nn.Parameter(torch.log(A))

    @staticmethod
    def D_init(d_inner):
        return nn.Parameter(torch.ones(d_inner))

    def core_ssm(self, u, x_proj, dt_proj):
        """执行单向扫描的核心逻辑"""
        # u: (B, D_inner, L)
        # x_inner: (B, L, D_inner)
        x_inner = u.transpose(1, 2).contiguous()
        x_dbl = x_proj(x_inner)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # 准备 selective_scan 参数
        dt = dt_proj(dt).transpose(1, 2).contiguous()  # (B, D, L)
        B = B.transpose(1, 2).contiguous()  # (B, N, L)
        C = C.transpose(1, 2).contiguous()  # (B, N, L)
        A = -torch.exp(self.A_logs.float())  # (D, N)

        # 官方内核直接扫描 (返回 y = C * h)
        y = self.selective_scan(
            u, dt, A, B, C,
            D=self.Ds.float(),
            z=None,
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False
        )
        return y

    def forward(self, x):
        """
        Input x: (B, L, D)
        Output: (B, L, D)
        """
        B, L, C = x.shape

        # 1. 输入投影
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, L, D_inner)

        # 2. 局部卷积增强 (空间信息预处理)
        x_conv = x.transpose(1, 2).contiguous()
        x_conv = self.act(self.conv1d(x_conv))  # (B, D_inner, L)

        x_inner = x_conv.transpose(1, 2).contiguous()  # (B, L, D_inner)

        # 3. 双向扫描实现非因果属性
        # 正向路径
        y_fwd = self.core_ssm(x_conv, self.x_proj_fwd, self.dt_proj_fwd)

        # if self.bidirectional:
        #     # 反向路径：翻转序列 -> 扫描 -> 翻转还原
        #     x_bwd_in = x_conv.flip(dims=[2])  # 在长度维度 L 翻转
        #     y_bwd_raw = self.core_ssm(x_bwd_in, self.x_proj_bwd, self.dt_proj_bwd)
        #     y_bwd = y_bwd_raw.flip(dims=[2])  # 翻转回原始位置
        #
        #     # 融合正反向特征 (非因果全图可见性)
        #     y = (y_fwd + y_bwd) * 0.5
        # else:
        #     y = y_fwd

        if self.bidirectional:
            # 改进：反向路径直接复用正向的投影参数 (Weight Sharing)
            # 这能起到极强的正则化作用，防止分子分型任务过拟合
            x_bwd_in = x_conv.flip(dims=[2])
            y_bwd_raw = self.core_ssm(x_bwd_in, self.x_proj_fwd, self.dt_proj_fwd)
            y_bwd = y_bwd_raw.flip(dims=[2])

            y = (y_fwd + y_bwd) * 0.5
        else:
            y = y_fwd

        # 4. 结构感知融合与输出门控
        # y: (B, D_inner, L)
        y_fused = self.state_fusion(y)  # (B, D_inner, L)

        y = y_fused.transpose(1, 2).contiguous()  # (B, L, D_inner)

        # 结合 D 路径残差
        # y = y + x_inner * self.Ds.view(1, 1, -1)

        y = self.out_norm(y)

        # 结合门控分支 z 并投影回原维度
        y = y * F.silu(z)
        y = self.out_proj(y)

        return self.dropout(y) if self.dropout else y



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


# class ASSM(nn.Module):
#     """
#     高级语义扫描模块 (ASSM) - 序列原生版
#     设计者：高级代码工程师 & 数字病理专家
#     适配：[B, N, 512] 病理特征输入
#     """
#     # def __init__(self, dim=512, d_state=16, num_tokens=64, inner_rank=128, mlp_ratio=2.0):
#     def __init__(self, dim=512, d_state=32, num_tokens=64, inner_rank=128, mlp_ratio=2.0):
#         super().__init__()
#         self.dim = dim
#         self.num_tokens = num_tokens
#         self.inner_rank = inner_rank
#         self.d_state = d_state
#
#         # 1. 内部集成 EmbeddingA 和 EmbeddingB
#         # 将 embeddingA 写入 ASSM 内部，作为可学习参数
#         self.embeddingA = nn.Parameter(torch.empty(self.inner_rank, d_state))
#         self.embeddingB = nn.Parameter(torch.empty(num_tokens, inner_rank))
#
#         nn.init.uniform_(self.embeddingA, -1 / inner_rank, 1 / inner_rank)
#         nn.init.uniform_(self.embeddingB, -1 / num_tokens, 1 / num_tokens)
#
#         # 2. Mamba 与特征处理组件
#         hidden_dim = int(self.dim * mlp_ratio)
#         self.in_proj = nn.Linear(self.dim, hidden_dim)
#
#         # 1D 局部上下文增强 (针对 Patch 序列)
#         self.cpe_1d = nn.Sequential(
#             nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
#             nn.SiLU()
#         )
#
#         # 核心扫描算子
#         # from mamba_ssm import SelectiveScan
#         # self.selectiveScan = SelectiveScan(d_model=hidden_dim, d_state=self.d_state, expand=1)
#         self.selectiveScan = selective_scan_fn
#
#         self.out_norm = nn.LayerNorm(hidden_dim)
#         self.out_proj = nn.Linear(hidden_dim, self.dim)
#
#         # 3. 语义路由
#         self.route = nn.Sequential(
#             nn.Linear(self.dim, self.dim // 4),
#             nn.GELU(),
#             nn.Linear(self.dim // 4, self.num_tokens),
#             nn.LogSoftmax(dim=-1)
#         )
#
#     def _semantic_neighbor_internal(self, x, index):
#         """
#         集成你提供的实现逻辑：
#         使用 torch.gather 灵活对齐维度并进行重排
#         """
#         dim = index.dim()  # 通常是 2 (B, N)
#         # 维度对齐逻辑
#         for _ in range(x.dim() - index.dim()):
#             index = index.unsqueeze(-1)
#         index = index.expand(x.shape)
#
#         # 执行重排 (在 dim=1，即 N 维度上进行 gather)
#         return torch.gather(x, dim=dim - 1, index=index)
#
#     def forward(self, x):
#         """
#         Input x: [B, N, 512]
#         """
#         B, N, C = x.shape
#
#         # --- 阶段 1: 语义 Prompt 生成 ---
#         full_embedding = self.embeddingB @ self.embeddingA  # [64, d_state]
#         pred_route = self.route(x)  # [B, N, 64]
#         cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)
#         # 生成动态 Prompt: [B, N, d_state]
#         prompt = torch.matmul(cls_policy, full_embedding)
#
#         # --- 阶段 2: 索引计算 ---
#         # 获取每个实例的组别索引
#         detached_index = torch.argmax(cls_policy.detach(), dim=-1)  # [B, N]
#         # 计算排序索引 (Unfold)
#         _, x_sort_indices = torch.sort(detached_index, dim=-1, stable=True)
#         # 计算逆向索引 (Fold)
#         x_sort_indices_reverse = torch.argsort(x_sort_indices, dim=-1)
#
#         # --- 阶段 3: 特征投影与局部增强 ---
#         x_hid = self.in_proj(x)  # [B, N, hidden]
#         # 1D CPE 增强
#         x_res = x_hid.transpose(1, 2)
#         x_hid = (x_res + self.cpe_1d(x_res)).transpose(1, 2)
#
#         # --- 阶段 4: 语义邻居重排与扫描 ---
#         # 1. 语义重排 (Unfold)
#         semantic_x = self._semantic_neighbor_internal(x_hid, x_sort_indices)
#         semantic_prompt = self._semantic_neighbor_internal(prompt, x_sort_indices)
#
#         # 2. Mamba 核心扫描
#         y = self.selectiveScan(semantic_x, semantic_prompt)
#
#         # 3. 空间还原 (Fold)
#         y_restored = self._semantic_neighbor_internal(y, x_sort_indices_reverse)
#
#         # --- 阶段 5: 输出投影 ---
#         out = self.out_proj(self.out_norm(y_restored))
#         return out

class ASSM(nn.Module):
    """
    高级语义扫描模块 (ASSM) - 修正参数版
    """

    def __init__(self,
                 dim=512,
                 d_state=32,
                 num_tokens=64,
                 inner_rank=128,
                 mlp_ratio=4.0,  # 建议与 Parallel 层保持一致
                 dt_init="random",  # 添加此参数
                 dt_rank="auto",  # 添加此参数
                 **kwargs):  # 添加 kwargs 接收多余参数
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank
        self.d_state = d_state

        # 1. 语义嵌入矩阵
        self.embeddingA = nn.Parameter(torch.empty(self.inner_rank, d_state))
        self.embeddingB = nn.Parameter(torch.empty(num_tokens, inner_rank))
        nn.init.uniform_(self.embeddingA, -1 / inner_rank, 1 / inner_rank)
        nn.init.uniform_(self.embeddingB, -1 / num_tokens, 1 / num_tokens)

        # 2. 特征投影
        hidden_dim = int(self.dim * mlp_ratio)
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(self.dim, hidden_dim)

        # 对齐 dt_rank
        self.dt_rank = math.ceil(self.dim / 16) if dt_rank == "auto" else dt_rank
        self.x_proj = nn.Linear(hidden_dim, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, hidden_dim, bias=True)

        # 3. 核心参数初始化 (使用传入的 dt_init)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(hidden_dim, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(hidden_dim))

        # --- 重点：利用 dt_init 进行动力学对齐 ---
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

        # 4. 算子与组件
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

    # ... forward 和 _semantic_neighbor_internal 保持不变 ...

    def _semantic_neighbor_internal(self, x, index):
        dim = index.dim()
        for _ in range(x.dim() - index.dim()):
            index = index.unsqueeze(-1)
        index = index.expand(x.shape)
        return torch.gather(x, dim=dim - 1, index=index)

    def forward(self, x):
        B, N, C = x.shape

        # --- 阶段 1: 语义引导生成 ---
        # 计算全局语义原型并生成动态引导
        full_embedding = self.embeddingB @ self.embeddingA  # [64, d_state]
        pred_route = self.route(x)
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, N, 64]
        # prompt 将作为后续 B, C 矩阵的语义补充
        prompt = torch.matmul(cls_policy, full_embedding)  # [B, N, d_state]

        # --- 阶段 2: 语义排序与特征重排 ---
        detached_index = torch.argmax(cls_policy.detach(), dim=-1)
        _, x_sort_indices = torch.sort(detached_index, dim=-1, stable=True)
        x_sort_indices_reverse = torch.argsort(x_sort_indices, dim=-1)

        # 投影并局部增强
        x_hid = self.in_proj(x)  # [B, N, hidden]
        x_res = x_hid.transpose(1, 2)
        x_hid = (x_res + self.cpe_1d(x_res)).transpose(1, 2)

        # 执行语义重排 (Unfold)
        # 将语义相近的 Patch 在序列中拉近
        semantic_x = self._semantic_neighbor_internal(x_hid, x_sort_indices)

        # --- 阶段 3: SSM 参数映射 (关键修改) ---
        # 从重排后的序列中生成 dt, B, C
        x_dbl = self.x_proj(semantic_x)  # [B, N, dt_rank + d_state*2]
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dts = self.dt_proj(dts)  # [B, N, hidden]

        # 注入语义 Prompt 的影响力
        # 将 prompt (来自路由) 与 Bs, Cs 相结合，实现语义层面的权重干预
        # 这是分子分型的“全局知识”与“局部实例”的结合点
        Bs = Bs + prompt
        Cs = Cs + prompt

        # 转换维度以适配 selective_scan_fn (B, D, L)
        u = semantic_x.transpose(1, 2).contiguous()
        delta = dts.transpose(1, 2).contiguous()
        A = -torch.exp(self.A_log.float())  # [hidden, d_state]
        B_ssm = Bs.transpose(1, 2).contiguous()  # [B, d_state, N]
        C_ssm = Cs.transpose(1, 2).contiguous()  # [B, d_state, N]

        # --- 阶段 4: 官方内核扫描 ---
        # 直接传入 C_ssm，由内核直接计算输出 y = C * h
        y_semantic = self.selectiveScan(
            u, delta, A, B_ssm, C_ssm,
            D=self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False
        )  # 返回 (B, hidden, N)

        # --- 阶段 5: 还原与输出 ---
        # 将处理后的语义序列还原回原始空间顺序 (Fold)
        y_restored = self._semantic_neighbor_internal(y_semantic.transpose(1, 2), x_sort_indices_reverse)

        out = self.out_proj(self.out_norm(y_restored))
        return out


class GatedMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim * 2)
        self.gate_act = nn.SiLU()  # SiLU 对病理特征的响应比 ReLU 更平滑
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, N, 512]
        x = self.norm(x)
        x_proj = self.proj(x)
        x_main, x_gate = x_proj.chunk(2, dim=-1)

        # 门控逻辑：用 x_gate 的非线性激活去控制 x_main 的信息流
        x_filtered = x_main * self.gate_act(x_gate)

        return self.out_proj(x_filtered)


# --- 核心融合模块：Cross-Gated Fusion (CGF) ---
class CrossGatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_s = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_a = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.proj_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, f_struct, f_seman):
        # f_struct: 结构分支输出 [B, N, 512]
        # f_seman: 语义分支输出 [B, N, 512]

        # 交叉门控逻辑：使用语义特征生成的门控去过滤结构特征，反之亦然
        # 功效：排除“结构正常但语义异常”或“语义匹配但结构无意义”的区域
        g_s = self.gate_s(f_struct)
        g_a = self.gate_a(f_seman)

        # 融合：相互调制后相加
        fused = (f_struct * g_a) + (f_seman * g_s)
        return self.proj_out(self.norm(fused))

class PSSMambaModule(nn.Module):
    """
    优化后的 Parallel Structure-Semantic Mamba Module (PSS-Mamba)
    针对分子分型任务优化：
    1. 引入分支学习缩放 (Learnable Branch Scaling)，自动平衡结构与语义贡献。
    2. 统一特征映射能力 (MLP Ratio 对齐)。
    3. 强化特征流交互，减少分子信号在并行路径中的损失。
    """

    def __init__(self, dim, d_state=32, dt_init="random", mlp_ratio=4.0,
                 num_tokens=64, attn_drop_rate=0.0, use_gated=True, **kwargs):
        super().__init__()
        self.use_gated = use_gated

        # 1. 动态对齐控制参数
        # 对齐 dt_rank 以确保两个分支的时间步长感官一致
        dt_rank = math.ceil(dim / 16)
        mamba_kwargs = {
            "d_state": d_state,
            "dt_init": dt_init,
            "dt_rank": dt_rank,
            "dropout": attn_drop_rate,
        }

        # --- 分支 A: 结构感知分支 (保序扫描) ---
        # 负责捕捉组织架构、细胞排列等拓扑特征
        self.branch_struct = StructureAwareSSM1d(
            d_model=dim,
            d_conv=3,
            expand=mlp_ratio,  # 统一扩展倍率，增强表达容量
            **mamba_kwargs,
            **kwargs
        )

        # --- 分支 B: 语义聚类分支 (重排扫描) ---
        # 负责跨空间聚类稀疏的分子信号，剔除背景噪声
        self.branch_seman = ASSM(
            dim=dim,
            num_tokens=num_tokens,
            inner_rank=128,
            mlp_ratio=mlp_ratio,  # 对齐扩展倍率，确保语义挖掘深度
            **mamba_kwargs,
            **kwargs
        )

        # 2. 分支增益控制 (Learnable Scaling)
        # 针对分子分型，不同病例对结构/语义的依赖不同，引入学习权重平衡初期梯度
        self.gamma_s = nn.Parameter(1e-5 * torch.ones(dim), requires_grad=True)
        self.gamma_a = nn.Parameter(1e-5 * torch.ones(dim), requires_grad=True)

        # 3. 语义增强：Gated-Mixer
        if self.use_gated:
            self.ln_gated = nn.LayerNorm(dim)
            self.gated_mixer = GatedMixer(dim)

            # 4. 交叉门控融合 (核心校准逻辑)
        self.fusion = CrossGatedFusion(dim)

        # 5. 最终投影 (Feature Refinement)
        # 在融合后进行一次线性映射，整合两个专家的观点
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Input: x [B, N, 512]
        """
        # --- 步骤 1: 并行处理 ---
        # 两个分支独立观察原始特征，产出不同维度的见解
        f_s = self.branch_struct(x)
        f_a = self.branch_seman(x)

        # --- 步骤 2: 语义路径增强 ---
        # 仅在语义分支后执行门控筛选，因为语义路径更依赖于“特征蒸馏”
        if self.use_gated:
            f_a = f_a + self.gated_mixer(self.ln_gated(f_a))

        # --- 步骤 3: 学习缩放与融合 ---
        # 应用 LayerScale 逻辑，提高深层网络的训练稳定性
        f_s = f_s * self.gamma_s
        f_a = f_a * self.gamma_a

        # 交叉门控融合：结构与语义特征相互调制
        # 过滤掉“结构虽好但语义无关”或“语义匹配但形态不对”的区域
        fused_x = self.fusion(f_s, f_a)

        # --- 步骤 4: 最终整合 ---
        # 通过线性层对融合后的多维特征进行降维压缩，提取核心 3 分类判别向量
        return self.out_proj(fused_x)


class MorphSharpener(nn.Module):
    """
    针对 512 维病理特征设计的微小信号增强器
    """

    def __init__(self, dim):
        super().__init__()
        # 1. 深度扩张卷积：捕捉更广范围的细胞交互
        self.dilated_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=4, dilation=2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU()
        )
        # 2. 通道选择器：锁定对分型敏感的特征维度
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // 16, 1),
            nn.ReLU(),
            nn.Conv1d(dim // 16, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, N, 512] -> [B, 512, N]
        x = x.transpose(1, 2)

        # 提取高频局部形态细节
        detail = self.dilated_conv(x)
        # 通道加权
        scale = self.ca(x)

        # 锐化：原特征 + 增强的细节
        out = x * scale + detail
        return out.transpose(1, 2)

class WaveletFeatureSharpener(nn.Module):
    """
    针对 512 维特征的小波锐化模块
    利用 Haar 小波思路提取高频分量
    """

    def __init__(self, dim):
        super().__init__()
        # 模拟小波分解的两个分量：低频聚合（Avg）和高频差分（Diff）
        self.low_freq = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.high_freq = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.fuse = nn.Linear(dim * 2, dim)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        # x: [B, N, 512] -> [B, 512, N]
        identity = x
        x_t = x.transpose(1, 2)

        # 1. 提取低频（平滑背景）
        low = self.low_freq(x_t)
        # 2. 提取高频（形态细节/突变）
        # 通过原信号减去低频得到残差高频
        high = x_t - low

        # 3. 动态增强高频细节
        # 在分子分型中，我们给高频分量更大的权重
        combined = torch.cat([low, high * 2.0], dim=1).transpose(1, 2)
        refined = self.fuse(combined)

        # 4. 门控残差连接
        return identity + refined


class ResidualWavelet(nn.Module):
    """
    针对数字病理 MIL 任务优化的 1D 残差小波变换模块
    适配输入维度: [B, N, C] (Batch, Instances, Channels)

    功效：
    1. 特征提纯：通过 1D Haar 小波分离特征序列的低频（结构）和高频（形态突变）信号。
    2. 局部上下文增强：在 N 维度（Patch 序列）上捕捉细微的形态学抖动。
    3. 灵活下采样：可选择是否减少 Patch 数量 (stride=2)。
    """

    def __init__(self, in_channels, out_channels=None, stride=1):
        super(ResidualWavelet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        self.stride = stride

        # 1. 残差路径 (适配 1D 序列)
        self.identity = nn.Sequential(
            nn.Conv1d(in_channels, self.out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(self.out_channels)
        )

        # 2. 1D Haar 小波分解逻辑 (手动实现以适配 [B, C, N])
        # Haar 低通: (x1 + x2)/2, 高通: (x1 - x2)/2
        self.register_buffer('filter_low', torch.tensor([0.5, 0.5]).view(1, 1, 2))
        self.register_buffer('filter_high', torch.tensor([0.5, -0.5]).view(1, 1, 2))

        # 3. 小波特征编码器
        # 分解后会得到 2 个分量 (L, H)，通道数变 2 倍
        self.encode = nn.Sequential(
            nn.Conv1d(in_channels * 2, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.out_channels),
            nn.SiLU()  # SiLU 对病理分子信号更灵敏
        )

        # 4. 针对分子分型的通道注意力 (SE-Block)
        # 用于自动筛选对分类有贡献的小波分量
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.out_channels, self.out_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.out_channels // 8, self.out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Input x: [B, N, C]
        """
        # 转换维度: [B, N, C] -> [B, C, N] 以进行 1D 卷积
        x = x.transpose(1, 2)
        B, C, N = x.shape

        # --- 步骤 1: 1D 离散小波分解 ---
        # 如果 N 是奇数，补齐到偶数
        if N % 2 != 0:
            x = F.pad(x, (0, 1), mode='replicate')

        # 重新排列以进行 Haar 计算
        # 通过 depthwise conv 模拟 1D DWT
        low = F.conv1d(x, self.filter_low.repeat(C, 1, 1), stride=2, groups=C)
        high = F.conv1d(x, self.filter_high.repeat(C, 1, 1), stride=2, groups=C)

        # 拼接低频和高频信息 [B, 2*C, N/2]
        dwt_feat = torch.cat([low, high], dim=1)

        # --- 步骤 2: 特征编码与锐化 ---
        x_dwt = self.encode(dwt_feat)

        # 如果不希望下采样 (stride=1)，则插值回原始长度
        if self.stride == 1:
            x_dwt = F.interpolate(x_dwt, size=N, mode='linear', align_corners=False)

        # 通道注意力筛选分型信号
        x_dwt = x_dwt * self.ca(x_dwt)

        # --- 步骤 3: 残差融合 ---
        # 处理残差路径
        res = self.identity(x[:, :, :N])  # 确保长度对齐

        # 如果 stride=1 且之前补过位，截断
        if self.stride == 1:
            out = x_dwt + res
        else:
            out = x_dwt + res

        # 还原维度: [B, C, N'] -> [B, N', C']
        return out.transpose(1, 2)




class StructureAwareSSM1d_MultiScale(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=32,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 1. 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)

        # 2. 并行多尺度深度卷积分支
        self.conv1d_k3 = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            groups=self.d_inner, kernel_size=3, padding=1
        )
        self.conv1d_k7 = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            groups=self.d_inner, kernel_size=7, padding=3
        )

        # 自适应尺度融合权重
        self.scale_weight = nn.Parameter(torch.ones(2))
        self.act = nn.SiLU()

        # 3. SSM 参数投影
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A, D 矩阵初始化
        # A 保持为 (D_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 4. 核心算子
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        self.selective_scan = selective_scan_fn

        # 5. 结构融合与输出
        self.state_fusion = AdaptiveStateFusion1d(self.d_inner)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, x):
        # x: [B, L, D] (例如 B, 2160, 512)
        B, L, D = x.shape

        # 步骤 1: 进入并行分支并投影
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)  # x_in: [B, L, D_inner]

        # 步骤 2: 多尺度卷积 (在此阶段保持 B, D, L 格式)
        x_conv_in = x_in.transpose(1, 2).contiguous()  # [B, D_inner, L]

        feat_k3 = self.conv1d_k3(x_conv_in)
        feat_k7 = self.conv1d_k7(x_conv_in)

        # 自适应融合
        w = torch.softmax(self.scale_weight, dim=0)
        u_conv = w[0] * feat_k3 + w[1] * feat_k7
        u_conv = self.act(u_conv)  # [B, D_inner, L]，这作为 SSM 的输入 u

        # 步骤 3: 准备 SSM 参数 (转置为 L 维度进行 Linear 投影)
        x_ssm_input = u_conv.transpose(1, 2).contiguous()  # [B, L, D_inner]
        x_dbl = self.x_proj(x_ssm_input)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # 再次投影 dt 并在内核调用前转置
        dt = self.dt_proj(dt).transpose(1, 2).contiguous()  # [B, D_inner, L]
        B_ssm = B_ssm.transpose(1, 2).contiguous()  # [B, d_state, L]
        C_ssm = C_ssm.transpose(1, 2).contiguous()  # [B, d_state, L]

        As = -torch.exp(self.A_log.float())  # [D_inner, d_state]

        # 步骤 4: Selective Scan 内核调用
        # 直接传入 Cs 模式
        y_raw = self.selective_scan(
            u_conv, dt, As, B_ssm, C_ssm,
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False
        )  # y_raw: [B, D_inner, L]

        # 步骤 5: 结构融合精修 (StateFusion)
        y_fused = self.state_fusion(y_raw)  # [B, D_inner, L]

        # 步骤 6: 修复后的残差路径 D 与输出映射
        # 技巧：在 B, D, L 维度下进行 D 残差计算，D.view(-1, 1) 能完美匹配 [D_inner, 1]
        y = y_fused + u_conv * self.D.view(-1, 1)

        # 步骤 7: 还原到 [B, L, D] 空间
        y = y.transpose(1, 2).contiguous()  # [B, L, D_inner]
        y = self.out_norm(y)

        # 结合门控分支 z 并映射回 d_model
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y

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
            current_block_type="StructureAwareSSM1d",
            is_last_layer=False,  # 传入 flag
            **kwargs,
    ):
        super().__init__()

        # CPE (Conditional Positional Encoding) 改为 1D 卷积
        self.cpe1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)
        self.is_last_layer = is_last_layer

        # self.self_attention = StructureAwareSSM1d(
        #     d_model=hidden_dim,
        #     dropout=attn_drop_rate,
        #     d_state=d_state,
        #     dt_init=dt_init,
        #     **kwargs
        # )

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

        # self.fremlp = FreMLP1D(nc=hidden_dim, expand=2)

        # --- 方案 1 增加部分 ---
        # self.use_fourier = use_fourier
        # if self.use_fourier:
        #     self.ln_fourier = norm_layer(hidden_dim)
        #     self.fourier_unit = FourierUnit(hidden_dim)
        #     self.gamma_f = nn.Parameter(1e-5 * torch.ones(hidden_dim), requires_grad=True)
        self.current_block_type = current_block_type
        if self.current_block_type == "StructureAwareSSM1d":
            self.self_attention = StructureAwareSSM1d(
            # # self.self_attention = StructureAwareSSM1d_v2(
            # # self.self_attention = StructureAwareSSM1d_v3(
            # self.self_attention = StructureAwareSSM1d_MultiScale(
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
        # self.aug=MorphSharpener(dim=hidden_dim)
        # self.aug=WaveletFeatureSharpener(dim=hidden_dim)

        # if self.is_last_layer:
        #     self.aug=ResidualWavelet(in_channels=512, stride=1)

            # self.self_attention = ASSM(dim=hidden_dim,
            #     num_tokens=64,    # 分子分型推荐 64，用于精细过滤间质/坏死/免疫细胞
            #     inner_rank=128,   # 保持低秩分解以提取核心病理基底
            #     mlp_ratio=2.0)
            # # 专门为 Gated-Mixer 准备的归一化层
            # self.ln_gated = norm_layer(hidden_dim)
            # # 实例化之前定义的 GatedMixer
            # self.gated_mixer = GatedMixer(hidden_dim)

        # if self.is_last_layer:
        #     self.ln_gated = norm_layer(hidden_dim)
        #     self.gated_mixer = GatedMixer(hidden_dim)


    def forward(self, x: torch.Tensor):

        # if self.is_last_layer:
        #     x= self.aug(x)


        # Input: (B, L, D)

        # CPE 1: 需要 (B, D, L)
        res = x
        x_cpe = x.transpose(1, 2).contiguous()
        x_cpe = self.cpe1(x_cpe).transpose(1, 2)
        x = res + x_cpe

        # SSM Part
        x = x + self.drop_path(self.self_attention(self.ln_1(x)))

        # 3. Gated-Mixer (核心改进：特征筛选层)
        # 建议仅在 self.current_block_type == "ASSM" 时激活

        # if self.current_block_type == "ASSM":
        #     # 这里的 ln_gated 是专门为门控单元准备的归一化
        #     x = x + self.drop_path(self.gated_mixer(self.ln_gated(x)))

        # if self.is_last_layer:
        #     x = x + self.drop_path(self.gated_mixer(self.ln_gated(x)))

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

        # if self.use_fourier:
        #     x = x + self.drop_path(self.gamma_f * self.fourier_unit(self.ln_fourier(x)))

        # MLP Part
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        # x = x + self.drop_path(self.fremlp(self.ln_2(x)))
        return x


class SoftSemanticRefiner(nn.Module):
    """
    软语义细化器：专门适配预提取的 .pt 特征
    不再使用硬小波，而是使用‘自适应维度选择’来提取细微信号
    """

    def __init__(self, dim=512):
        super().__init__()
        # 1. 局部上下文感知（使用深度可分离卷积，不破坏语义，只对齐邻域）
        self.context_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU()
        )
        # 2. 维度筛选器（Channel-wise Attention）
        # 作用：在 512 维中自动寻找对“分子分型”敏感的通道
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // 16, dim, 1),
            nn.Sigmoid()
        )
        # 3. 学习型残差权重
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [B, N, 512]
        res = x
        x = x.transpose(1, 2)  # [B, 512, N]

        # 提取邻域一致性特征
        context = self.context_conv(x)
        # 自动增强关键通道
        gate = self.channel_gate(x)

        refined = context * gate
        # 以极小的权重开始注入，保证模型基础性能
        out = res + self.alpha * refined.transpose(1, 2)
        return out
#
# class SpatialMambaLayer(nn.Module):
#     """
#     Modified Layer to handle (B, L, D).
#     Removed 'downsample' logic related to 2D resizing.
#     """
#
#     def __init__(
#             self,
#             dim,
#             depth,
#             attn_drop=0.,
#             drop_path=0.,
#             norm_layer=nn.LayerNorm,
#             use_checkpoint=False,
#             d_state=16,
#             dt_init="random",
#             mlp_ratio=4.0,
#             current_block_type="StructureAwareSSM1d",
#             is_last_layer=False,  # 新增参数
#             **kwargs,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.use_checkpoint = use_checkpoint
#
#         # self.pre_refiner = SoftSemanticRefiner(dim=512)
#
#         for i in range(depth):
#             self.blocks = nn.ModuleList([
#                 SpatialMambaBlock(
#                     hidden_dim=dim,
#                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                     norm_layer=norm_layer,
#                     attn_drop_rate=attn_drop,
#                     d_state=d_state,
#                     dt_init=dt_init,
#                     mlp_ratio=mlp_ratio,
#                     # 方案 1：只有最后一层的最后一个 Block 开启傅里叶
#                     # use_fourier=(is_last_layer and i == depth - 1),
#                     is_last_layer=is_last_layer,
#                     # is_last_layer=(is_last_layer and i == depth - 2),
#                     current_block_type = current_block_type,
#                     **kwargs
#                 )
#                 for i in range(depth)])
#
#     def forward(self, x):
#
#         # 信号预处理（软性提纯）
#         # x = self.pre_refiner(x)
#
#         # Input x: (B, L, D)
#         for blk in self.blocks:
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 x = blk(x)
#         return x

class SpatialMambaLayer(nn.Module):
    """
    Handles (B, L, D) token sequence.
    Correctly builds 'depth' blocks sequentially.
    """

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        drop_path=0.,              # can be float or list length=depth
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        d_state=16,
        dt_init="random",
        mlp_ratio=4.0,
        current_block_type="StructureAwareSSM1d",
        is_last_layer=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # normalize drop_path to list for each block
        if isinstance(drop_path, (list, tuple)):
            assert len(drop_path) == depth, f"drop_path list length {len(drop_path)} != depth {depth}"
            dpr_list = list(drop_path)
        else:
            dpr_list = [float(drop_path)] * depth

        # ✅ correct: build exactly 'depth' blocks
        self.blocks = nn.ModuleList([
            SpatialMambaBlock(
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
        # x: (B, L, D)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x