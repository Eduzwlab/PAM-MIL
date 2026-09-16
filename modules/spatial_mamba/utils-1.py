import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from timm.models.layers import to_2tuple


# ===========================================================================
# 1. 纯 Python 核心循环 (移除了 @torch.jit.script 以修复 Checkpoint 崩溃)
# ===========================================================================

def selective_scan_loop_no_C(
        l: int,
        deltaA: torch.Tensor,
        deltaB: torch.Tensor,
        u_unsq: torch.Tensor,
        x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    专门处理 C 为 None 的情况（返回 Hidden States）
    纯 Python 实现，兼容 Gradient Checkpointing
    """
    ys = []

    for i in range(l):
        # 提取当前步的参数
        dA_i = deltaA[:, :, i, :]
        dB_i = deltaB[:, :, i, :]
        u_i = u_unsq[:, :, i, :]

        # 状态更新: x[t] = A_bar[t] * x[t-1] + B_bar[t] * u[t]
        x = dA_i * x + dB_i * u_i

        # 保存状态
        ys.append(x.clone())

    # 堆叠结果: (B, D, N, L)
    y = torch.stack(ys, dim=-1)
    return y, x


def selective_scan_loop_with_C(
        l: int,
        deltaA: torch.Tensor,
        deltaB: torch.Tensor,
        u_unsq: torch.Tensor,
        x: torch.Tensor,
        C: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    专门处理 C 不为 None 的情况（返回 Output）
    纯 Python 实现，兼容 Gradient Checkpointing
    """
    ys = []

    for i in range(l):
        dA_i = deltaA[:, :, i, :]
        dB_i = deltaB[:, :, i, :]
        u_i = u_unsq[:, :, i, :]

        # 状态更新
        x = dA_i * x + dB_i * u_i

        # 输出计算: y[t] = (x[t] * C[t]).sum(dim=-1)
        C_i = C[:, :, i]
        # (B, D, N) * (B, N) -> 需要广播 C 为 (B, 1, N)
        # 结果为 (B, D, N) -> sum -> (B, D)
        y_i = (x * C_i.unsqueeze(1)).sum(dim=-1)

        ys.append(y_i)

    # 堆叠结果: (B, D, L)
    y = torch.stack(ys, dim=2)
    return y, x


# ===========================================================================
# 2. 主入口函数
# ===========================================================================

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    # 确保输入是 float32 (SSM 对精度敏感)
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if C is not None: C = C.float()
    if D is not None: D = D.float()
    if z is not None: z = z.float()
    if delta_bias is not None: delta_bias = delta_bias.float()

    b, d, l = u.shape
    n = A.shape[1]

    # 1. 处理 delta
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)

    # 2. 离散化 (向量化计算)
    # deltaA = exp(delta * A)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    # deltaB = delta * B
    deltaB = torch.einsum('bdl,bnl->bdln', delta, B)

    # 3. 准备循环变量
    x = torch.zeros((b, d, n), device=u.device, dtype=u.dtype)
    u_unsq = u.unsqueeze(-1)

    # 4. 根据是否有 C 调用不同的函数 (移除了 JIT)
    if C is None:
        y, x_last = selective_scan_loop_no_C(l, deltaA, deltaB, u_unsq, x)
    else:
        y, x_last = selective_scan_loop_with_C(l, deltaA, deltaB, u_unsq, x, C)

    # 5. 后处理 (Residual & Gate)
    if D is not None:
        if C is not None:
            y = y + u * D.unsqueeze(-1)

    if z is not None:
        if C is not None:
            y = y * F.silu(z)

    y = y.to(dtype_in)

    if return_last_state:
        return y, x_last
    return y


# ===========================================================================
# 3. 兼容性函数 (FLOPs, Stem, DownSampling)
# ===========================================================================

def selective_scan_state_flop_jit(inputs, outputs):
    u = inputs[0]
    B_size, D, L = u.shape
    N = inputs[2].shape[1]
    return B_size * D * L * N * 6


def selective_scan_flop_jit(inputs, outputs):
    return selective_scan_state_flop_jit(inputs, outputs)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None: x = self.dropout(x)
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.act: x = self.act(x)
        return x


class Stem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DownSampling(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x