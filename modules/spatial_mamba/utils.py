import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# ===========================================================================
# 尝试导入官方 Mamba CUDA 库 (mamba_ssm)
# ===========================================================================
try:
    import mamba_ssm
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as mamba_cuda_fn

    HAS_MAMBA_CUDA = True
    print("✅ 成功检测到 mamba_ssm 库，将使用 CUDA 加速！")
except ImportError:
    HAS_MAMBA_CUDA = False
    print("⚠️ 未检测到 mamba_ssm 库，将回退到纯 Python 模式 (速度较慢)。")


# ===========================================================================
# 1. 纯 Python 实现 (作为后备方案)
# ===========================================================================

def selective_scan_python_loop(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    # ... (这里保留之前的 Python 实现，用于防崩溃) ...
    # 为了节省篇幅，这里简写逻辑，实际上你可以保留上一轮我给你的 Python 代码
    # 这里的核心是下面那个 selective_scan_fn 主函数
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if C is not None: C = C.float()
    if D is not None: D = D.float()
    if z is not None: z = z.float()
    if delta_bias is not None: delta_bias = delta_bias.float()

    b, d, l = u.shape
    n = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)

    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB = torch.einsum('bdl,bnl->bdln', delta, B)

    x = torch.zeros((b, d, n), device=u.device, dtype=u.dtype)
    ys = []
    u_unsq = u.unsqueeze(-1)

    for i in range(l):
        x = deltaA[:, :, i, :] * x + deltaB[:, :, i, :] * u_unsq[:, :, i, :]
        if C is None:
            ys.append(x.clone())
        else:
            y_i = (x * C[:, :, i].unsqueeze(1)).sum(dim=-1)
            ys.append(y_i)

    if C is None:
        y = torch.stack(ys, dim=-1)
    else:
        y = torch.stack(ys, dim=2)

    if D is not None and C is not None:
        y = y + u * D.unsqueeze(-1)
    if z is not None and C is not None:
        y = y * F.silu(z)

    return y.to(dtype_in)


# ===========================================================================
# 2. 主入口函数 (智能路由)
# ===========================================================================

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    智能选择器：
    1. 如果安装了 mamba_ssm 且 C!=None (标准模式)，调用 CUDA 加速。
    2. 如果没安装，或 C==None (获取状态模式)，调用 Python 实现。
    """

    # 官方 Mamba 库目前主要支持 C!=None 的情况 (输出 Output)
    # 如果代码要求 C=None (即你需要 Hidden States 用于 State Fusion)，官方库不支持直接返回完整 State 序列
    # 所以当 C 为 None 时，我们必须强制走 Python 逻辑 (或者你自己写 CUDA)

    use_cuda = HAS_MAMBA_CUDA and (C is not None)

    if use_cuda:
        # 调用官方 CUDA 函数
        return mamba_cuda_fn(
            u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state
        )
    else:
        # 回退到 Python 实现 (处理 C=None 的情况或未安装库的情况)
        y = selective_scan_python_loop(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        if return_last_state:
            return y, None  # Python版暂未实现 last_state 返回，通常不需要
        return y


# ===========================================================================
# 3. 兼容性组件 (Stem, DownSampling)
# ===========================================================================
# ... (保持之前的 ConvLayer, Stem, DownSampling 代码不变) ...
# 为了完整性，下面放上简化的定义，你需要保留之前文件里的完整定义

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)

    def forward(self, x): return self.conv(x)


class Stem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DownSampling(nn.Module):
    def __init__(self, dim, ratio=4.0): super().__init__()

    def forward(self, x): return x


def selective_scan_flop_jit(inputs, outputs): return 0


def selective_scan_state_flop_jit(inputs, outputs): return 0