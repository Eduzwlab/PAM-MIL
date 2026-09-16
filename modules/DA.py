import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


"""

这个 DeformableAttention 模块是一个空间注意力机制（Spatial Attention）的变体。它结合了类似 CBAM（卷积块注意力模块）的特征压缩思想，并引入了下采样（增加感受野）和可选的畸变/调制模式（Distortion Mode）。
其核心目标是：确定特征图中“哪些空间位置”更重要，并生成一个权重掩码（Mask）乘以原始特征。


A. 为什么要下采样再上采样？
该模块在计算注意力掩码时，先用 stride=2 的卷积进行下采样，随后用 Upsample 恢复。这种 Encoder-Decoder（编码器-解码器） 结构有两个好处：
增大感受野：在较小的特征图上做 3x3 卷积，相当于在原图上考虑了更大的区域。
过滤噪声：下采样可以看作是一种特征抽象，有助于提取关键的空间结构。
B. 畸变模式 (Distortion Mode) 的作用
当 distortionmode=True 时，代码并没有简单地拼接 Mean 和 Max，而是执行了：
d_avg_out * max_out 和 d_max_out * avg_out。
这是一种交叉相互增强（Cross-Modulation）：
利用平均池化的特征去生成一个权重，来修正/加权最大池化的特征。
这种做法通常是为了让模型学习到更鲁棒的空间特征，抑制由于极值（Max）或背景噪声（Avg）带来的干扰。
C. 梯度钩子 (Backward Hooks) 的玄机
代码中定义了 _set_lra 和 _set_lrm 两个静态方法，并注册为 backward_hook：
grad * 0.4 和 grad * 0.1：这意味着在反向传播时，这两个卷积层的梯度会被大幅度削减。
目的：这实际上是手动降低了这两层的“等效学习率”。作者可能认为这两层（负责学习调制系数）应该缓慢、稳健地更新，防止注意力权重在训练初期发生剧烈抖动，从而导致模型不收敛。

"""

class DeformableAttention(nn.Module):
    def __init__(self, stride=1, distortionmode=False):
        super(DeformableAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.distortionmode = distortionmode
        self.upsample = nn.Upsample(scale_factor=2)
        self.downavg = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downmax = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        if distortionmode:  # 是否调制
            self.d_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv.weight, 0)
            self.d_conv.register_full_backward_hook(self._set_lra)  # 在指定网络层执行完backward()之后调用钩子函数

            self.d_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.d_conv1.weight, 0)
            self.d_conv1.register_full_backward_hook(self._set_lrm)

    @staticmethod
    def _set_lra(module, grad_input, grad_output):  # 设置学习率的大小
        grad_input = [g * 0.4 if g is not None else None for g in grad_input]
        grad_output = [g * 0.4 if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input
        # return grad_output

    @staticmethod
    def _set_lrm(module, grad_input, grad_output):
        grad_input = [g * 0.1 if g is not None else None for g in grad_input]
        grad_output = [g * 0.1 if g is not None else None for g in grad_output]
        grad_input = tuple(grad_input)
        grad_output = tuple(grad_output)
        return grad_input
        # return grad_output

    def forward(self, x):

        B, L, C = x.shape
        H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
        x =x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # 输入 x: [B, C, H, W]

        # 1. 沿通道维度进行平均池化和最大池化，压缩通道信息
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 输出: [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 输出: [B, 1, H, W]

        # 2. 空间下采样：通过 stride=2 的卷积减少分辨率，增大感受野
        avg_out = self.downavg(avg_out)  # 输出: [B, 1, H/2, W/2]
        max_out = self.downmax(max_out)  # 输出: [B, 1, H/2, W/2]

        # 3. 拼接特征
        out = torch.cat([max_out, avg_out], dim=1)  # 输出: [B, 2, H/2, W/2]


        # out = torch.cat([avg_out, max_out], dim=1)
        # out = self.conv(out)

        # 4. 如果开启了调制模式 (Distortion Mode)
        if self.distortionmode:
            # 通过学习到的 d_conv 得到一个调制系数 (0~1)
            # d_avg_out 是由 avg_out 学习来的系数，用来给 max_out 加权
            d_avg_out = torch.sigmoid(self.d_conv(avg_out))  # 输出: [B, 1, H/2, W/2]
            d_max_out = torch.sigmoid(self.d_conv1(max_out))  # 输出: [B, 1, H/2, W/2]

            # 交叉调制并重新拼接
            out = torch.cat([d_avg_out * max_out, d_max_out * avg_out], dim=1)
            # 输出: [B, 2, H/2, W/2]

            # out = d * out  # 为偏移添加调制标量
            # 5. 卷积融合：将 2 个通道合并回 1 个通道
            out = self.conv(out)  # 输出: [B, 1, H/2, W/2]

            # 6. 上采样 + 激活：恢复到原始分辨率并生成 0-1 之间的权重掩码
            # self.upsample(out) -> [B, 1, H, W]
            # mask = self.sigmoid(self.upsample(out))  # 输出: [B, 1, H, W]

            # 【核心修改点】：不再使用 self.upsample(out)
            # 而是根据输入的 H, W 动态调整 mask 的大小
            mask = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            mask = self.sigmoid(mask)


            # 7. 应用注意力：将掩码乘回原特征图
            att_out = x * mask  # 输出: [B, C, H, W]

            att_out = F.relu(att_out)  # 输出: [B, C, H, W]
            att_out = att_out.flatten(2).permute(0, 2, 1)  # 输出: [B, C, H, W]

            return att_out