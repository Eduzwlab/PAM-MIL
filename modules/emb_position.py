import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
class PPEG(nn.Module):
    def __init__(self, dim=512,k=7,conv_1d=False,bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5,1), 1, (5//2,0), groups=dim,bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3,1), 1, (3//2,0), groups=dim,bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        
        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) 

        if H < 7:
            H,W = 7,7
            zero_pad = H * W - (N+add_length)
            x = torch.cat([x, torch.zeros((B,zero_pad,C),device=x.device)],dim = 1)
            add_length += zero_pad

        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # print(add_length)
        if add_length >0:
            x = x[:,:-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class PEG(nn.Module):
    def __init__(self, dim=512,k=7,bias=True,conv_1d=False):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        x = torch.cat([x, x[:,:add_length,:]],dim = 1)

        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat

        x = x.flatten(2).transpose(1, 2)
        if add_length >0:
            x = x[:,:-add_length]

        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class SINCOS(nn.Module):
    def __init__(self,embed_dim=512):
        super(SINCOS, self).__init__()
        self.embed_dim = embed_dim
        self.pos_embed = self.get_2d_sincos_pos_embed(embed_dim, 8)
    def get_1d_sincos_pos_embed_from_grid(self,embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def get_2d_sincos_pos_embed_from_grid(self,embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    def get_2d_sincos_pos_embed(self,embed_dim, grid_size, cls_token=False):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed

    def forward(self, x):
        #B, N, C = x.shape
        B,H,W,C = x.shape
        # # padding
        # H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        # add_length = H * W - N
        # x = torch.cat([x, x[:,:add_length,:]],dim = 1)

        # pos_embed = torch.zeros(1, H * W + 1, self.embed_dim)
        # pos_embed = self.get_2d_sincos_pos_embed(pos_embed.shape[-1], int(H), cls_token=True)
        #pos_embed = torch.from_numpy(self.pos_embed).float().unsqueeze(0).to(x.device)

        pos_embed = torch.from_numpy(self.pos_embed).float().to(x.device)

        # print(pos_embed.size())
        # print(x.size())
        x = x + pos_embed.unsqueeze(1).unsqueeze(1).repeat(1,H,W,1)
        

        #x = x + pos_embed[:, 1:, :]

        # if add_length >0:
        #     x = x[:,:-add_length]

        return x


def ca_weight(proj_query, proj_key):
    [b, c, h, w] = proj_query.shape
    # print("---------------------------------")
    # print('proj_query.shape:', proj_query.shape)
    # torch.Size([1, 64, 62, 62])
    proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
    # print('proj_query_H.shape:', proj_query_H.shape)
    # torch.Size([62, 62, 64])
    proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
    # print('proj_query_W.shape:', proj_query_W.shape)
    # torch.Size([62, 62, 64])
    proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
    # print('proj_key_H.shape:', proj_key_H.shape)
    # torch.Size([62, 64, 62])
    proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
    # print('proj_key_W.shape:', proj_key_W.shape)
    # torch.Size([62, 64, 62])
    energy_H = torch.bmm(proj_query_H, proj_key_H).view(b, w, h, h).permute(0, 2, 1, 3)
    # print('energy_H.shape:', energy_H.shape)
    # torch.Size([1, 62, 62, 62])
    energy_W = torch.bmm(proj_query_W, proj_key_W).view(b, h, w, w)
    # print('energy_W.shape:', energy_W.shape)
    # torch.Size([1, 62, 62, 62])
    concate = torch.cat([energy_H, energy_W], 3)
    # print('concate.shape:', concate.shape)
    # torch.Size([1, 62, 62, 124])
    return concate
def ca_map(attention, proj_value):
    [b, c, h, w] = proj_value.shape
    proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
    proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
    att_H = attention[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
    att_W = attention[:, :, :, h:h + w].contiguous().view(b * h, w, w)
    out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
    out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)
    out = out_H + out_W
    return out
class CCAttention(nn.Module):
    def __init__(self, in_dim, dim=512,k=7,conv_1d=False,bias=True):
    # def __init__(self, in_dim, n_classes):
        super(CCAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.fc = nn.Linear(in_dim, 3)
        # self.fc1 = nn.Linear(in_dim,3)
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k, 1),
                                                                                                           1, (k // 2, 0),
                                                                                                           groups=dim,
                                                                                                           bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (5, 1), 1,
                                                                                                            (5 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (3, 1), 1,
                                                                                                            (3 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)

        # self.proj3 = nn.Conv2d(dim, dim, 1, 1, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,(1, 1), 1,groups=dim,bias=bias)

    def forward(self, x):
        B, N, C = x.shape
        # print("--x---x----x----")
        # print(x.shape)

        H1 = x.shape[1]
        # print("H shape",H)                          # H shape 6000
        H, W = int(np.ceil(np.sqrt(H1))), int(np.ceil(np.sqrt(H1)))
        add_length = H * W - N
        # if add_length >0:
        x1 = torch.cat([x, x[:, :add_length, :]], dim=1)
        # print("--x1---x1----x1----")
        # print(x1.shape)
        # print("--H---H----H----")
        # print(H)

        if H < 7:
            H, W = 7, 7
            zero_pad = H * W - (N + add_length)
            x1 = torch.cat([x1, torch.zeros((B, zero_pad, C), device=x1.device)], dim=1)
            add_length += zero_pad
        cnn_feat = x1.transpose(1, 2).view(B, C, H, W)
        # cnn_feat = feat_token.transpose(0, 1).view(1, C, H, W)

        # print("-----cnn_feat--------")
        # print(cnn_feat.shape)
        # torch.Size([1, 512, 60, 60])

        # out = self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat) + cnn_feat

        #### 第一次 #############

        proj_query = self.query_conv(cnn_feat)
        proj_key = self.key_conv(cnn_feat)
        proj_value = self.value_conv(cnn_feat)

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        # print('attention.shape:', attention.shape)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + cnn_feat

        # try-11 不添加可训练倍数
        # out = out + cnn_feat

        # out.shape = torch.Size([1, 512, 75, 75])

        # try-5
        cnn_feat1 = self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        out = out + cnn_feat1

        # try-7
        # cnn_feat1 = self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat) + self.proj3(cnn_feat)
        # out = out + cnn_feat1

        # try-8
        # out = out + self.proj2(cnn_feat)

        # try-9
        # cnn_feat1 = self.proj1(cnn_feat) + self.proj2(cnn_feat) + self.proj3(cnn_feat)
        # out = out + cnn_feat1

        # try-6
        # out = self.proj(out) + self.proj1(out) + self.proj2(out) +cnn_feat

        # try-10
        # cnn_feat1 = self.proj1(cnn_feat) + self.proj2(cnn_feat)
        # out = out + cnn_feat1

        #### 重复一次 #######

        proj_query1 = self.query_conv1(out)
        proj_key1 = self.key_conv1(out)
        proj_value1 = self.value_conv1(out)

        energy1 = ca_weight(proj_query1, proj_key1)
        attention1 = F.softmax(energy1, 1)
        # print("***********************************")
        # print('attention.shape:', attention.shape)
        # attention.shape: torch.Size([1, 62, 62, 124])

        out1 = ca_map(attention1, proj_value1)
        # print('out1.shape:', out1.shape)
        # torch.Size([1, 124, 62, 62])
        out1 = self.gamma1 * out1 + out


        # out1 = out1 + out

        # x = out1.flatten(2).transpose(1, 2)
        # x.shape = torch.Size([1, 5625, 512])

        # try-1
        # out1 = self.proj(out) + cnn_feat + self.proj1(out) + self.proj2(out)
        # try-2
        # out1 = self.proj(cnn_feat) + out + self.proj1(cnn_feat) + self.proj2(cnn_feat)

        out2 = out1.flatten(2).transpose(1, 2)

        if add_length > 0:
            out2 = out2[:, :-add_length,:]
        # aa = self.fc1(torch.squeeze(out2))
        # A_A = F.softmax(aa, dim=1)
        # atte = torch.unsqueeze(A_A, 0)
        # final = torch.mul(x, atte)
        return out2

class CCAttention_v(nn.Module):
    def __init__(self, in_dim, dim=512,k=7,conv_1d=False,bias=True):
    # def __init__(self, in_dim, n_classes):
        super(CCAttention_v, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.fc = nn.Linear(in_dim, 3)
        self.fc1 = nn.Linear(in_dim,1)
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k, 1),
                                                                                                           1, (k // 2, 0),
                                                                                                           groups=dim,
                                                                                                           bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (5, 1), 1,
                                                                                                            (5 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (3, 1), 1,
                                                                                                            (3 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)

        self.proj3 = nn.Conv2d(dim, dim, 1, 1, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                        (1, 1), 1,
                                                                                                        groups=dim,
                                                                                                        bias=bias)
        self.ffc =[nn.Linear(1024, 512), nn.ReLU()]

        # self.poslayer = PPEG(1024)

    def forward(self, x):
        B, N, C = x.shape
        # print("--x---x----x----")
        # print(x.shape)

        H1 = x.shape[1]

        # print("H shape",H)                          # H shape 6000
        H, W = int(np.ceil(np.sqrt(H1))), int(np.ceil(np.sqrt(H1)))
        add_length = H * W - N
        # if add_length >0:
        x1 = torch.cat([x, x[:, :add_length, :]], dim=1)

        # x1 = self.ffc(x1)
        # print("--x1---x1----x1----")
        # print(x1.shape)
        # print("--H---H----H----")
        # print(H)

        if H < 7:
            H, W = 7, 7
            zero_pad = H * W - (N + add_length)
            x1 = torch.cat([x1, torch.zeros((B, zero_pad, C), device=x1.device)], dim=1)
            add_length += zero_pad
        x1 = x1[:, :, 0:]
        cnn_feat = x1.transpose(1, 2).view(B, C, H, W)
        # cnn_feat = feat_token.transpose(0, 1).view(1, C, H, W)

        # print("-----cnn_feat--------")
        # print(cnn_feat.shape)
        # torch.Size([1, 512, 60, 60])

        # out = self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat) + cnn_feat

        #### 第一次 #############

        proj_query = self.query_conv(cnn_feat)
        proj_key = self.key_conv(cnn_feat)
        proj_value = self.value_conv(cnn_feat)

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        # print('attention.shape:', attention.shape)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + cnn_feat

        # try-11 不添加可训练倍数
        # out = out + cnn_feat

        # out.shape = torch.Size([1, 512, 75, 75])

        # try-5
        # cnn_feat1 = self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        # out = out + cnn_feat1

        # try-7
        # cnn_feat1 = self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat) + self.proj3(cnn_feat)
        # out = out + cnn_feat1

        # try-8
        # out = out + self.proj2(cnn_feat)

        # try-9
        # cnn_feat1 = self.proj1(cnn_feat) + self.proj2(cnn_feat) + self.proj3(cnn_feat)
        # out = out + cnn_feat1

        # try-6
        # out = self.proj(out) + self.proj1(out) + self.proj2(out) +cnn_feat

        # try-10
        # cnn_feat1 = self.proj1(cnn_feat) + self.proj2(cnn_feat)
        # out = out + cnn_feat1

        #### 重复一次 #######

        proj_query1 = self.query_conv1(out)
        proj_key1 = self.key_conv1(out)
        proj_value1 = self.value_conv1(out)

        energy1 = ca_weight(proj_query1, proj_key1)
        attention1 = F.softmax(energy1, 1)
        # print('attention.shape:', attention.shape)
        out1 = ca_map(attention1, proj_value1)
        out1 = self.gamma1 * out1 + out


        # out1 = out1 + out

        # x = out1.flatten(2).transpose(1, 2)
        # x.shape = torch.Size([1, 5625, 512])

        # try-1
        # out1 = self.proj(out) + cnn_feat + self.proj1(out) + self.proj2(out)
        # try-2
        # out1 = self.proj(cnn_feat) + out + self.proj1(cnn_feat) + self.proj2(cnn_feat)

        out2 = out1.flatten(2).transpose(1, 2)

        if add_length > 0:
            out2 = out2[:, :-add_length,:]
        # print("------------out2-----")
        # print(out2.shape)
        # x.shape = torch.Size([1, 5625, 512])


        aa = self.fc1(torch.squeeze(out2))
        A_A = F.softmax(aa, dim=1)
        atte = torch.unsqueeze(A_A, 0)
        # print("------------atte-----")
        # print(atte.shape)
        # print("------------xxx-----")
        # print(x.shape)
        final = torch.mul(atte, x)
        return final,A_A,x



class CCAttention_try2(nn.Module):
    def __init__(self, in_dim, dim=512,k=7,conv_1d=False,bias=True,act_fn=nn.GELU, gate_fn=nn.Sigmoid):
    # def __init__(self, in_dim, n_classes):
        super(CCAttention_try2, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.fc1 = nn.Linear(in_dim,1)
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k, 1),
                                                                                                           1, (k // 2, 0),
                                                                                                           groups=dim,
                                                                                                           bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (5, 1), 1,
                                                                                                            (5 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                            (3, 1), 1,
                                                                                                            (3 // 2, 0),
                                                                                                            groups=dim,
                                                                                                            bias=bias)

        self.proj3 = nn.Conv2d(dim, dim, 1, 1, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim,
                                                                                                        (1, 1), 1,
                                                                                                        groups=dim,
                                                                                                        bias=bias)

        self.norm = nn.LayerNorm(dim)

        reduce_channels = int(in_dim * 0.125)
        self.global_reduce = nn.Linear(in_dim, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_dim)
        self.gate_fn = gate_fn()

        self.local_reduce = nn.Linear(in_dim, reduce_channels)
        self.spatial_select = nn.Linear(reduce_channels * 2, 1)


    def forward(self, x):
        B, N, C = x.shape

        H1 = x.shape[1]
        # print("H shape",H)                          # H shape 6000
        H, W = int(np.ceil(np.sqrt(H1))), int(np.ceil(np.sqrt(H1)))
        add_length = H * W - N
        # if add_length >0:
        x1 = torch.cat([x, x[:, :add_length, :]], dim=1)

        if H < 7:
            H, W = 7, 7
            zero_pad = H * W - (N + add_length)
            x1 = torch.cat([x, torch.zeros((B, zero_pad, C), device=x.device)], dim=1)
            add_length += zero_pad
        cnn_feat = x1.transpose(1, 2).view(B, C, H, W)
        # cnn_feat = feat_token.transpose(0, 1).view(1, C, H, W)

        # print("-----cnn_feat--------")
        # print(cnn_feat.shape)
        # torch.Size([1, 512, 60, 60])

        # out1 = self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat) + cnn_feat

        #### 第一次 #############

        proj_query = self.query_conv(cnn_feat)
        proj_key = self.key_conv(cnn_feat)
        proj_value = self.value_conv(cnn_feat)

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        # print('attention.shape:', attention.shape)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + cnn_feat

        # try-11 不添加可训练倍数
        # out = out + cnn_feat

        # out.shape = torch.Size([1, 512, 75, 75])

        # try-5
        # cnn_feat1 = self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        # out = out + cnn_feat1

        # try-7
        # cnn_feat1 = self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat) + self.proj3(cnn_feat)
        # out = out + cnn_feat1

        # try-8
        # out = out + self.proj2(cnn_feat)

        # try-9
        # cnn_feat1 = self.proj1(cnn_feat) + self.proj2(cnn_feat) + self.proj3(cnn_feat)
        # out = out + cnn_feat1

        # try-6
        # out = self.proj(out) + self.proj1(out) + self.proj2(out) +cnn_feat

        # try-10
        # cnn_feat1 = self.proj1(cnn_feat) + self.proj2(cnn_feat)
        # out = out + cnn_feat1

        #### 重复一次 #######

        proj_query1 = self.query_conv1(out)
        proj_key1 = self.key_conv1(out)
        proj_value1 = self.value_conv1(out)

        energy1 = ca_weight(proj_query1, proj_key1)
        attention1 = F.softmax(energy1, 1)
        # print('attention.shape:', attention.shape)
        out1 = ca_map(attention1, proj_value1)
        out1 = self.gamma1 * out1 + out


        # out1 = out1 + out

        # x = out1.flatten(2).transpose(1, 2)
        # x.shape = torch.Size([1, 5625, 512])

        # try-1
        # out1 = self.proj(out) + cnn_feat + self.proj1(out) + self.proj2(out)
        # try-2
        # out1 = self.proj(cnn_feat) + out + self.proj1(cnn_feat) + self.proj2(cnn_feat)

        out2 = out1.flatten(2).transpose(1, 2)

        if add_length > 0:
            out2 = out2[:, :-add_length,:]

        # print("=========")
        # print(out2.shape)
        # torch.Size([1, 5496, 512])


       ### 通道注意力 ###########
        # out2 = self.norm(out2)
        # x_global = out2.mean(1, keepdim=True)
        # x_global = self.act_fn(self.global_reduce(x_global))
        # c_attn = self.channel_select(x_global)
        # c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        # x_copy = x.clone()
        # return x_copy * c_attn

        ### 通道+空间注意力  ###
        # out2 = self.norm(out2)
        # x_global = out2.mean(1, keepdim=True)
        # x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))
        # c_attn = self.channel_select(x_global)
        # c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        # s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        # s_attn = self.gate_fn(s_attn)  # [B, N, 1]
        # attn = c_attn * s_attn
        # x_copy = x.clone()
        # return x_copy * attn


        ### 没有cc模块 + channel  #########
        # out2 = self.norm(out2)
        # x_global = out2.mean(1, keepdim=True)
        # x_global = self.act_fn(self.global_reduce(x_global))
        # c_attn = self.channel_select(x_global)
        # c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        # x_copy = x.clone()
        # return x_copy * c_attn

        ### 没有cc模块 + 通道+空间注意力  ###
        out2 = self.norm(out2)
        x_global = out2.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(out2))
        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, out2.shape[1], -1)], dim=-1))
        s_attn = self.gate_fn(s_attn)  # [B, N, 1]
        # print("-------s_attn----------")
        # print(s_attn.shape)
        # torch.Size([1, 5496, 1])

        # print("------c_attn-----------")
        # print(c_attn.shape)
        # torch.Size([1, 1, 512])

        attn = c_attn * s_attn
        # print("------attn-----------")
        # print(attn.shape)
        # torch.Size([1, 5496, 512])

        x_copy = x.clone()
        final = x_copy * attn
        # print("-----------------")
        # print(final.shape)
        # torch.Size([1, 5496, 512])

        return final


