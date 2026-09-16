import torch
import torch.nn as nn
from einops import repeat
from .nystrom_attention import NystromAttention
from modules.emb_position import *

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,head=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = head,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x, need_attn=True):
        if need_attn:
            z,attn = self.attn(self.norm(x),return_attn=need_attn)
            x = x+z
            return x,attn
        else:
            x = x + self.attn(self.norm(x))
            return x


class ABMixAttention(nn.Module):
    def __init__(self, dim=512, num_heads=4, use_cls_token=True):
        super(ABMixAttention, self).__init__()
        self.use_cls_token = use_cls_token
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x, *args, **kwargs):
        # x: [1, N, 512]
        x = x.contiguous()  # 🔧 防止 AsStrided 警告
        cls_token = None

        if self.use_cls_token and x.shape[1] > 1:
            cls_token = x[:, 0:1, :].clone()  # 🔧 clone 避免 inplace
            tokens = x[:, 1:, :].clone()      # 🔧 clone 避免 inplace
        else:
            tokens = x.clone()

        # Multi-head Attention
        attn_out, _ = self.mha(tokens, tokens, tokens)

        # 注意力融合（out-of-place）
        tokens_mixed = torch.add(tokens, attn_out)

        if cls_token is not None:
            out = torch.cat([cls_token, tokens_mixed], dim=1)
        else:
            out = tokens_mixed

        return out


class ABMaskAttention(nn.Module):
    def __init__(self, dim=512, hidden_dim=128, use_cls_token=True):
        super(ABMaskAttention, self).__init__()
        self.use_cls_token = use_cls_token
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, *args, **kwargs):
        x = x.contiguous()
        cls_token = None

        if self.use_cls_token and x.shape[1] > 1:
            cls_token = x[:, 0:1, :].clone()
            tokens = x[:, 1:, :].clone()
        else:
            tokens = x.clone()

        u = torch.tanh(self.fc1(tokens))       # [B, T, hidden_dim]
        scores = self.fc2(u)                   # [B, T, 1]
        weights = torch.sigmoid(scores)        # [B, T, 1]

        # out-of-place attention masking
        tokens_masked = torch.mul(tokens, weights)  # [B, T, dim]

        if cls_token is not None:
            out = torch.cat([cls_token, tokens_masked], dim=1)
        else:
            out = tokens_masked

        return out


class ABMixMaskAttention(nn.Module):
    def __init__(self, dim=512, num_heads=4, hidden_dim=128, use_cls_token=True):
        super(ABMixMaskAttention, self).__init__()
        self.use_cls_token = use_cls_token
        self.mix_module = ABMixAttention(dim=dim, num_heads=num_heads, use_cls_token=use_cls_token)
        self.mask_module = ABMaskAttention(dim=dim, hidden_dim=hidden_dim, use_cls_token=use_cls_token)

    def forward(self, x, *args, **kwargs):
        x = x.contiguous()
        x_mixed = self.mix_module(x, *args, **kwargs)      # no inplace
        x_masked = self.mask_module(x_mixed, *args, **kwargs)
        return x_masked


class SAttention(nn.Module):

    def __init__(self,mlp_dim=512,pos_pos=0,pos='ppeg',peg_k=7,head=8):
        super(SAttention, self).__init__()
        self.norm = nn.LayerNorm(mlp_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))

        self.layer1 = TransLayer(dim=mlp_dim,head=head)
        self.layer2 = TransLayer(dim=mlp_dim,head=head)

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim,k=peg_k)
        elif pos == 'ccpp':
            self.pos_embedding = CCAttention(in_dim=512, dim=mlp_dim, k=peg_k)
            # self.pos_embedding = CCAttention_v(in_dim=512, dim=mlp_dim, k=peg_k)
        elif pos == 'ABMix':
            self.pos_embedding = ABMixAttention(dim=512)
        elif pos == 'ABMask':
            self.pos_embedding = ABMaskAttention(dim=512)
        elif pos == 'ABMixMask':
            self.pos_embedding = ABMixMaskAttention(dim=512)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(512,k=peg_k)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

    # Modified by MAE@Meta
    def masking(self, x, ids_shuffle=None,len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        assert ids_shuffle is not None

        _,ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False,mask_enable=False):


        batch, num_patches, C = x.shape

        attn = []

        if self.pos_pos == -2:
            x = self.pos_embedding(x)
        
        # masking
        if mask_enable and mask_ids is not None:
            x, _, _ = self.masking(x,mask_ids,len_keep)

        # cls_token
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = batch)
        x = torch.cat((cls_tokens, x), dim=1)
        # print("888888888888888888888888888")
        # print(x.shape)
        # print(x)
        # torch.Size([1, 7325, 512])

        if self.pos_pos == -1:
            x = self.pos_embedding(x)

        # translayer1
        if return_attn:
            x,_attn = self.layer1(x,True)
            attn.append(_attn.clone())
        else:
            x = self.layer1(x)

        # add pos embedding
        if self.pos_pos == 0:
            if isinstance(x, tuple):
                x = x[0]
            x[:,1:,:] = self.pos_embedding(x[:,1:,:])

        # translayer2
        if return_attn:
            x,_attn = self.layer2(x,True)
            attn.append(_attn.clone())
        else:
            x = self.layer2(x)

        # print("999999999999999999999999")
        # print(x.shape)
        # print(x)
        # torch.Size([1, 7325, 512])

        #---->cls_token
        if isinstance(x, tuple):
            x = x[0]

        x = self.norm(x)

        logits = x[:,0,:]
 
        if return_attn:
            _a = attn
            return logits ,_a
        else:
            return logits


class SAttention_v(nn.Module):

    def __init__(self, mlp_dim=512, pos_pos=0, pos='ccpp', peg_k=7, head=8):
        super(SAttention_v, self).__init__()
        self.norm = nn.LayerNorm(mlp_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))

        self.layer1 = TransLayer(dim=mlp_dim, head=head)
        self.layer2 = TransLayer(dim=mlp_dim, head=head)

        self.pos_embed = PPEG(dim=mlp_dim, k=peg_k)

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim, k=peg_k)
        elif pos == 'ccpp':
            # self.pos_embedding = CCAttention(in_dim=512, dim=mlp_dim, k=peg_k)
            # self.pos_embedding = CCAttention_try2(in_dim=512, dim=mlp_dim, k=peg_k)
            self.pos_embedding = CCAttention_v(in_dim=512, dim=mlp_dim, k=peg_k)

        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(512, k=peg_k)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

    # Modified by MAE@Meta
    def masking(self, x, ids_shuffle=None, len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        assert ids_shuffle is not None

        _, ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False, mask_enable=False):
        batch, num_patches, C = x.shape

        attn = []
        #
        # if self.pos_pos == -2:
        #     x = self.pos_embedding(x)

        # masking
        if mask_enable and mask_ids is not None:
            x, _, _ = self.masking(x, mask_ids, len_keep)

        # cls_token
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=batch)
        x = torch.cat((cls_tokens, x), dim=1)

        # if self.pos_pos == -1:
        #     x = self.pos_embedding(x)

        # translayer1
        if return_attn:
            x, _attn = self.layer1(x, True)
            attn.append(_attn.clone())
            # print("++++++++++_attn 1111111111_attn _attn+++++++++=")
            # print(_attn.shape)
        else:
            x = self.layer1(x)

        # add pos embedding
        if self.pos_pos == 0:
            x[:, 1:, :] = self.pos_embed(x[:, 1:, :])
            x, AA, orignial = self.pos_embedding(x)

            # AA =AA.transpose(0, 1)
            # x = torch.squeeze(orignial)
            # logits = torch.mm(AA,x)



            # logits = self.pos_embedding(x)


        # translayer2
        if return_attn:
            x, _attn = self.layer2(x, True)
            attn.append(_attn.clone())

            # print("++++++++++_attn 2222222222_attn _attn+++++++++=")
            # print(_attn.shape)
            # torch.Size([1, 8, 3891])

            # _attn_visual = _attn.mean(axis=1)
            # _attn_visual = _attn.max(axis=1)[0]
            # _attn_visual = _attn.min(axis=1)[0]

            # print("++++++++++_attn 33333333333333_attn _attn+++++++++=")
            # print(_attn_visual.shape)
            # torch.Size([1, 3891])

            global_avg_pool = nn.AdaptiveAvgPool2d((1,_attn.shape[2]))
            _attn_visual = global_avg_pool(_attn)
            _attn_visual = _attn_visual.squeeze(0)
            # print("++++++++++_attn 444444444444444444443_attn _attn+++++++++=")
            # print(_attn_visual.shape)





        else:
            x = self.layer2(x)

        # ---->cls_token
        x = self.norm(x)
        # print("++++++++++x x x x x x +++++++++=")
        # print(x.shape)
        x_raw = x[:, 1:, :]
        x_raw = x_raw.mean(dim=-1)
        # x_raw = x_raw.squeeze(0).transpose(0, 1)

        # print("-x_raw-x_raw-x_raw----x_raw--x_raw-----")
        # print(x_raw.shape)



        logits = x[:, 0, :]



        if return_attn:
            # _a = attn
            _a = _attn_visual
            return logits, _a
        else:
            return logits
    