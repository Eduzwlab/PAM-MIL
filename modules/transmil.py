import torch
import torch.nn as nn
import numpy as np
from .nystrom_attention import NystromAttention
import torch.nn.functional as F

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
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

        H1 = x.shape[1]
        # print("H shape",H)                          # H shape 6000
        H, W = int(np.ceil(np.sqrt(H1))), int(np.ceil(np.sqrt(H1)))
        add_length = H * W - N
        # if add_length >0:
        x1 = torch.cat([x, x[:, :add_length, :]], dim=1)

        if H < 7:
            H, W = 7, 7
            zero_pad = H * W - (N + add_length)
            x1 = torch.cat([x1, torch.zeros((B, zero_pad, C), device=x1.device)], dim=1)
            add_length += zero_pad
        cnn_feat = x1.transpose(1, 2).view(B, C, H, W)

        #### 第一次 #############

        proj_query = self.query_conv(cnn_feat)
        proj_key = self.key_conv(cnn_feat)
        proj_value = self.value_conv(cnn_feat)

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        # print('attention.shape:', attention.shape)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + cnn_feat


        # try-5
        cnn_feat1 = self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        out = out + cnn_feat1

        #### 重复一次 #######

        proj_query1 = self.query_conv1(out)
        proj_key1 = self.key_conv1(out)
        proj_value1 = self.value_conv1(out)

        energy1 = ca_weight(proj_query1, proj_key1)
        attention1 = F.softmax(energy1, 1)

        out1 = ca_map(attention1, proj_value1)
        # print('out1.shape:', out1.shape)
        # torch.Size([1, 124, 62, 62])
        out1 = self.gamma1 * out1 + out


        out2 = out1.flatten(2).transpose(1, 2)

        if add_length > 0:
            out2 = out2[:, :-add_length,:]

        return out2

class TransMIL(nn.Module):
    def __init__(self, n_classes,dropout,act):
        super(TransMIL, self).__init__()
        # self.pos_layer = PPEG(dim=512)
        self.pos_layer_ccpp = CCAttention(in_dim=512)
        #self.pos_layer = nn.Identity()
        # self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),nn.Dropout(0.25))
        self._fc1 = [nn.Linear(1024, 512)]
        # self._fc1 = [nn.Linear(768, 512)]

        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]

        if dropout:
            self._fc1 += [nn.Dropout(0.25)]

        #self._fc1 += [SwinEncoder(attn='swin',pool='none',n_heads=2,trans_conv=False)]
        
        self._fc1 = nn.Sequential(*self._fc1)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

        self.apply(initialize_weights)

    def forward(self, x):

        h = x.float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        # h = self.pos_layer(h, _H, _W) #[B, N, 512]

        # ---->ccpp
        h = self.pos_layer_ccpp(h)

        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return logits

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024))
    model = TransMIL(n_classes=2,dropout=False,act='relu')
    for k, v in model.state_dict().items():
        print(k)
    # print(model.eval())
    # results_dict = model(data = data)
    # print(results_dict)