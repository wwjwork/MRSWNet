
from functools import partial
#from collections import OrderedDict

import torch
import torch.nn as nn
#import torch.nn.functional as F
#import numpy as np
#import math


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0.1, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads                 
        self.scale = qk_scale or head_dim ** -0.5   

        self.q = nn.Linear(dim, dim, bias=qkv_bias)  
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio == 16:
            self.sr   = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=5, stride=4, padding=3, groups=dim),
                                     nn.BatchNorm2d(dim), 
                                     nn.Conv2d(dim, dim, kernel_size=5, stride=4, padding=0, groups=dim),)
            self.norm = nn.LayerNorm(dim)
        elif sr_ratio == 8:
            self.sr   = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=5, stride=4, padding=3, groups=dim),
                                     nn.BatchNorm2d(dim), 
                                     nn.Conv2d(dim, dim, kernel_size=5, stride=2, padding=0, groups=dim),)
            self.norm = nn.LayerNorm(dim)
        elif sr_ratio == 4:
            self.sr   = nn.Conv2d(dim, dim, kernel_size=4, stride=4, padding=0, groups=dim)
            self.norm = nn.LayerNorm(dim)
        elif sr_ratio == 2:
            self.sr   = nn.Conv2d(dim, dim, kernel_size=2, stride=2, padding=0, groups=dim)
            self.norm = nn.LayerNorm(dim)


    def forward(self, x, H, W):
        B, NUM, N, C = x.shape
        q = self.q(x).reshape(B, NUM, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)   

        if self.sr_ratio > 1:
            xx=[]
            x_ = x.permute(1, 0, 3, 2).reshape(NUM, B, C, H, W)
            for ii in range(NUM):
                x__ = self.sr(x_[ii])
                xx.append(x__)
            x_ = torch.stack(xx,0)
            x_ = x_.reshape(NUM, B, C, -1).permute(1, 0, 3, 2)   
            x_ = self.norm(x_)                                   
            kv = self.kv(x_).reshape(B, NUM, -1, 2, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        else:
            kv = self.kv(x).reshape(B, NUM, -1, 2, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, NUM, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., sr_ratio=4, act_layer=nn.GELU, norm_layer = nn.LayerNorm):
        super(Block, self).__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop_ratio, proj_drop=drop_path_ratio, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        B,N,H,W,C = x.shape
        x = x.view(B,N,-1,C)
        x = x + self.drop_path(self.attn(self.norm1(x),H,W))
        x = x + self.drop_path(self.mlp (self.norm2(x)))
        x = x.reshape(B,N,H,W,C)

        return x


class PSSA(nn.Module):
    def __init__(self, embed_dim=256, depth=4, num_heads=4,
                 mlp_ratio=4, sr_ratio=[4], cr_ratio=[4],
                 qkv_bias=True, qk_scale=None, 
                 drop_ratio=0., attn_drop_ratio=0.1, drop_path_ratio=0.1, 
                 norm_layer=None, act_layer=None):
        super(PSSA, self).__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer  = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule

        self.cr_ratio = cr_ratio
        self.norm = norm_layer(embed_dim)
        self.fuse = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)

        # num_patches = int((img_size[0]/patch_size)*(img_size[1]/patch_size))
        # self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # self.pos_drop   = nn.Dropout(p=drop_ratio)

        self.blocks = []
        num_blocks = min(len(sr_ratio), 5)

        for i in range(num_blocks):
            block = nn.Sequential(*[
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[j],
                    norm_layer=norm_layer, act_layer=act_layer, sr_ratio=sr_ratio[i])
                for j in range(depth)
            ])
            setattr(self, f'blocks{i+1}', block)
            self.blocks.append(block)
        self.apply(_init_vit_weights)


    def forward(self, x):
        identity = x
        cr = self.cr_ratio
        B,C,H,W = x.shape
        
        num_levels = min(len(cr), 5)
        x_list = []

        for i in range(num_levels):
            cr_i = 2**cr[i]
            x_i = x.view(B,C,H//cr_i,cr_i,W//cr_i,cr_i).permute(0,3,5,2,4,1).reshape(B,-1,H//cr_i,W//cr_i,C)
            x_i = getattr(self, f'blocks{i+1}')(x_i)
            x_i = self.norm(x_i)
            x_i = x_i.view(B,cr_i,cr_i,H//cr_i,W//cr_i,C).permute(0,5,3,1,4,2).reshape(B,C,H,W)
            x_list.append(x_i)
        xout = self.fuse(sum(x_list)) + identity

        return xout


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)







if __name__ == '__main__':

    input = torch.rand(4,64,160,160)
   

