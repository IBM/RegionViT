# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn

from timm.models.layers import to_2tuple, trunc_normal_


class AttentionWithRelPos(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 attn_map_dim=None, num_cls_tokens=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num_cls_tokens = num_cls_tokens
        if attn_map_dim is not None:
            one_dim = attn_map_dim[0]
            rel_pos_dim = (2 * one_dim - 1)
            self.rel_pos = nn.Parameter(torch.zeros(num_heads, rel_pos_dim ** 2))
            tmp = torch.arange(rel_pos_dim ** 2).reshape((rel_pos_dim, rel_pos_dim))
            out = []
            offset_x = offset_y = one_dim // 2
            for y in range(one_dim):
                for x in range(one_dim):
                    for dy in range(one_dim):
                        for dx in range(one_dim):
                            out.append(tmp[dy - y + offset_y, dx - x + offset_x])
            self.rel_pos_index = torch.tensor(out, dtype=torch.long)
            trunc_normal_(self.rel_pos, std=.02)
        else:
            self.rel_pos = None

    def forward(self, x, patch_attn=False, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rel_pos is not None and patch_attn:
            # use for the indicating patch + cls:
            rel_pos = self.rel_pos[:, self.rel_pos_index.to(attn.device)].reshape(self.num_heads, N - self.num_cls_tokens, N - self.num_cls_tokens)
            attn[:, :, self.num_cls_tokens:, self.num_cls_tokens:] = attn[:, :, self.num_cls_tokens:, self.num_cls_tokens:] + rel_pos

        if mask is not None:
            ## mask is only (BH_sW_s)(ksks)(ksks), need to expand it
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn.masked_fill(mask == 0, torch.finfo(attn.dtype).min)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
