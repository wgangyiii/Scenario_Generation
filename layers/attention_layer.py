# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from utils import weight_init


class AttentionLayer(MessagePassing):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 bipartite: bool,
                 has_pos_emb: bool,
                 dst_pos_emb: bool = False,
                 activation: str = 'relu',
                 **kwargs) -> None:
        super(AttentionLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.has_pos_emb = has_pos_emb
        self.dst_pos_emb = dst_pos_emb
        self.activation = activation
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(hidden_dim, num_heads * head_dim)
        self.to_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, num_heads * head_dim)
        if has_pos_emb:
            self.to_k_r = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
            self.to_v_r = nn.Linear(hidden_dim, num_heads * head_dim)
        if dst_pos_emb:
            self.to_q_r = nn.Linear(hidden_dim, num_heads * head_dim)
        self.to_s = nn.Linear(hidden_dim, num_heads * head_dim)
        self.to_g = nn.Linear(num_heads * head_dim + hidden_dim, num_heads * head_dim)
        self.to_out = nn.Linear(num_heads * head_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        if activation == 'relu':
            self.ff_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
        elif activation == 'swi_glu':
            self.w1 = nn.Linear(hidden_dim, hidden_dim * 4)
            self.w2 = nn.Linear(hidden_dim * 4, hidden_dim)
            self.w3 = nn.Linear(hidden_dim, hidden_dim * 4)
        else:
            raise ValueError('{} is not a valid activation function'.format(activation))
        if bipartite:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src
        if has_pos_emb:
            self.attn_prenorm_r = nn.LayerNorm(hidden_dim)
        if dst_pos_emb:
            self.attn_prenorm_r_dst = nn.LayerNorm(hidden_dim)
        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def forward(self,
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                r: Optional[torch.Tensor],
                edge_index: torch.Tensor,
                r_dst: Optional[torch.Tensor] = None,
                valid_index: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            size = (x.size(0), x.size(0))
            valid_index_src = valid_index_dst = valid_index
            if valid_index is None:
                x_src = x_dst = self.attn_prenorm_x_src(x)
            else:
                x_src = x_dst = self.attn_prenorm_x_src(x[valid_index])
        else:
            x_src, x_dst = x
            size = (x_src.size(0), x_dst.size(0))
            if valid_index is None:
                valid_index_src = valid_index_dst = None
                x_src = self.attn_prenorm_x_src(x_src)
                x_dst = self.attn_prenorm_x_dst(x_dst)
            else:
                valid_index_src, valid_index_dst = valid_index
                x_src = self.attn_prenorm_x_src(x_src[valid_index_src])
                x_dst = self.attn_prenorm_x_dst(x_dst[valid_index_dst])
            x = x[1]
        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)
        if self.dst_pos_emb and r_dst is not None:
            if valid_index_dst is None:
                r_dst = self.attn_prenorm_r_dst(r_dst)
            else:
                r_dst = self.attn_prenorm_r_dst(r_dst[valid_index_dst])
        if valid_index is None:
            x = x + self.attn_postnorm(self._attn_block(x_src, x_dst, r, r_dst, edge_index, size))
            x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(x)))
        else:
            x_valid = x[valid_index_dst] + self.attn_postnorm(
                self._attn_block(x_src, x_dst, r, r_dst, edge_index, size, valid_index_src, valid_index_dst))
            x = torch.zeros_like(x).scatter_(
                dim=0,
                index=valid_index_dst.unsqueeze(-1).repeat(1, x.size(-1)),
                src=x_valid + self.ff_postnorm(self._ff_block(self.ff_prenorm(x_valid))),
            )
        return x

    def message(self,
                q_i: torch.Tensor,
                k_j: torch.Tensor,
                v_j: torch.Tensor,
                r: Optional[torch.Tensor],
                index: Optional[torch.Tensor],
                ptr: Optional[torch.Tensor]) -> torch.Tensor:
        if self.has_pos_emb and r is not None:
            k_j = k_j + self.to_k_r(r).view(-1, self.num_heads, self.head_dim)
            v_j = v_j + self.to_v_r(r).view(-1, self.num_heads, self.head_dim)
        sim = (q_i * k_j).sum(dim=-1) * self.scale
        attn = softmax(sim, index, ptr)
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x_dst: torch.Tensor,
               valid_index_dst: Optional[torch.Tensor]) -> torch.Tensor:
        if valid_index_dst is None:
            inputs = inputs.view(-1, self.num_heads * self.head_dim)
        else:
            inputs = inputs.view(-1, self.num_heads * self.head_dim)[valid_index_dst]
            x_dst = x_dst[valid_index_dst]
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))
        return inputs + g * (self.to_s(x_dst) - inputs)

    def _attn_block(self,
                    x_src: torch.Tensor,
                    x_dst: torch.Tensor,
                    r: Optional[torch.Tensor],
                    r_dst: Optional[torch.Tensor],
                    edge_index: torch.Tensor,
                    size: Tuple[int, int],
                    valid_index_src: Optional[torch.Tensor] = None,
                    valid_index_dst: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.to_q(x_dst)
        if self.dst_pos_emb and r_dst is not None:
            q = q + self.to_q_r(r_dst)
        k = self.to_k(x_src)
        v = self.to_v(x_src)
        if valid_index_src is not None:
            k = k.new_zeros((size[0], k.size(-1))).scatter_(
                dim=0,
                index=valid_index_src.unsqueeze(-1).repeat(1, k.size(-1)),
                src=k,
            )
            v = v.new_zeros((size[0], v.size(-1))).scatter_(
                dim=0,
                index=valid_index_src.unsqueeze(-1).repeat(1, v.size(-1)),
                src=v,
            )
        if valid_index_dst is not None:
            x_dst = x_dst.new_zeros((size[1], x_dst.size(-1))).scatter_(
                dim=0,
                index=valid_index_dst.unsqueeze(-1).repeat(1, x_dst.size(-1)),
                src=x_dst,
            )
            q = q.new_zeros((size[1], q.size(-1))).scatter_(
                dim=0,
                index=valid_index_dst.unsqueeze(-1).repeat(1, q.size(-1)),
                src=q,
            )
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)
        # propagate_type: (x_dst: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, r: Optional[torch.Tensor], valid_index_dst: Optional[torch.Tensor])
        agg = self.propagate(edge_index=edge_index, x_dst=x_dst, q=q, k=k, v=v, r=r, valid_index_dst=valid_index_dst,
                             size=size)
        return self.to_out(agg)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == 'relu':
            return self.ff_mlp(x)
        elif self.activation == 'swi_glu':
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
        else:
            raise ValueError('{} is not a valid activation function'.format(self.activation))
