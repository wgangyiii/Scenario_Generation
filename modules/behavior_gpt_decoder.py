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
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import subgraph

from layers.attention_layer import AttentionLayer
from layers.fourier_embedding import FourierEmbedding
from utils import angle_between_2d_vectors
from utils import complete_graph
from utils import weight_init
from utils import wrap_angle

try:
    from torch_cluster import knn
    from torch_cluster import knn_graph
    is_torch_cluster_available = True
except ImportError:
    knn = object
    knn_graph = object
    is_torch_cluster_available = False


class BehaviorGPTDecoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_steps: int,
                 time_span: Optional[int],
                 num_m2a_nbrs: int,
                 num_a2a_nbrs: int,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(BehaviorGPTDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.time_span = time_span if time_span is not None else num_steps
        self.num_m2a_nbrs = num_m2a_nbrs
        self.num_a2a_nbrs = num_a2a_nbrs
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        num_agent_types = 5
        num_map_types = 17
        input_dim_x_a = 5
        input_dim_x_m = input_dim - 1
        input_dim_r_t = 2 + input_dim
        input_dim_r_m2a = 1 + input_dim
        input_dim_r_a2a = 1 + input_dim

        self.type_a_emb = nn.Embedding(num_agent_types, hidden_dim)
        self.type_m_emb = nn.Embedding(num_map_types, hidden_dim)
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.x_m_emb = FourierEmbedding(input_dim=input_dim_x_m, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_patch_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_m2a_emb = FourierEmbedding(input_dim=input_dim_r_m2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.to_patch = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                                       bipartite=False, has_pos_emb=True)
        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.apply(weight_init)

    def forward(self,
                data: HeteroData) -> torch.Tensor:
        mask = data['agent']['valid_mask'][:, :self.num_steps].contiguous()
        pos_a = data['agent']['position'][:, :self.num_steps, :self.input_dim].contiguous()
        head_a = data['agent']['heading'][:, :self.num_steps].contiguous()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pos_m = data['map_point']['position'][:, :self.input_dim].contiguous()
        orient_m = data['map_point']['orientation'].contiguous()

        vel = data['agent']['velocity'][:, :self.num_steps, :self.input_dim].contiguous()
        length = data['agent']['length'][:, :self.num_steps].contiguous()
        width = data['agent']['width'][:, :self.num_steps].contiguous()
        height = data['agent']['height'][:, :self.num_steps].contiguous()
        type_a_emb = [self.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=self.num_steps, dim=0)]
        type_m_emb = [self.type_m_emb(data['map_point']['type'].long())]

        x_a = torch.stack(
            [torch.norm(vel[:, :, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2]),
             length,
             width,
             height], dim=-1)
        if self.input_dim == 2:
            x_m = data['map_point']['magnitude'].unsqueeze(-1)
        elif self.input_dim == 3:
            x_m = torch.stack([data['map_point']['magnitude'], data['map_point']['height']], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        valid_index_t = torch.where(mask.view(-1))[0]
        x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=type_a_emb,
                           valid_index=valid_index_t)
        x_a = x_a.view(-1, self.num_steps, self.hidden_dim)
        x_m = self.x_m_emb(continuous_inputs=x_m, categorical_embs=type_m_emb)

        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_patch = edge_index_t[:, edge_index_t[1] - edge_index_t[0] < 10]
        rel_pos_patch = pos_t[edge_index_patch[0]] - pos_t[edge_index_patch[1]]
        rel_head_patch = wrap_angle(head_t[edge_index_patch[0]] - head_t[edge_index_patch[1]])
        if self.input_dim == 2:
            r_patch = torch.stack(
                [torch.norm(rel_pos_patch[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_patch[1]],
                                          nbr_vector=rel_pos_patch[:, :2]),
                 rel_head_patch,
                 edge_index_patch[0] - edge_index_patch[1]], dim=-1)
        else:
            r_patch = torch.stack(
                [torch.norm(rel_pos_patch[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_patch[1]],
                                          nbr_vector=rel_pos_patch[:, :2]),
                 rel_pos_patch[:, -1],
                 rel_head_patch,
                 edge_index_patch[0] - edge_index_patch[1]], dim=-1)
        r_patch = self.r_patch_emb(continuous_inputs=r_patch, categorical_embs=None)
        edge_index_t = edge_index_t[:, (edge_index_t[1] - edge_index_t[0]) % 10 == 0]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        if self.input_dim == 2:
            r_t = torch.stack(
                [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
                 rel_head_t,
                 edge_index_t[0] - edge_index_t[1]], dim=-1)
        else:
            r_t = torch.stack(
                [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
                 rel_pos_t[:, -1],
                 rel_head_t,
                 edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        mask_t = mask.reshape(-1)
        if is_torch_cluster_available:
            if isinstance(data, Batch):
                batch_t = data['agent']['batch'].repeat_interleave(self.num_steps)
                batch_m = data['map_point']['batch']
            else:
                batch_t = pos_t.new_zeros(data['agent']['num_nodes'] * self.num_steps, dtype=torch.long)
                batch_m = pos_m.new_zeros(data['map_point']['num_nodes'], dtype=torch.long)
            edge_index_m2a = knn(x=pos_m[:, :2], y=pos_t[:, :2], k=self.num_m2a_nbrs, batch_x=batch_m, batch_y=batch_t)
            edge_index_m2a = edge_index_m2a[[1, 0]]
        else:
            num_agents_batch = data['agent']['ptr'][1:] - data['agent']['ptr'][:-1]
            num_agents_batch_t = num_agents_batch * self.num_steps
            agent_ptr_t = num_agents_batch_t.cumsum(dim=0)
            agent_ptr_t = torch.cat([agent_ptr_t.new_zeros(1), agent_ptr_t], dim=0)
            edge_index_m2a = complete_graph(num_nodes=(data['map_point']['num_nodes'],
                                                       data['agent']['num_nodes'] * self.num_steps),
                                            ptr=(data['map_point']['ptr'], agent_ptr_t),
                                            loop=False,
                                            device=x_a.device)
        edge_index_m2a = edge_index_m2a[:, mask_t[edge_index_m2a[1]]]
        valid_index_m = edge_index_m2a[0].unique()
        rel_pos_m2a = pos_m[edge_index_m2a[0]] - pos_t[edge_index_m2a[1]]
        if not is_torch_cluster_available:
            dist_m2a = torch.norm(rel_pos_m2a, p=2, dim=-1)
            mask_m2a = dist_m2a < 20.0
            edge_index_m2a = edge_index_m2a[:, mask_m2a]
            rel_pos_m2a = rel_pos_m2a[mask_m2a]
        rel_orient_m2a = wrap_angle(orient_m[edge_index_m2a[0]] - head_t[edge_index_m2a[1]])
        if self.input_dim == 2:
            r_m2a = torch.stack(
                [torch.norm(rel_pos_m2a[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_m2a[1]], nbr_vector=rel_pos_m2a[:, :2]),
                 rel_orient_m2a], dim=-1)
        else:
            r_m2a = torch.stack(
                [torch.norm(rel_pos_m2a[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_m2a[1]], nbr_vector=rel_pos_m2a[:, :2]),
                 rel_pos_m2a[:, -1],
                 rel_orient_m2a], dim=-1)
        r_m2a = self.r_m2a_emb(continuous_inputs=r_m2a, categorical_embs=None)

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        mask_s = mask.transpose(0, 1).reshape(-1)
        valid_index_s = torch.where(mask_s)[0]
        if is_torch_cluster_available:
            if isinstance(data, Batch):
                batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                     for t in range(self.num_steps)], dim=0)
            else:
                batch_s = torch.arange(self.num_steps,
                                       device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
            edge_index_a2a = knn_graph(x=pos_s[:, :2], k=self.num_a2a_nbrs, batch=batch_s, loop=False)
        else:
            agent_ptr_s = torch.cat([data['agent']['ptr'][1:] + data['agent']['num_nodes'] * t
                                     for t in range(self.num_steps)], dim=0)
            agent_ptr_s = torch.cat([agent_ptr_s.new_zeros(1), agent_ptr_s], dim=0)
            edge_index_a2a = complete_graph(num_nodes=data['agent']['num_nodes'] * self.num_steps,
                                            ptr=agent_ptr_s,
                                            loop=False,
                                            device=x_a.device)
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        if not is_torch_cluster_available:
            dist_a2a = torch.norm(rel_pos_a2a, p=2, dim=-1)
            mask_a2a = dist_a2a < 20.0
            edge_index_a2a = edge_index_a2a[:, mask_a2a]
            rel_pos_a2a = rel_pos_a2a[mask_a2a]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        if self.input_dim == 2:
            r_a2a = torch.stack(
                [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
                 rel_head_a2a], dim=-1)
        else:
            r_a2a = torch.stack(
                [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
                 rel_pos_a2a[:, -1],
                 rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

        x_a = x_a.reshape(-1, self.hidden_dim)
        x_a = self.to_patch(x_a, r_patch, edge_index_patch, valid_index=valid_index_t)
        for i in range(self.num_layers):
            x_a = x_a.reshape(-1, self.hidden_dim)
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t, valid_index=valid_index_t)
            x_a = self.m2a_attn_layers[i]((x_m, x_a), r_m2a, edge_index_m2a, valid_index=(valid_index_m, valid_index_t))
            x_a = x_a.reshape(-1, self.num_steps,
                              self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a, valid_index=valid_index_s)
            x_a = x_a.reshape(self.num_steps, -1, self.hidden_dim).transpose(0, 1)

        return x_a
