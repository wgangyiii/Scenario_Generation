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
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from layers import MLPLayer
from utils import weight_init


class BehaviorGPTHead(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 pos_dim: int,
                 vel_dim: int,
                 theta_dim: int,
                 num_steps: int,
                 num_modes: int) -> None:
        super(BehaviorGPTHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.vel_dim = vel_dim
        self.theta_dim = theta_dim
        self.num_steps = num_steps
        self.num_modes = num_modes

        self.loc_emb = MLPLayer(input_dim=(pos_dim + vel_dim + theta_dim) * num_modes, hidden_dim=hidden_dim,
                                output_dim=hidden_dim)
        self.rnn = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim, bias=True)
        if num_modes > 1:
            self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_modes)
        else:
            self.to_pi = None
        self.to_loc = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                               output_dim=(pos_dim + vel_dim + theta_dim) * num_modes)
        self.to_scale = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                 output_dim=(pos_dim + vel_dim + theta_dim) * num_modes)
        self.apply(weight_init)

    def forward(self,
                data: HeteroData,
                x_a: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.num_modes > 1:
            pi = self.to_pi(x_a)
        else:
            pi = x_a.new_zeros(*x_a.shape[:-1], self.num_modes)
        # predict the first step
        h = x_a
        loc = self.to_loc(h).unsqueeze(-2)
        scale = self.to_scale(h).unsqueeze(-2)
        pos_loc, vel_loc, theta_loc = loc.split([self.pos_dim * self.num_modes,
                                                 self.vel_dim * self.num_modes,
                                                 self.theta_dim * self.num_modes], dim=-1)
        pos_scale, vel_scale, theta_conc = scale.split([self.pos_dim * self.num_modes,
                                                        self.vel_dim * self.num_modes,
                                                        self.theta_dim * self.num_modes], dim=-1)
        theta_loc = torch.tanh(theta_loc) * math.pi
        pos_scale = F.elu(pos_scale, alpha=1.0) + 1.0
        vel_scale = F.elu(vel_scale, alpha=1.0) + 1.0
        theta_conc = 1.0 / (F.elu(theta_conc, alpha=1.0) + 1.0 + 1e-4)
        # predict the subsequent steps autoregressively
        new_pos_loc, new_vel_loc, new_theta_loc = pos_loc, vel_loc, theta_loc
        for _ in range(10 - 1):
            # embed the new state
            i = self.loc_emb(torch.cat([new_pos_loc,
                                        new_vel_loc,
                                        new_theta_loc],
                                       dim=-1)).reshape(-1, self.hidden_dim)  # [A * T, D]
            h = h.reshape(-1, self.hidden_dim)
            # update the hidden state
            h = self.rnn(i, h).reshape(*x_a.shape[:-1], self.hidden_dim)  # [A, T, D]
            # predict the next state
            new_loc = self.to_loc(h).unsqueeze(-2)
            new_scale = self.to_scale(h).unsqueeze(-2)
            new_pos_loc, new_vel_loc, new_theta_loc = new_loc.split([self.pos_dim * self.num_modes,
                                                                     self.vel_dim * self.num_modes,
                                                                     self.theta_dim * self.num_modes],
                                                                    dim=-1)
            new_pos_scale, new_vel_scale, new_theta_conc = new_scale.split([self.pos_dim * self.num_modes,
                                                                            self.vel_dim * self.num_modes,
                                                                            self.theta_dim * self.num_modes],
                                                                           dim=-1)
            new_theta_loc = torch.tanh(new_theta_loc) * math.pi
            new_pos_scale = F.elu(new_pos_scale, alpha=1.0) + 1.0
            new_vel_scale = F.elu(new_vel_scale, alpha=1.0) + 1.0
            new_theta_conc = 1.0 / (F.elu(new_theta_conc, alpha=1.0) + 1.0 + 1e-4)
            # concat with the preceding states
            pos_loc = torch.cat([pos_loc, new_pos_loc], dim=-2)
            vel_loc = torch.cat([vel_loc, new_vel_loc], dim=-2)
            theta_loc = torch.cat([theta_loc, new_theta_loc], dim=-2)
            pos_scale = torch.cat([pos_scale, new_pos_scale], dim=-2)
            vel_scale = torch.cat([vel_scale, new_vel_scale], dim=-2)
            theta_conc = torch.cat([theta_conc, new_theta_conc], dim=-2)
        pos_loc = pos_loc.reshape(*x_a.shape[:-1], -1, self.num_modes, self.pos_dim).transpose(-3, -2)
        vel_loc = vel_loc.reshape(*x_a.shape[:-1], -1, self.num_modes, self.vel_dim).transpose(-3, -2)
        theta_loc = theta_loc.reshape(*x_a.shape[:-1], -1, self.num_modes, self.theta_dim).transpose(-3, -2)
        pos_scale = pos_scale.reshape(*x_a.shape[:-1], -1, self.num_modes, self.pos_dim).transpose(-3, -2)
        vel_scale = vel_scale.reshape(*x_a.shape[:-1], -1, self.num_modes, self.vel_dim).transpose(-3, -2)
        theta_conc = theta_conc.reshape(*x_a.shape[:-1], -1, self.num_modes, self.theta_dim).transpose(-3, -2)

        return {
            'pi': pi,
            'pos_loc': pos_loc,
            'pos_scale': pos_scale,
            'vel_loc': vel_loc,
            'vel_scale': vel_scale,
            'theta_loc': theta_loc,
            'theta_conc': theta_conc,
        }
