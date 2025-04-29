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
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class SimAgentFilter(BaseTransform):

    def __init__(self,
                 max_num_agents: int,
                 num_historical_steps: int) -> None:
        self.max_num_agents = max_num_agents
        self.num_historical_steps = num_historical_steps

    def __call__(self, data: HeteroData) -> HeteroData:
        if data['agent']['num_nodes'] <= self.max_num_agents:
            return data

        pos = data['agent']['position']
        av_index = data['agent']['av_index']
        current_step = self.num_historical_steps - 1
        valid_mask = data['agent']['valid_mask']

        # add sdv and track_to_predict
        agent_inds = pos.new_tensor([av_index], dtype=torch.long)
        target_mask = data['agent']['target_mask'].clone()
        target_mask[av_index] = False
        agent_inds = torch.cat([agent_inds, torch.where(data['agent']['target_mask'])[0]], dim=0)

        # add context agents
        max_num_context_agents = self.max_num_agents - agent_inds.numel()
        dist = torch.norm(pos[:, current_step, :2] - pos[av_index, current_step, :2], p=2, dim=-1)
        valid_dist_mask = valid_mask[:, current_step].clone()
        valid_dist_mask[agent_inds] = False
        dist[~valid_dist_mask] = torch.finfo(torch.float).max
        context_agent_inds = torch.topk(dist, k=min(max_num_context_agents, data['agent']['num_nodes']),
                                        largest=False)[1]
        context_agent_inds = context_agent_inds[valid_dist_mask[context_agent_inds]]
        agent_inds = torch.cat([agent_inds, context_agent_inds], dim=0)
        max_num_context_agents = self.max_num_agents - agent_inds.numel()
        if max_num_context_agents > 0:
            num_states = valid_mask.sum(dim=-1)
            valid_state_mask = torch.ones_like(valid_dist_mask)
            valid_state_mask[agent_inds] = False
            num_states[~valid_state_mask] = torch.iinfo(torch.long).min
            context_agent_inds = torch.topk(num_states, k=min(max_num_context_agents, data['agent']['num_nodes']),
                                            largest=True)[1]
            context_agent_inds = context_agent_inds[valid_state_mask[context_agent_inds]]
            agent_inds = torch.cat([agent_inds, context_agent_inds], dim=0)

        agent_inds = torch.unique(agent_inds)

        data['agent']['num_nodes'] = agent_inds.numel()
        data['agent']['av_index'] = 0
        data['agent']['valid_mask'] = data['agent']['valid_mask'][agent_inds].clone()
        data['agent']['target_mask'] = data['agent']['target_mask'][agent_inds].clone()
        data['agent']['id'] = data['agent']['id'][agent_inds].clone()
        data['agent']['type'] = data['agent']['type'][agent_inds].clone()
        data['agent']['position'] = data['agent']['position'][agent_inds].clone()
        data['agent']['heading'] = data['agent']['heading'][agent_inds].clone()
        data['agent']['velocity'] = data['agent']['velocity'][agent_inds].clone()
        data['agent']['length'] = data['agent']['length'][agent_inds].clone()
        data['agent']['width'] = data['agent']['width'][agent_inds].clone()
        data['agent']['height'] = data['agent']['height'][agent_inds].clone()

        return data
