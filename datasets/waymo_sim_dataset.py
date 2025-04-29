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
import os
import pickle
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf
import torch
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm

from utils import interp_arc

try:
    from waymo_open_dataset.protos import map_pb2
    from waymo_open_dataset.protos import scenario_pb2
except ImportError:
    map_pb2 = object
    scenario_pb2 = object

tf.config.set_visible_devices([], 'GPU')


class WaymoSimDataset(Dataset):
    """Dataset class for Waymo Open Sim Agents Challenge.

    See https://waymo.com/open/data/motion for more information about the dataset.

    Args:
        root (string): the root folder of the dataset.
        split (string): specify the split of the dataset: `"train"` | `"val"` | `"test"`.
        interactive (boolean, Optional): if True, use the interactive split of the validation/test set. (default: False)
        raw_dir (string, optional): optionally specify the directory of the raw data. By default, the raw directory is
            path/to/root/split/raw. If specified, the path of the raw tfrecord files is path/to/raw_dir/*.
            (default: None)
        processed_dir (string, optional): optionally specify the directory of the processed data. By default, the
            processed directory is path/to/root/split/processed/. If specified, the path of the processed .pkl files is
            path/to/processed_dir/*.pkl. If all .pkl files exist in the processed directory, data preprocessing will be
            skipped. (default: None)
        transform (callable, optional): a function/transform that takes in an :obj:`torch_geometric.data.Data` object
            and returns a transformed version. The data object will be transformed before every access. (default: None)
        dim (int, Optional): 2D or 3D data. (default: 3)
        num_historical_steps (int, Optional): the number of historical time steps. (default: 11)
        num_future_steps (int, Optional): the number of future time steps. (default: 80)
        resolution_meters (float, Optional): the resolution of HD map's sampling distance in meters. (default: 5.0)
    """

    def __init__(self,
                 root: str,
                 split: str,
                 interactive: bool = False,
                 raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 dim: int = 3,
                 num_historical_steps: int = 11,
                 num_future_steps: int = 80,
                 resolution_meters: float = 5.0,
                 **kwargs) -> None:
        root = os.path.expanduser(os.path.normpath(root))
        if not os.path.isdir(root):
            os.makedirs(root)
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        self.split = split
        self.dir = {
            'train': 'training',
            'val': 'validation' if not interactive else 'validation_interactive',
            'test': 'testing' if not interactive else 'testing_interactive',
        }[split]

        if raw_dir is None:
            raw_dir = os.path.join(root, self.dir, 'raw')
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                self._raw_file_names = [name for name in os.listdir(self._raw_dir) if
                                        os.path.isfile(os.path.join(self._raw_dir, name))]
            else:
                self._raw_file_names = []
        else:
            raw_dir = os.path.expanduser(os.path.normpath(raw_dir))
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                self._raw_file_names = [name for name in os.listdir(self._raw_dir) if
                                        os.path.isfile(os.path.join(self._raw_dir, name))]
            else:
                self._raw_file_names = []

        if processed_dir is None:
            processed_dir = os.path.join(root, self.dir, 'processed')
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = [name for name in os.listdir(self._processed_dir) if
                                              os.path.isfile(os.path.join(self._processed_dir, name)) and
                                              name.endswith(('pkl', 'pickle'))]
            else:
                self._processed_file_names = []
        else:
            processed_dir = os.path.expanduser(os.path.normpath(processed_dir))
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = [name for name in os.listdir(self._processed_dir) if
                                              os.path.isfile(os.path.join(self._processed_dir, name)) and
                                              name.endswith(('pkl', 'pickle'))]
            else:
                self._processed_file_names = []

        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.resolution_meters = resolution_meters
        self._num_raw_files = {
            'train': 1000,
            'val': 150,
            'test': 150,
        }[split]
        self._num_samples = {
            'train': 486995,
            'val': 44097 if not interactive else 43479,
            'test': 44920 if not interactive else 44154,
        }[split]
        self._lane_type_dict = {
            map_pb2.LaneCenter.TYPE_UNDEFINED: 'UNDEFINED',
            map_pb2.LaneCenter.TYPE_FREEWAY: 'FREEWAY',
            map_pb2.LaneCenter.TYPE_SURFACE_STREET: 'SURFACE_STREET',
            map_pb2.LaneCenter.TYPE_BIKE_LANE: 'BIKE_LANE',
        }
        self._road_edge_type_dict = {
            map_pb2.RoadEdge.TYPE_UNKNOWN: 'ROAD_EDGE_UNKNOWN',
            map_pb2.RoadEdge.TYPE_ROAD_EDGE_BOUNDARY: 'ROAD_EDGE_BOUNDARY',
            map_pb2.RoadEdge.TYPE_ROAD_EDGE_MEDIAN: 'ROAD_EDGE_MEDIAN',
        }
        self._road_line_type_dict = {
            map_pb2.RoadLine.TYPE_UNKNOWN: 'ROAD_LINE_UNKNOWN',
            map_pb2.RoadLine.TYPE_BROKEN_SINGLE_WHITE: 'ROAD_LINE_BROKEN_SINGLE_WHITE',
            map_pb2.RoadLine.TYPE_SOLID_SINGLE_WHITE: 'ROAD_LINE_SOLID_SINGLE_WHITE',
            map_pb2.RoadLine.TYPE_SOLID_DOUBLE_WHITE: 'ROAD_LINE_SOLID_DOUBLE_WHITE',
            map_pb2.RoadLine.TYPE_BROKEN_SINGLE_YELLOW: 'ROAD_LINE_BROKEN_SINGLE_YELLOW',
            map_pb2.RoadLine.TYPE_BROKEN_DOUBLE_YELLOW: 'ROAD_LINE_BROKEN_DOUBLE_YELLOW',
            map_pb2.RoadLine.TYPE_SOLID_SINGLE_YELLOW: 'ROAD_LINE_SOLID_SINGLE_YELLOW',
            map_pb2.RoadLine.TYPE_SOLID_DOUBLE_YELLOW: 'ROAD_LINE_SOLID_DOUBLE_YELLOW',
            map_pb2.RoadLine.TYPE_PASSING_DOUBLE_YELLOW: 'ROAD_LINE_PASSING_DOUBLE_YELLOW',
        }
        self._agent_types = ['UNSET', 'VEHICLE', 'PEDESTRIAN', 'CYCLIST', 'OTHER']
        self._point_types = ['UNDEFINED', 'FREEWAY', 'SURFACE_STREET', 'BIKE_LANE',
                             'CROSSWALK', 'SPEED_BUMP', 'DRIVEWAY',
                             'ROAD_EDGE_BOUNDARY', 'ROAD_EDGE_MEDIAN',
                             'ROAD_LINE_BROKEN_SINGLE_WHITE', 'ROAD_LINE_SOLID_SINGLE_WHITE',
                             'ROAD_LINE_SOLID_DOUBLE_WHITE', 'ROAD_LINE_BROKEN_SINGLE_YELLOW',
                             'ROAD_LINE_BROKEN_DOUBLE_YELLOW', 'ROAD_LINE_SOLID_SINGLE_YELLOW',
                             'ROAD_LINE_SOLID_DOUBLE_YELLOW', 'ROAD_LINE_PASSING_DOUBLE_YELLOW']
        super(WaymoSimDataset, self).__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)

    @property
    def raw_dir(self) -> str:
        return self._raw_dir

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    def download(self) -> None:
        raise NotImplementedError

    def process(self) -> None:
        for raw_file_name in tqdm(self.raw_file_names):
            records = tf.data.TFRecordDataset(os.path.join(self.raw_dir, raw_file_name))
            for record in records:
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(record.numpy())
                self._processed_file_names.append(f'{scenario.scenario_id}.pkl')
                data = dict()
                data['scenario_id'] = scenario.scenario_id
                data['agent'] = self.get_agent_features(scenario)
                data['map_point'] = self.get_map_features(scenario)
                with open(os.path.join(self.processed_dir, f'{scenario.scenario_id}.pkl'), 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_agent_features(self, scenario) -> Dict[str, Any]:
        agent_ids = [track.id for track in scenario.tracks]
        num_agents = len(agent_ids)
        av_idx = scenario.sdc_track_index

        valid_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        target_mask = torch.zeros(num_agents, dtype=torch.bool)
        agent_id = torch.zeros(num_agents, dtype=torch.int32)
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        velocity = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        length = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        width = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        height = torch.zeros(num_agents, self.num_steps, dtype=torch.float)

        for track in scenario.tracks:
            agent_idx = agent_ids.index(track.id)
            agent_steps = [t for t, state in enumerate(track.states) if state.valid]
            valid_mask[agent_idx, agent_steps] = True
            agent_id[agent_idx] = track.id
            agent_type[agent_idx] = track.object_type
            num_states = len(track.states)
            position[agent_idx, :num_states, 0] = torch.tensor([state.center_x for state in track.states],
                                                               dtype=torch.float)
            position[agent_idx, :num_states, 1] = torch.tensor([state.center_y for state in track.states],
                                                               dtype=torch.float)
            if self.dim == 3:
                position[agent_idx, :num_states, 2] = torch.tensor([state.center_z for state in track.states],
                                                                   dtype=torch.float)
            heading[agent_idx, :num_states] = torch.tensor([state.heading for state in track.states], dtype=torch.float)
            velocity[agent_idx, :num_states, 0] = torch.tensor([state.velocity_x for state in track.states],
                                                               dtype=torch.float)
            velocity[agent_idx, :num_states, 1] = torch.tensor([state.velocity_y for state in track.states],
                                                               dtype=torch.float)
            length[agent_idx, :num_states] = torch.tensor([abs(state.length) for state in track.states],
                                                          dtype=torch.float)
            width[agent_idx, :num_states] = torch.tensor([abs(state.width) for state in track.states],
                                                         dtype=torch.float)
            height[agent_idx, :num_states] = torch.tensor([abs(state.height) for state in track.states],
                                                          dtype=torch.float)

        for track_to_predict in scenario.tracks_to_predict:
            target_mask[track_to_predict.track_index] = True

        return {
            'num_nodes': num_agents,
            'av_index': av_idx,
            'valid_mask': valid_mask,
            'target_mask': target_mask,
            'id': agent_id,
            'type': agent_type,
            'position': position,
            'heading': heading,
            'velocity': velocity,
            'length': length,
            'width': width,
            'height': height,
        }

    def get_map_features(self, scenario) -> Dict[str, Any]:
        lane_ids, road_line_ids, road_edge_ids, crosswalk_ids, speed_bump_ids, driveway_ids = (set(), set(), set(),
                                                                                               set(), set(), set())
        for map_feature in scenario.map_features:
            feature_type = map_feature.WhichOneof('feature_data')
            if feature_type == 'lane':
                lane = getattr(map_feature, 'lane')
                if len(lane.polyline) > 1:
                    lane_ids.add(map_feature.id)
            elif feature_type == 'road_line':
                road_line = getattr(map_feature, 'road_line')
                if len(road_line.polyline) > 1:
                    road_line_ids.add(map_feature.id)
            elif feature_type == 'road_edge':
                road_edge = getattr(map_feature, 'road_edge')
                if len(road_edge.polyline) > 1:
                    road_edge_ids.add(map_feature.id)
            elif feature_type == 'crosswalk':
                crosswalk = getattr(map_feature, 'crosswalk')
                if len(crosswalk.polygon) > 2:
                    crosswalk_ids.add(map_feature.id)
            elif feature_type == 'speed_bump':
                speed_bump = getattr(map_feature, 'speed_bump')
                if len(speed_bump.polygon) > 2:
                    speed_bump_ids.add(map_feature.id)
            elif feature_type == 'driveway':
                driveway = getattr(map_feature, 'driveway')
                if len(driveway.polygon) > 2:
                    driveway_ids.add(map_feature.id)
            else:
                continue

        lanes, road_lines, road_edges, stop_signs, crosswalks, speed_bumps, driveways = (dict(), dict(), dict(), dict(),
                                                                                         dict(), dict(), dict())
        point_position, point_orientation, point_magnitude, point_height, point_type = [], [], [], [], []

        feature_type_dict = {
            'lane': lanes,
            'road_line': road_lines,
            'road_edge': road_edges,
            'stop_sign': stop_signs,
            'crosswalk': crosswalks,
            'speed_bump': speed_bumps,
            'driveway': driveways,
        }
        for map_feature in scenario.map_features:
            feature_type = map_feature.WhichOneof('feature_data')
            feature_type_dict[feature_type][map_feature.id] = getattr(map_feature, feature_type)

        for lane_id, lane in lanes.items():
            if lane_id not in lane_ids:
                continue
            raw_centerline = torch.tensor([[point.x, point.y, point.z] for point in lane.polyline], dtype=torch.float)
            step_size = math.floor(self.resolution_meters / 0.5)
            sample_inds = torch.arange(0, raw_centerline.size(0), step_size)
            if (raw_centerline.size(0) - 1) % step_size != 0:
                sample_inds = torch.cat([sample_inds, torch.tensor([raw_centerline.size(0) - 1])], dim=0)
            sampled_centerline = raw_centerline[sample_inds]
            centerline_vectors = sampled_centerline[1:] - sampled_centerline[:-1]
            point_position.append(sampled_centerline[:-1, :self.dim])
            point_orientation.append(torch.atan2(centerline_vectors[:, 1], centerline_vectors[:, 0]))
            point_magnitude.append(torch.norm(centerline_vectors[:, :2], p=2, dim=-1))
            point_height.append(centerline_vectors[:, 2])
            point_type.append(torch.full((len(centerline_vectors),),
                                         self._point_types.index(self._lane_type_dict[lane.type]),
                                         dtype=torch.uint8))

        for road_line_id, road_line in road_lines.items():
            if road_line_id not in road_line_ids:
                continue
            raw_road_line = torch.tensor([[point.x, point.y, point.z] for point in road_line.polyline],
                                         dtype=torch.float)
            step_size = math.floor(self.resolution_meters / 0.5)
            sample_inds = torch.arange(0, raw_road_line.size(0), step_size)
            if (raw_road_line.size(0) - 1) % step_size != 0:
                sample_inds = torch.cat([sample_inds, torch.tensor([raw_road_line.size(0) - 1])], dim=0)
            sampled_road_line = raw_road_line[sample_inds]
            road_line_vectors = sampled_road_line[1:] - sampled_road_line[:-1]
            point_position.append(sampled_road_line[:-1, :self.dim])
            point_orientation.append(torch.atan2(road_line_vectors[:, 1], road_line_vectors[:, 0]))
            point_magnitude.append(torch.norm(road_line_vectors[:, :2], p=2, dim=-1))
            point_height.append(road_line_vectors[:, 2])
            point_type.append(torch.full((len(road_line_vectors),),
                                         self._point_types.index(self._road_line_type_dict[road_line.type]),
                                         dtype=torch.uint8))

        for road_edge_id, road_edge in road_edges.items():
            if road_edge_id not in road_edge_ids:
                continue
            raw_road_edge = torch.tensor([[point.x, point.y, point.z] for point in road_edge.polyline],
                                         dtype=torch.float)
            step_size = math.floor(self.resolution_meters / 0.5)
            sample_inds = torch.arange(0, raw_road_edge.size(0), step_size)
            if (raw_road_edge.size(0) - 1) % step_size != 0:
                sample_inds = torch.cat([sample_inds, torch.tensor([raw_road_edge.size(0) - 1])], dim=0)
            sampled_road_edge = raw_road_edge[sample_inds]
            road_edge_vectors = sampled_road_edge[1:] - sampled_road_edge[:-1]
            point_position.append(sampled_road_edge[:-1, :self.dim])
            point_orientation.append(torch.atan2(road_edge_vectors[:, 1], road_edge_vectors[:, 0]))
            point_magnitude.append(torch.norm(road_edge_vectors[:, :2], p=2, dim=-1))
            point_height.append(road_edge_vectors[:, 2])
            point_type.append(torch.full((len(road_edge_vectors),),
                                         self._point_types.index(self._road_edge_type_dict[road_edge.type]),
                                         dtype=torch.uint8))

        for crosswalk_id, crosswalk in crosswalks.items():
            if crosswalk_id not in crosswalk_ids:
                continue
            crosswalk_polygon = Polygon([[point.x, point.y, point.z] for point in crosswalk.polygon])
            crosswalk_polygon = orient(crosswalk_polygon, sign=1.0)
            raw_boundary = torch.tensor(list(crosswalk_polygon.exterior.coords), dtype=torch.float)
            boundary = []
            for point_idx in range(len(raw_boundary) - 1):
                num_boundary_points = math.ceil(torch.norm(raw_boundary[point_idx + 1] -
                                                           raw_boundary[point_idx], p=2, dim=-1).item() /
                                                self.resolution_meters) + 1
                boundary.append(
                    torch.from_numpy(interp_arc(int(num_boundary_points),
                                                raw_boundary[point_idx: point_idx + 2].numpy())[:-1]).float())
            boundary.append(raw_boundary[0].unsqueeze(0))
            boundary = torch.cat(boundary, dim=0)
            boundary_vectors = boundary[1:] - boundary[:-1]
            point_position.append(boundary[:-1, :self.dim])
            point_orientation.append(torch.atan2(boundary_vectors[:, 1], boundary_vectors[:, 0]))
            point_magnitude.append(torch.norm(boundary_vectors[:, :2], p=2, dim=-1))
            point_height.append(boundary_vectors[:, 2])
            point_type.append(torch.full((len(boundary_vectors),),
                                         self._point_types.index('CROSSWALK'),
                                         dtype=torch.uint8))

        for speed_bump_id, speed_bump in speed_bumps.items():
            if speed_bump_id not in speed_bump_ids:
                continue
            speed_bump_polygon = Polygon([[point.x, point.y, point.z] for point in speed_bump.polygon])
            speed_bump_polygon = orient(speed_bump_polygon, sign=1.0)
            raw_boundary = torch.tensor(list(speed_bump_polygon.exterior.coords), dtype=torch.float)
            boundary = []
            for point_idx in range(len(raw_boundary) - 1):
                num_boundary_points = math.ceil(torch.norm(raw_boundary[point_idx + 1] -
                                                           raw_boundary[point_idx], p=2, dim=-1).item() /
                                                self.resolution_meters) + 1
                boundary.append(
                    torch.from_numpy(interp_arc(int(num_boundary_points),
                                                raw_boundary[point_idx: point_idx + 2].numpy())[:-1]).float())
            boundary.append(raw_boundary[0].unsqueeze(0))
            boundary = torch.cat(boundary, dim=0)
            boundary_vectors = boundary[1:] - boundary[:-1]
            point_position.append(boundary[:-1, :self.dim])
            point_orientation.append(torch.atan2(boundary_vectors[:, 1], boundary_vectors[:, 0]))
            point_magnitude.append(torch.norm(boundary_vectors[:, :2], p=2, dim=-1))
            point_height.append(boundary_vectors[:, 2])
            point_type.append(torch.full((len(boundary_vectors),),
                                         self._point_types.index('SPEED_BUMP'),
                                         dtype=torch.uint8))

        for driveway_id, driveway in driveways.items():
            if driveway_id not in driveway_ids:
                continue
            driveway_polygon = Polygon([[point.x, point.y, point.z] for point in driveway.polygon])
            driveway_polygon = orient(driveway_polygon, sign=1.0)
            raw_boundary = torch.tensor(list(driveway_polygon.exterior.coords), dtype=torch.float)
            boundary = []
            for point_idx in range(len(raw_boundary) - 1):
                num_boundary_points = math.ceil(torch.norm(raw_boundary[point_idx + 1] -
                                                           raw_boundary[point_idx], p=2, dim=-1).item() /
                                                self.resolution_meters) + 1
                boundary.append(
                    torch.from_numpy(interp_arc(int(num_boundary_points),
                                                raw_boundary[point_idx: point_idx + 2].numpy())[:-1]).float())
            boundary.append(raw_boundary[0].unsqueeze(0))
            boundary = torch.cat(boundary, dim=0)
            boundary_vectors = boundary[1:] - boundary[:-1]
            point_position.append(boundary[:-1, :self.dim])
            point_orientation.append(torch.atan2(boundary_vectors[:, 1], boundary_vectors[:, 0]))
            point_magnitude.append(torch.norm(boundary_vectors[:, :2], p=2, dim=-1))
            point_height.append(boundary_vectors[:, 2])
            point_type.append(torch.full((len(boundary_vectors),),
                                         self._point_types.index('DRIVEWAY'),
                                         dtype=torch.uint8))

        num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)

        map_point = dict()
        if len(num_points) == 0:
            map_point['num_nodes'] = 0
            map_point['position'] = torch.tensor([], dtype=torch.float)
            map_point['orientation'] = torch.tensor([], dtype=torch.float)
            map_point['magnitude'] = torch.tensor([], dtype=torch.float)
            if self.dim == 3:
                map_point['height'] = torch.tensor([], dtype=torch.float)
            map_point['type'] = torch.tensor([], dtype=torch.uint8)
        else:
            map_point['num_nodes'] = num_points.sum().item()
            map_point['position'] = torch.cat(point_position, dim=0)
            map_point['orientation'] = torch.cat(point_orientation, dim=0)
            map_point['magnitude'] = torch.cat(point_magnitude, dim=0)
            if self.dim == 3:
                map_point['height'] = torch.cat(point_height, dim=0)
            map_point['type'] = torch.cat(point_type, dim=0)

        return map_point

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int) -> HeteroData:
        with open(self.processed_paths[idx], 'rb') as handle:
            return HeteroData(pickle.load(handle))

    def _download(self) -> None:
        # if complete raw/processed files exist, skip downloading
        if ((os.path.isdir(self.raw_dir) and len(self.raw_file_names) == self._num_raw_files) or
                (os.path.isdir(self.processed_dir) and len(self.processed_file_names) == len(self))):
            return
        if os.path.isdir(os.path.join(self.root, self.dir)):
            self._raw_file_names = [name for name in os.listdir(os.path.join(self.root, self.dir)) if
                                    os.path.isfile(os.path.join(self.root, self.dir, name))]
            if len(self.raw_file_names) == self._num_raw_files:
                if os.path.isdir(self.raw_dir):
                    shutil.rmtree(self.raw_dir)
                os.makedirs(self.raw_dir)
                for raw_file_name in self.raw_file_names:
                    shutil.move(os.path.join(self.root, self.dir, raw_file_name), self.raw_dir)
                return
            else:
                shutil.rmtree(os.path.join(self.root, self.dir))
                self._raw_file_names = []
        self._processed_file_names = []
        self.download()

    def _process(self) -> None:
        # if complete processed files exist, skip processing
        if os.path.isdir(self.processed_dir) and len(self.processed_file_names) == len(self):
            return
        print('Processing...', file=sys.stderr)
        if os.path.isdir(self.processed_dir):
            for name in os.listdir(self.processed_dir):
                if name.endswith(('pkl', 'pickle')):
                    os.remove(os.path.join(self.processed_dir, name))
        else:
            os.makedirs(self.processed_dir)
        self._processed_file_names = []
        self.process()
        print('Done!', file=sys.stderr)
