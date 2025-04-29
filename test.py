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
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.loader import DataLoader

from datasets import WaymoSimDataset
from simulators import BehaviorGPT

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    model = {
        'BehaviorGPT': BehaviorGPT,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)
    test_dataset = WaymoSimDataset(root=args.root, split='test', interactive=args.interactive)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy=DDPStrategy(process_group_backend='gloo'))
    trainer.test(model, dataloader)
