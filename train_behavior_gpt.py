from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import WaymoSimDataModule
from simulators import BehaviorGPT

if __name__ == '__main__':
    pl.seed_everything(2024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=30)
    BehaviorGPT.add_model_specific_args(parser)
    args = parser.parse_args()

    model = BehaviorGPT(**vars(args))
    datamodule = WaymoSimDataModule(**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs)
    trainer.fit(model, datamodule)
