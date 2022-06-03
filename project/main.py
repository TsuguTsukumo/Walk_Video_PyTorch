# %%
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from dataloader.data_loader import WalkDataModule
from models.pytorchvideo_models import WalkVideoClassificationLightningModule
from argparse import ArgumentParser
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning
from yaml import parse

# %%


def get_parameters():

    parser = ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'csn'])
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--version', type=str, default='test', help='the version of logger, such data')
    parser.add_argument('--model_class_num', type=int, default=2, help='the class num of model')

    # Training setting
    parser.add_argument('--max_epochs', type=int, default=100, help='numer of epochs of training')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for the dataloader')
    parser.add_argument('--num_workers', type=int, default=2, help='dataloader for load video')
    parser.add_argument('--clip_duration', type=int, default=2, help='clip duration for the video')

    # TTUR
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for optimizer')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Path
    parser.add_argument('--data_path', type=str, default="/workspace/data/handle_video/", help='meta dataset path')
    parser.add_argument('--split_data_path', type=str, default="/workspace/data/dataset/", help="split dataset path")

    parser.add_argument('--log_path', type=str, default='./logs', help='the lightning logs saved path')

    return parser.parse_args()


# %%

def train(hparams):

    # fixme will occure bug, with deterministic = true
    # seed_everything(42, workers=True)

    classification_module = WalkVideoClassificationLightningModule(hparams)

    # instance the data module
    data_module = WalkDataModule(hparams)
    # data_module.setup()

    # for the tensorboard
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=hparams.log_path, name=hparams.model, version=hparams.version)

    trainer = Trainer(accelerator="auto",
                      devices=1, 
                      max_epochs=100,
                      logger=tb_logger,
                      log_every_n_steps=10,
                    #   deterministic=True 
                      )

    # training and val
    trainer.fit(classification_module, data_module)

    # testing
    trainer.test(dataloaders=data_module)


# %%
if __name__ == '__main__':

    config = get_parameters()

    train(config)
