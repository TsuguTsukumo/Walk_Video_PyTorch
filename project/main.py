# %%
import pytorchvideo.models as models
import torch
import torch.nn as nn
import pytorchvideo
import torch.nn.functional as F

from pytorch_lightning import LightningModule
import pytorch_lightning
from zmq import device


# %%
class opt:
        _DATA_PATH = "/workspace/data/dataset/train"
        _CLIP_DURATION = 2 # Duration of sampled clip for each video
        _BATCH_SIZE = 16
        _NUM_WORKERS = 0  # Number of parallel processes fetching data
        model_type = 'resnet'
# %%
from models.pytorchvideo_models import WalkVideoClassificationLightningModule
from dataloader.data_loader import WalkDataModule
from pytorch_lightning import loggers as pl_loggers

def train():
    classification_module = WalkVideoClassificationLightningModule(opt.model_type)

    data_module = WalkDataModule(opt)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs', version=opt.model_type)
    trainer = pytorch_lightning.Trainer(accelerator="gpu", devices=2, max_epochs=100, logger=tb_logger, log_every_n_steps=10)
    trainer.fit(classification_module, data_module)

# %%
if __name__ == '__main__':
    
    train()


