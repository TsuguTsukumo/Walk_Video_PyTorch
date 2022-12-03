import torch 
import os 
import sys 
sys.path.append('/workspace/Walk_Video_PyTorch/project')

from models.pytorchvideo_models import WalkVideoClassificationLightningModule
from dataloader.data_loader import WalkDataModule

from pytorch_lightning import Trainer
from utils.utils import get_ckpt_path

from pytorch_lightning import loggers as pl_loggers 
from pytorch_lightning import seed_everything

seed_everything(42, workers=True)

from main import get_parameters

opt, _ = get_parameters()

opt.gpu_num = 1

opt.version = '1201_1_16'
opt.model = "resnet"
opt.model_depth = 50
opt.model_class_num = 1

opt.clip_duraion = 1
opt.uniform_temporal_subsample_num = 16
opt.version = opt.version + '_' + opt.model + '_depth' + str(opt.model_depth)

opt.train_path = '/workspace/data/split_pad_dataset_512/flod2'

opt.transfor_learning = True
opt.pre_process_flag = True


# for the tensorboard
tb_logger = pl_loggers.TensorBoardLogger(save_dir="/workspace/Walk_Video_PyTorch/project/tests/logs", name=opt.model, version=opt.version)

model = WalkVideoClassificationLightningModule(opt)

data_module = WalkDataModule(opt)

# get last ckpt path
# ckpt_path = get_ckpt_path(opt)

ckpt_path = '/workspace/Walk_Video_PyTorch/logs/resnet/1119_1_8_resnet_depth50/flod2/checkpoints/epoch=11-val_loss=0.63-val_acc=0.8926.ckpt'

trainer = Trainer(
    devices=[opt.gpu_num,],
    accelerator="gpu",
    logger=tb_logger,
)

list = trainer.test(model, dataloaders=data_module, ckpt_path=ckpt_path)
