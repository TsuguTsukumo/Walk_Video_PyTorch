# %%
from torchinfo import summary
import string
import pytorchvideo.models as models
import torch
import torch.nn as nn
import pytorchvideo
import torch.nn.functional as F

from pytorch_lightning import LightningModule
import pytorch_lightning
import os

# %%
# %cd /workspace/Walk_Video_PyTorch/project/

# %%


def make_walk_resnet(model_class_num: int = 2, model_depth:int = 50):
    return models.resnet.create_resnet(
        input_channel=3,
        model_depth=model_depth,
        model_num_class=model_class_num,
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )

# %%


def make_walk_csn(model_class_num: int = 2, model_depth:int=50):
    return models.create_csn(
        input_channel=3,
        model_depth=model_depth,
        model_num_class=model_class_num,
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )
# %%

class WalkVideoClassificationLightningModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type = hparams.model

        self.lr = hparams.lr
        self.model_class_num = hparams.model_class_num
        self.model_depth = hparams.model_depth

        if self.model_type == 'resnet':
            self.model = make_walk_resnet(self.model_class_num, self.model_depth)
        elif self.model_type == 'csn':
            self.model = make_walk_csn(self.model_class_num, self.model_depth)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])

        loss = F.cross_entropy(y_hat, batch["label"])

        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type


# %%

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    classification_module = WalkVideoClassificationLightningModule(model_type='resnet')

    batch_size = 16
    summary = summary(classification_module, input_size=(batch_size, 3, 8, 255, 255))
    print(summary)
    print(classification_module._get_name())


# %%
