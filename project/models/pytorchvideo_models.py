# %%
from pytorchvideo.models import x3d, resnet, csn, slowfast, r2plus1d
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from utils.metrics import get_AUC, get_Accuracy, get_Average_precision, get_Dice, get_F1Score, get_Precision_Recall

# %%
class MakeVideoModule(nn.Module):
    '''
    the module zoo from the PytorchVideo lib.

    Args:
        nn (_type_): 
    '''
    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num=hparams.model_class_num
        self.model_depth=hparams.model_depth

    def make_walk_csn(self):
        return csn.create_csn(
            input_channel=3,
            model_depth=self.model_depth,
            model_num_class=self.model_class_num,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )

    def make_walk_resnet(self):
        return resnet.create_resnet(
        input_channel=3,
        model_depth=self.model_depth,
        model_num_class=self.model_class_num,
        norm=nn.BatchNorm3d,
        # activation=nn.Sigmoid, # todo try sigmoid
        activation = nn.ReLU,
        )

    def make_walk_x3d(self) -> nn.Module:
        return x3d.create_x3d(
            input_channel=3,
            model_num_class=self.model_class_num,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
            input_clip_length=4,
        )

    #todo question with input tensor
    def make_walk_slowfast(self) -> nn.Module:
        return slowfast.create_slowfast(

        )
    
    # todo question with input tensor
    def make_walk_r2plus1d(self) -> nn.Module:
        return r2plus1d.create_r2plus1d(
            input_channel=3,
            model_num_class=self.model_class_num,
            model_depth=self.model_depth,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )
# %%

class WalkVideoClassificationLightningModule(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type=hparams.model
        self.img_size = hparams.img_size

        self.lr=hparams.lr

        self.model = MakeVideoModule(hparams)

        # select the network structure 
        if self.model_type == 'resnet':
            self.model=self.model.make_walk_resnet()

        elif self.model_type == 'csn':
            self.model=self.model.make_walk_csn()

        elif self.model_type == 'x3d':
            self.model = self.model.make_walk_x3d()

        # select the metrics
        self.accuracy = get_Accuracy()
        self.dice = get_Dice()
        self.average_precision = get_Average_precision()
        self.AUC = get_AUC()
        self.f1_score = get_F1Score()
        self.precision_recall = get_Precision_Recall()

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        '''
        train steop when trainer.fit called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss
        '''

        label = batch['label'].detach()
        # classification task
        y_hat=self.model(batch["video"])

        y_hat_softmax = torch.softmax(y_hat, dim=-1)

        loss=F.cross_entropy(y_hat, label)

        accuracy = self.accuracy(y_hat_softmax, label)

        self.log('train_loss', loss)
        self.log('train_acc', accuracy)

        return loss

    def training_epoch_end(self, outputs) -> None:
        '''
        after validattion_step end.

        Args:
            outputs (list): a list of the train_step return value.
        '''
        
        # log epoch metric
        # self.log('train_acc_epoch', self.accuracy)
        pass

    def validation_step(self, batch, batch_idx):
        '''
        val step when trainer.fit called.

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss 
            accuract: selected accuracy result.
        '''

        label = batch['label'].detach()

        preds = self.model(batch["video"])

        preds_softmax = torch.softmax(preds, dim=-1)

        val_loss=F.cross_entropy(preds, label)

        # calc the metric, function from torchmetrics
        accuracy = self.accuracy(preds_softmax, label)

        average_precision = self.average_precision(preds_softmax, label)

        # log the val loss and val acc, in step and in epoch.
        self.log_dict({'val_loss': val_loss, 'val_acc': accuracy, 'val_average_precision': average_precision}, on_step=False, on_epoch=True)
        
        return accuracy

    def validation_epoch_end(self, outputs):
        
        val_metric = torch.stack(outputs, dim=0)

        final_acc = torch.sum(val_metric) / len(val_metric)

        print(final_acc)

        return final_acc

    def test_step(self, batch, batch_idx):
        '''
        test step when trainer.test called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_
        '''

        labels = batch['label'].detach()

        preds = self.model(batch["video"])

        preds_softmax = torch.softmax(preds, dim=-1)

        test_loss = F.cross_entropy(preds, labels)

        # calculate acc 
        accuracy = self.accuracy(preds_softmax, labels)

        average_precision = self.average_precision(preds_softmax, labels)
        # AUC = self.AUC(F.softmax(test_pred, dim=-1), batch["label"])
        f1_score = self.f1_score(preds_softmax, labels)
        # precision, recall, threshold = self.precision_recall(F.softmax(test_pred, dim=-1), batch["label"])

        # self.dice(test_pred, target)

        # log the test loss, and test acc, in step and in epoch
        self.log_dict({'test_loss': test_loss, 'test_acc': self.accuracy, 'test_average_precision': self.average_precision, 'test_f1_score': self.f1_score}, on_step=False, on_epoch=True)

        return accuracy
        
    def test_epoch_end(self, outputs):

        test_metric = torch.stack(outputs, dim=0)

        final_acc = torch.sum(test_metric) / len(test_metric)
        
        print(final_acc)

    def configure_optimizers(self):
        '''
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        '''

        return torch.optim.Adam(self.parameters(), lr=self.lr)
        # return torch.optim.SGD(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type