# %%
from pytorchvideo.models import resnet, csn
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%

class MakeVideoModule(nn.Module):
    '''
    the module zoo from the PytorchVideo lib.

    Args:
        nn (_type_): 
    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model_class_num
        self.model_depth = hparams.model_depth

        self.transfor_learning = hparams.transfor_learning

        self.fix_layer = hparams.fix_layer

    def set_parameter_requires_grad(self, model: torch.nn.Module, flag:bool = True):

        for param in model.parameters():
            param.requires_grad = flag

    def make_walk_csn(self):
        
        return csn.create_csn(
            input_channel=3,
            model_depth=self.model_depth,
            model_num_class=self.model_class_num,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )

    def make_walk_resnet(self):
        
        # make model
        if self.transfor_learning:
            slow = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
            
            # change the knetics-400 output 400 to model class num
            slow.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

            model_stem = slow.blocks[0]
            model_head = slow.blocks[-1]
            model_stage = slow.blocks[1:-1]

            # ablation expermentional

            # first make sure all param to false 
            self.set_parameter_requires_grad(slow, False)

            if self.fix_layer == 'all':
                # train all param
                self.set_parameter_requires_grad(slow, True)
            elif self.fix_layer == 'head':
                # train model head, fix other
                self.set_parameter_requires_grad(model_head, True)
            elif self.fix_layer == 'stem_head':
                # train model head and stem, fix other
                self.set_parameter_requires_grad(model_head, True)
                self.set_parameter_requires_grad(model_stem, True)
            elif self.fix_layer == 'stage_head':
                # train model stage and head, fix stem
                self.set_parameter_requires_grad(model_stage, True)
                self.set_parameter_requires_grad(model_head, True)

        else:
            slow = resnet.create_resnet(
                input_channel=3,
                model_depth=self.model_depth,
                model_num_class=self.model_class_num,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )

        return slow

# # %%
# class opt: 
#     model_class_num = 2
#     model_depth = 50
#     transfor_learning = True
#     fix_layer = 'stage_head'


# make_video_module = MakeVideoModule(opt)

# model = make_video_module.make_walk_resnet()

# # %%
# for param in model.parameters():
#     print(param.requires_grad)
# # %%
