# %% 
from base64 import encode
from inspect import Parameter
from pytest import param
import torch
import numpy as np

import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from torchvision.transforms.functional import crop, pad, resize

# %%

class batch_detection():
    def __init__(self, img_size) -> None:

        # set for detection
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(cfg)

        self.img_size = img_size

    def get_person_bboxes(self, inp_img, predictor):

        predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
        predicted_boxes = boxes[np.logical_and(classes==0, scores>0.9 )].tensor.cpu() # only person
        return predicted_boxes, predictions

    def get_frame_box(self, inp_imgs:list):
        '''
        get the predict bbox from the inp_imgs

        Args:
            inp_imgs (list): (c, t, h, w)

        Returns:
            list: frame_list, box_list, pred_list
        '''

        frame_list = []
        box_list = []
        pred_list = []

        channel, frames, h, w = inp_imgs.size()
        
        # 1 batch different frame
        for frame in range(frames):

            inp_img = inp_imgs[:, frame, :, :]
            inp_img = inp_img.permute(1, 2, 0) # (c, h, w) to (h, w, c)
            frame_list.append(inp_img)

            predicted_boxes, pred = self.get_person_bboxes(inp_img, self.predictor)
            pred_list.append(pred)
            box_list.append(predicted_boxes)

        print('clip %i frames!' % frames)

        return frame_list, box_list, pred_list

    def clip_pad_with_bbox(self, imgs: list, boxes: list, img_size: int = 256, bias:int = 3):
        '''
        based torchvision function to crop, pad, resize img.

        clip with the bbox, (x1-bias, y1) and padd with the (gap-bais) in left and right.

        Args:
            imgs (list): imgs with (c, t, h, w)
            boxes (list): (x1, y1, x2, y2)
            img_size (int, optional): croped img size. Defaults to 256.
            bias (int, optional): the bias of bbox, with the (x1-bias) and (x2+bias). Defaults to 5.

        Returns:
            tensor: (b, c, t, h, w)
        ''' 

        frame_list = []

        for num in range(len(imgs)):

            x1, y1, x2, y2 = boxes[num].int().squeeze() # dtype must int for resize, crop function

            box_width = x2 - x1
            box_height = y2 - y1 

            width_gap = ((box_height - box_width) / 2).int() # keep int type

            img = imgs[num].permute(2, 0, 1) # (h, w, c) to (c, h, w), for pytorch function

            # give a bias for the left and right crop bbox.
            croped_img = crop(img, top=y1, left=(x1 - bias), height=box_height, width=(box_width + 2 * bias))

            pad_img = pad(croped_img, padding=(width_gap - bias, 0), fill=0)

            resized_img = resize(pad_img, size=(img_size, img_size))

            frame_list.append(resized_img)

        return torch.stack(frame_list, dim=1)

    def handel_batch_imgs(self, batchs:torch.tensor):

        batch_num, channels, frames, h, w = batchs.size()

        batch_list = []

        for batch in range(batch_num):

            # start 1 batch 
            frame_list, box_list, pred_list = self.get_frame_box(batchs[batch])

            one_batch = self.clip_pad_with_bbox(frame_list, box_list, self.img_size)

            batch_list.append(one_batch)

        return torch.stack(batch_list, dim=0)

# %%
# %cd ..
from dataloader.data_loader import WalkDataModule
from parameters import get_parameters

# %%

parames, unkonwn = get_parameters()

parames.uniform_temporal_subsample_num = 30
parames.clip_duration = 3

data_modeule = WalkDataModule(parames)
data_modeule.setup()
train_dataloader = data_modeule.train_dataloader()

batch = next(iter(train_dataloader))

video = batch['video']
label = batch['label']

# %% 
get_bbox = batch_detection(img_size=parames.img_size)

frame_list, box_list, predict_list = get_bbox.get_frame_box(video[0])

# %% 
from matplotlib import pyplot as plt

plt.figure(figsize=(256, 256))

for frame in range(len(frame_list)):
    plt.subplot(len(frame_list), 1, frame + 1)

    plt.imshow(frame_list[frame] / 255)

# %% 
clip_pad_imgs = get_bbox.handel_batch_imgs(video)

b, c, t, h, w = clip_pad_imgs.shape

plt.figure(figsize=(256, 256))

for frame in range(t):
    plt.subplot(t, 1, frame + 1)

    plt.imshow(clip_pad_imgs[0].permute(1, 2, 3, 0)[frame] / 256 )

# %%
from torchvision.io import write_video

write_video('test.mp4', video_array=clip_pad_imgs[0].permute(1, 2, 3, 0), fps=30, video_codec="h264")
# %%
