
# %%
from typing import Any, Callable, Dict, Optional, Type
from pytorch_lightning import LightningDataModule

import torch 
import pytorchvideo
from pytorchvideo.data.clip_sampling import ClipSampler

from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset, labeled_video_dataset


# %%
def WalkDataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = False,
    decoder: str = "pyav",
) -> LabeledVideoDataset:
    '''
    A helper function to create "LabeledVideoDataset" object for the Walk dataset.

    Args:
        data_path (str): Path to the data. The path defines how the data should be read. For a directory, the directory structure defines the classes (i.e. each subdirectory is class).
        clip_sampler (ClipSampler): Defines how clips should be sampled from each video. See the clip sampling documentation for more information.
        video_sampler (Type[torch.utils.data.Sampler], optional): Sampler for the internal video container. Defaults to torch.utils.data.RandomSampler.
        transform (Optional[Callable[[Dict[str, Any]], Dict[str, Any]]], optional): This callable is evaluated on the clip output before the clip is returned. Defaults to None.
        video_path_prefix (str, optional): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. Defaults to "".
        decode_audio (bool, optional): If True, also decode audio from video. Defaults to False. Notice that, if Ture will trigger the stack error.
        decoder (str, optional): Defines what type of decoder used to decode a video. Defaults to "pyav".

    Returns:
        LabeledVideoDataset: _description_
    '''
    return labeled_video_dataset(
        data_path,
        clip_sampler,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder
    )


# %%
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

# %%
class WalkDataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self._DATA_PATH = opt._DATA_PATH
        self._CLIP_DURATION = opt._CLIP_DURATION
        self._BATCH_SIZE = opt._BATCH_SIZE
        self._NUM_WORKERS = opt._NUM_WORKERS
    
    def train_dataloader(self):
        '''
        create the Walk train partition from the list of video labels 
        in directory and subdirectory. Add transform that subsamples and 
        normalizes the video before applying the scale, crop and flip augmentations.
        '''        
        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(244),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )

        train_dataset = WalkDataset(
            data_path=self._DATA_PATH,
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
            transform=train_transform
        )

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size = self._BATCH_SIZE,
            num_workers = self._NUM_WORKERS,
        )

    def val_dataloader(self):
        '''
        create the Walk train partition from the list of video labels 
        in directory and subdirectory. Add transform that subsamples and 
        normalizes the video before applying the scale, crop and flip augmentations.
        '''        
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(244),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )

        train_dataset = WalkDataset(
            data_path=self._DATA_PATH,
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
            transform=val_transform
        )

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size = self._BATCH_SIZE,
            num_workers = self._NUM_WORKERS,
        )


# %%
import matplotlib.pylab as plt 

if __name__ == '__main__':
    class opt:
        _DATA_PATH = "/workspace/data/dataset/train"
        _CLIP_DURATION = 2 # Duration of sampled clip for each video
        _BATCH_SIZE = 10
        _NUM_WORKERS = 0  # Number of parallel processes fetching data

    train_data_loader = WalkDataModule(opt).train_dataloader()
    val_data_loader = WalkDataModule(opt).val_dataloader()

    data = {"video": [], "class": [], 'tensorsize': []}

    batch = next(iter(train_data_loader))

    data["video"].append(batch['video_name'])
    data["class"].append(batch['label'])
    data["tensorsize"].append(batch['video'].size())

    print(data)

    location = 1

    plt.figure(figsize=(12, 12))

    for num in range(len(batch['video_index'])): # batch size
        for i in range(batch['video'].size()[2]): # 帧数
            plt.title(batch['video_name'][num])
            plt.subplot(len(batch['video_index']), batch['video'].size()[2], location)
            plt.imshow(batch["video"][num].permute(1, 2, 3, 0)[i])

            location += 1
            plt.axis("off")

    plt.show()