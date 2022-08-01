# %% 
%cd ..

from parameters import get_parameters

from batch_detection import batch_detection

# %%
import torch
import torchvision
from torchvision.io import read_video, read_video_timestamps, write_video
import pytorchvideo

import os 

from utils.utils import make_folder, del_folder, count_File_Number
# %%
parames, unkonwn = get_parameters()

# %%
DATA_PATH = parames.data_path
SPLIT_PAD_DATA_PATH = parames.split_pad_data_path

def make_split_pad_folder(split_pad_data_path:str):
    for flag in ('train', 'val'):

        data_path_flag = os.path.join(split_pad_data_path, flag)

        for diease_flag in ('ASD', 'ASD_not'):

            final_split_pad_data_path = os.path.join(data_path_flag, diease_flag)

            make_folder(final_split_pad_data_path)

    print('success make split pad folder!')

make_split_pad_folder(SPLIT_PAD_DATA_PATH)

# %%
final_video_path = {}

diease_path = []

def get_Diease_Path(data_path:str):

    for flag in ('train', 'val'):

        data_path_flag = os.path.join(data_path, flag) 

        for diease_flag in  (os.listdir(data_path_flag)): # ASD, ASD_not
            
            data_path_diease_flag =  os.path.join(data_path_flag, diease_flag)

            diease_path.append(data_path_diease_flag)

    return diease_path 

# %%
prefix_path = get_Diease_Path(DATA_PATH) # four folder, train/ASD, train/ASD_not, val/ASD, val/ASD_not

# %%
for data_path_flag_diease in prefix_path:

    compare_video_name_list = []

    # all video path
    video_path_list = os.listdir(data_path_flag_diease)

    # start compare video path list
    for compare_video_name in video_path_list:

        compare_video_name_list.append(compare_video_name[:15])

    compare_video_file_name =  list(set(compare_video_name_list))

    for name in compare_video_file_name:

        video_same_path_list = []

        for video_path_name in video_path_list:

            now_video_path_name = video_path_name[:15]

            if now_video_path_name in name:

                video_same_path_list.append(os.path.join(data_path_flag_diease, video_path_name))

        video_same_path_list.sort()
            
        final_video_path[os.path.join(data_path_flag_diease, name[:15])] = video_same_path_list
                
# %% 
def read_video_from_path(path:list, img_size:int = 256):

    get_bbox = batch_detection(img_size=img_size)

    for num in range(len(path)):

        video_frame, audio_frame, video_info = read_video(path[num]) # (t, h, w, c)
        clip_pad_imgs = get_bbox.handel_batch_imgs(video_frame.permute(3, 0, 1, 2)) # (c, t, h, w)

        after_path = path[num][24:]
        write_video(filename=os.path.join(SPLIT_PAD_DATA_PATH, after_path), video_array=clip_pad_imgs.permute(1, 2, 3, 0), fps=30, video_codec='h264') # (c, t, h, w) to (t, h, w, c)

        print(video_info['video_fps'], path[num], video_frame.shape)
        print('clip %i frames!, %i frams lost!' % (clip_pad_imgs.size()[1], video_frame.size()[0] - clip_pad_imgs.size()[1]))

    return video_frame, path[num], video_info['video_fps']

# %% 
for key in final_video_path.keys():
    
    now_video_path_list = final_video_path[key]
    now_video_path_list.sort()

    video_frame, now_video_path_list, video_info = read_video_from_path(now_video_path_list)

# %%

count_File_Number(SPLIT_PAD_DATA_PATH)

# %%
# merge differenct videos into one video.

splt_video_path = "/workspace/Walk_Video_PyTorch/project/prepare_video/test"

splt_video_path_list = os.listdir(splt_video_path)
#%%
video_list = []

for path in splt_video_path_list:
    video, audio, video_info = read_video(os.path.join(splt_video_path, path))
    video_list.append(video)

video_file = torch.cat(video_list, dim=(0))

# %%

write_video(filename=os.path.join(splt_video_path, 'full.mp4'), fps=30, video_codec='h264', video_array=video_file)

