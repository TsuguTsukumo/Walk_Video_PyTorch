# %%
import os
import torch

import random
import shutil

# %%


def del_folder(path, *args):
    '''
    delete the folder which path/version

    Args:
        path (str): path
        version (str): version
    '''
    if os.path.exists(os.path.join(path, *args)):
        shutil.rmtree(os.path.join(path, *args))


def make_folder(path, *args):
    '''
    make folder which path/version

    Args:
        path (str): path
        version (str): version
    '''
    if not os.path.exists(os.path.join(path, *args)):
        os.makedirs(os.path.join(path, *args))


def tensor2var(x, grad=False):
    '''
    put tensor to gpu, and set grad to false

    Args:
        x (tensor): input tensor
        grad (bool, optional):  Defaults to False.

    Returns:
        tensor: tensor in gpu and set grad to false 
    '''
    if torch.cuda.is_available():
        x = x.cuda()
        x.requires_grad_(grad)
    return x


def var2tensor(x):
    '''
    put date to cpu

    Args:
        x (tensor): input tensor 

    Returns:
        tensor: put data to cpu
    '''
    return x.data.cpu()


def var2numpy(x):
    return x.data.cpu().numpy()


def to_Tensor(x, *arg):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.LongTensor
    return Tensor(x, *arg)


def random_split_video(fileDir: str, tarDir: str, rate: float = 0.8, version_flag: list = ("train", "val"), disease_flag: list = ("ASD", "LCS")):
    '''
    random split the meta flie video to the given rate. 
    Divided into the train and val folder, and the subfolder is the disease name.

    like this :
    train
        - ASD
        - LCS
    val 
        - ASD
        - LCS

    Args:
        fileDir (str): meta video data path.
        tarDir (str): tar video data path.
        rate (float, optional): the split rate for the train video. Defaults to 0.8.
        version_flag (list, optional): a list have "train" and "val" tag. Defaults to ("train", "val").
        disease_flag (list, optional): a list have "ASD" and "LVS" disease tag. Defaults to ("ASD", "LCS").
    '''   
    # save the splited video info for print
    split_video_info = {}

    for disease_flag in disease_flag:
        # meta video file path
        fileDir_flag = os.path.join(fileDir, disease_flag)

        for flag in version_flag:

            # check folder
            del_folder(tarDir, flag, disease_flag)
            make_folder(tarDir, flag, disease_flag)

            pathDir = os.listdir(fileDir_flag)  # 取图片的原始路径
            filenumber = len(pathDir)

            if flag == "train":

                split_rate = rate  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
                picknumber = int(filenumber * split_rate)  # 按照rate比例从文件夹中取一定数量图片
                sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片

                for name in sample:
                    shutil.copy(os.path.join(fileDir_flag, name), os.path.join(tarDir, flag, disease_flag))

                split_video_info[disease_flag + "_" + flag] = len(sample)

            if flag == "val":
                leave_sample = []  # 计算剩下的视频
                for item in pathDir:
                    if item not in sample:
                        leave_sample.append(item)

                for name in leave_sample:
                    shutil.copy(os.path.join(fileDir_flag, name), os.path.join(tarDir, flag, disease_flag))

                split_video_info[disease_flag + "_" + flag] = len(leave_sample)

                # print(filenumber)

        print("Total " + disease_flag + ' Number\t' + str(filenumber))
    print(split_video_info)