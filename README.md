
<div align="center">    
 
# Walk Video Classification

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)   -->
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description 
📓 This project made with the PyTorch, PyTorch Lightning, PyTorch Video.

What it does   

## How to run

First, install dependencies   

``` bash
# clone project   
git clone https://github.com/ChenKaiXuSan/Walk_Video_PyTorch.git

# install project   
cd Walk_Video_PyTorch/ 
pip install -e .   
pip install -r requirements.txt
```

Next, navigate to any file and run it.

```bash
# module folder
cd Walk_Video_PyTorch/

# run module 
python project/main.py [option] > logs/output_log/xxx.log 
```

## Imports

This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

## Dataset

## Implementation

``` python

optional arguments:
  -h, --help            show this help message and exit
  --model {resnet,csn}
  --img_size IMG_SIZE
  --version VERSION     the version of logger, such data
  --model_class_num MODEL_CLASS_NUM
                        the class num of model
  --max_epochs MAX_EPOCHS
                        numer of epochs of training
  --batch_size BATCH_SIZE
                        batch size for the dataloader
  --num_workers NUM_WORKERS
                        dataloader for load video
  --clip_duration CLIP_DURATION
                        clip duration for the video
  --lr LR               learning rate for optimizer
  --beta1 BETA1
  --beta2 BETA2
  --pretrained_model PRETRAINED_MODEL
  --data_path DATA_PATH
                        meta dataset path
  --split_data_path SPLIT_DATA_PATH
                        split dataset path
  --log_path LOG_PATH   the lightning logs saved path

```

### Citation

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
