# %%
import torch 
import torch.nn.functional as F
import os 
import sys

sys.path.append('/workspace/Walk_Video_PyTorch/project')

from models.pytorchvideo_models import WalkVideoClassificationLightningModule
from dataloader.data_loader import WalkDataModule
from pytorchvideo.transforms.functional import uniform_temporal_subsample

from IPython.display import clear_output

clear_output()

from pytorch_lightning import seed_everything

seed_everything(42, workers=True)

import torchmetrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
# define the metrics.
_accuracy = torchmetrics.classification.BinaryAccuracy()
_precision = torchmetrics.classification.BinaryPrecision()
_binary_recall = torchmetrics.classification.BinaryRecall()
_binary_f1 = torchmetrics.classification.BinaryF1Score()

_confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix()

_aucroc = torchmetrics.classification.BinaryAUROC()

# %%
class opt:
    num_workers = 8
    batch_size = 16

    fusion_method = 'slow_fusion'
    fix_layer = 'all'

    model = "resnet"

    img_size = 224
    lr=0.0001
    model_class_num = 1
    model_depth = 50

    transfor_learning = True
    pre_process_flag = True

DATA_PATH = "/workspace/data/segmentation_dataset_512"
# DATA_PATH = "/workspace/data/split_pad_dataset_512"

# %%
def get_best_ckpt(length: str, frame: str, fold: str):

    ckpt_path = '/workspace/Walk_Video_PyTorch/logs/resnet/'
    version = '54_1_8_resnet'
    ckpt_path_list = os.listdir(ckpt_path)
    ckpt_path_list.sort()

    final_ckpt_list = [] 

    for i in ckpt_path_list:
        if version in i:
            final_ckpt_list.append(i)

    final_ckpt_list.sort()
    
    for name in final_ckpt_list:
        if length in name.split('_') and frame in name.split('_'):
            ckpt = name

    ckpt = os.path.join(ckpt_path, ckpt, fold, 'checkpoints')
    
    Acc = 0.0

    ckpt_list = os.listdir(ckpt)
    ckpt_list.sort()
    ckpt_list.remove('last.ckpt')

    for num, l in enumerate(ckpt_list):
        acc = l[-11:-5] # accuracy

        if float(acc) > float(Acc):
            Acc = acc
            NUM = num

    return os.path.join(ckpt, ckpt_list[NUM])

# ckpt_path = get_best_ckpt('1', '31', 'fold0')

# %%
def get_inference(test_data, model):
        
    pred_list = []
    label_list = []

    for i, batch in enumerate(test_data):

        # input and label
        video = batch['video'].detach().cuda() # b, c, t, h, w

        label = batch['label'].detach().cuda() # b, class_num

        model.cuda().eval()

        # pred the video frames
        with torch.no_grad():
            preds = model(video)

        # when torch.size([1]), not squeeze.
        if preds.size()[0] != 1 or len(preds.size()) != 1 :
            preds = preds.squeeze(dim=-1)
            preds_sigmoid = torch.sigmoid(preds)
        else:
            preds_sigmoid = torch.sigmoid(preds)

        pred_list.append(preds_sigmoid.tolist())
        label_list.append(label.tolist())

        total_pred_list = []
        total_label_list = []

        for i in pred_list:
            for number in i:
                total_pred_list.append(number)

        for i in label_list:
            for number in i: 
                total_label_list.append(number)

    pred = torch.tensor(total_pred_list)
    label = torch.tensor(total_label_list)

    return total_pred_list, total_label_list

# %%
VIDEO_LENGTH = ['1']
VIDEO_FRAME = ['8']

# %%
fold_num = os.listdir(DATA_PATH)
fold_num.sort()
fold_num.remove('raw')

symbol = '_'

one_condition_pred_list = []
one_condition_label_list = []

total_pred_list = []
total_label_list = []

for length in VIDEO_LENGTH:

    for frame in VIDEO_FRAME:

        for fold in fold_num:

            opt.train_path = os.path.join(DATA_PATH, fold)

            #################
            # start k Fold CV
            #################
            
            opt.clip_duration = int(length)
            opt.uniform_temporal_subsample_num = int(frame)
           
            ckpt_path = get_best_ckpt(length, frame, fold)

            print('#' * 50)
            print('Strat %s, %s length, %s frames' % (fold, length, frame))
            print('the data path: %s' % opt.train_path)
            print('ckpt: %s' % ckpt_path)
            model = WalkVideoClassificationLightningModule(opt).load_from_checkpoint(ckpt_path)

            data_module = WalkDataModule(opt)
            data_module.setup()
            test_data = data_module.test_dataloader()

            pred_list, label_list = get_inference(test_data, model)

            one_condition_pred_list.append(pred_list)

            one_condition_label_list.append(label_list)

        # total 5 fold pred and label
        for i in one_condition_pred_list:
            for number in i:
                total_pred_list.append(number)

        for i in one_condition_label_list:
            for number in i: 
                total_label_list.append(number)

        pred = torch.tensor(total_pred_list)    
        label = torch.tensor(total_label_list)

        print('*' * 100)
        print('the result of %ss %sframe:' % (length, frame))
        print('accuracy: %s' % _accuracy(pred, label))
        print('precision: %s' % _precision(pred, label))
        print('_binary_recall: %s' % _binary_recall(pred, label))
        print('_binary_f1: %s' % _binary_f1(pred, label))
        print('_aurroc: %s' % _aucroc(pred, label))
        print('_confusion_matrix: %s' % _confusion_matrix(pred, label))
        print('#' * 50)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

cm = _confusion_matrix(pred, label).to(float)

total = sum(sum(cm))

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        cm[i][j] = float(cm[i][j] / total)

ax = sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=['ASD', 'non_ASD'], yticklabels=['ASD', 'non_ASD'])

ax.set_title('Confusion Matrix')
ax.set(xlabel="pred class", ylabel="ground truth")
ax.xaxis.tick_top()
plt.show()

# %%
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay


fpr, tpr, _ = roc_curve(label, pred, pos_label=1)
roc_auc = auc(fpr, tpr)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="proposed method").plot()
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUROC curves")
plt.legend()
plt.show()
# %%
