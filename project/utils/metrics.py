import torch
import torchmetrics

def get_Accuracy():
    accuracy = torchmetrics.Accuracy(
        num_classes=2,
        average='micro',
        # top_k=1
    )
    return accuracy

def get_Dice():
    dice = torchmetrics.Dice(
        num_classes=2,
        average='micro',

    )
    return dice

def get_Average_precision():
    average_precision = torchmetrics.AveragePrecision(
        num_classes=2, 
        pos_label=0,
    )

    return average_precision

def get_Precision_Recall():
    precision_recall = torchmetrics.PrecisionRecallCurve(
        num_classes=2, 
        pos_label=1
    )

    return precision_recall

def get_AUC():
    AUC = torchmetrics.AUC(
        reorder=True
    )
    return AUC

def get_F1Score():
    F1_Score = torchmetrics.F1Score(
        num_classes=2,

    )

    return F1_Score

# %%
if __name__ == '__main__':

    preds = torch.tensor([1, 0.8, 0.6, 0])
    target = torch.tensor([1, 1, 0, 0])

    dice = get_Dice()
    accuracy = get_Accuracy()
    average_precision = get_Average_precision()

    # result = accuracy(preds, target)
    result = average_precision(preds, target)
    print(result)
# %%
