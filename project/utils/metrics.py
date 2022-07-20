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

# %%
if __name__ == '__main__':

    preds = torch.tensor([1, 0, 1, 0])
    target = torch.tensor([1, 1, 0, 0])

    dice = get_Dice()
    accuracy = get_Accuracy()

    result = accuracy(preds, target)
    print(result)
# %%
