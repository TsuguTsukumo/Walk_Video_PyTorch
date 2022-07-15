import torchmetrics

def get_Accuracy():
    accuracy = torchmetrics.Accuracy(
        num_classes=2,
        average='micro',
        # top_k=1
    )
    return accuracy.cuda()
