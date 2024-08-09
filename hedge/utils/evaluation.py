from sklearn import metrics
from torchmetrics import AveragePrecision,Accuracy,AUROC,F1Score,Precision,Recall

import torch

"""
https://lightning.ai/docs/torchmetrics/stable/all-metrics.html
"""
class Evaluation:
    def __init__(self, task = "binary", num_classes = 2) -> None:
        self.task = task
        
        # https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html#
        self.accuracy = Accuracy(task=task, num_classes=num_classes)
        
        
        # https://lightning.ai/docs/torchmetrics/stable/classification/average_precision.html#
        self.average_precision = AveragePrecision(task=task, num_classes=num_classes)
        
        # https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html#
        self.auroc = AUROC(task=task, num_classes=num_classes)
        
        # https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html
        self.f1 = F1Score(num_classes=num_classes, average='macro',task=task)
        
        # Precision=  (True Positives) / （True Positives+False Positives)

        # https://lightning.ai/docs/torchmetrics/stable/classification/precision.html
        self.precision = Precision(num_classes=num_classes, average='macro',task=task)
        
        # Recall= (True Positives)/（True Positives+False Negative )
        
        # https://lightning.ai/docs/torchmetrics/stable/classification/recall.html
        self.recall = Recall(num_classes=num_classes, average='macro',task=task)
        
    def __call__(self, preds, target):
        
        if type(preds) is not torch.Tensor:    
            raise TypeError("preds must be torch.Tensor. but got {}".format(type(preds)))
        
        if type(target) is not torch.Tensor:
            raise TypeError("target must be torch.Tensor. but got {}".format(type(target)))

        preds = preds.cpu()
        target = target.cpu()


        preds = preds - preds.mean() + 0.5
        accuracy = self.accuracy(preds, target)
        precision = self.precision(preds, target)
        recall = self.recall(preds, target)
        f1 = self.f1(preds, target)
        auroc = self.auroc(preds, target)
        
        result = {
            "accuracy": round(accuracy.item(), 4),
            "f1": round(f1.item(), 4),
            "auroc": round(auroc.item(), 4),
            "precision": round(precision.item(), 4),
            "recall": round(recall.item(), 4)
        }
    
        return result


# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# def evaluation(labels, predictions):
#     accuracy = accuracy_score(labels, predictions)
#     precision = precision_score(labels, predictions,average='micro')
#     recall = recall_score(labels, predictions,average='macro')
#     f1 = f1_score(labels, predictions,average='micro')
    
#     return accuracy, f1, precision, recall,
#     # return accuracy, accuracy, accuracy, recall
