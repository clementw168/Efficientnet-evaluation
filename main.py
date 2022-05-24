"""
This script evaluates Efficientnet on a dataset.
"""

from efficientnet_pytorch import EfficientNet
import torch
import torchmetrics

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import constants
from utils import eval, dataset

# Define metrics

global_metrics = [
    torchmetrics.Accuracy(num_classes=constants.num_classes).to(constants.device),
    torchmetrics.Accuracy(num_classes=constants.num_classes, top_k=5).to(constants.device),
    torchmetrics.F1Score(num_classes=constants.num_classes).to(constants.device),
    torchmetrics.Precision(num_classes=constants.num_classes).to(constants.device),
    torchmetrics.Recall(num_classes=constants.num_classes).to(constants.device),
    torchmetrics.Specificity(num_classes=constants.num_classes).to(constants.device),
    ]

global_metrics_name = ["Accuracy", "Top 5 Accuracy", "F1 score", "Precision", "Recall", "Specificity"]

sep_metrics = [
    torchmetrics.F1Score(num_classes=constants.num_classes, average="none").to(constants.device),
    torchmetrics.Accuracy(num_classes=constants.num_classes, average="none").to(constants.device),
    torchmetrics.Precision(num_classes=constants.num_classes, average="none").to(constants.device),
    torchmetrics.Recall(num_classes=constants.num_classes, average="none").to(constants.device),
    torchmetrics.Specificity(num_classes=constants.num_classes, average="none").to(constants.device),
    ]
sep_metrics_name = ["F1 score", "Accuracy", "Precision", "Recall", "Specificity"]

cm_fn = torchmetrics.ConfusionMatrix(num_classes=constants.num_classes,
                                    normalize="true").to(constants.device)


# Load model et data

model = EfficientNet.from_pretrained('efficientnet-b0').to(constants.device)
dl = dataset.create_dataloader(batch_size=32,
                                shuffle=False,
                                pin_memory=False,
                                num_workers=4,
                                drop_non_valid=False)

classes = dataset.load_json(constants.categories_file)

if __name__=="__main__":

    ## Get preds
    y_true, logits = eval.get_preds(dl, model)

    print(f"Testing set contains {len(y_true)} valid samples.")

    ## Print global metrics
    for name, metric in zip(global_metrics_name, global_metrics):
        print(f"{name}: {metric(logits, y_true)}")


    ## Print best and worst samples metrics according to F1 score
    sep_metrics_values = []
    for metric in sep_metrics:
        sep_metrics_values.append(metric(logits, y_true).cpu().numpy())

    df = {
        "Class": classes.loc[:, 0].values,
        **dict(zip(sep_metrics_name, sep_metrics_values))
        }
    df = pd.DataFrame(df)

    df.to_csv("class_scores.csv")
    print(df.sort_values("F1 score", ascending=False))


    # Plot confusion matrix
    confusion_matrix = cm_fn(logits, y_true).cpu().numpy()
    plt.imshow(confusion_matrix, interpolation='none')
    plt.colorbar()
    plt.savefig("confusion_matrix")
    plt.show()




