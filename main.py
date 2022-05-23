"""
This script evaluates Efficientnet on a dataset.
"""

from efficientnet_pytorch import EfficientNet
import torch
import torchmetrics

import numpy as np
import matplotlib.pyplot as plt

import constants
from utils import eval, dataset

# Define metrics

metrics = [
    torchmetrics.Accuracy(num_classes=constants.num_classes).to(constants.device),
    torchmetrics.Accuracy(num_classes=constants.num_classes, top_k=5).to(constants.device),
    torchmetrics.Accuracy(average="macro", num_classes=constants.num_classes).to(constants.device),
    ]
metrics_name = ["Accuracy", "Top 5 Accuracy", "Macro Accuracy"]

sep_acc_fn = torchmetrics.Accuracy(num_classes=constants.num_classes, average="none").to(constants.device)

cm_fn = torchmetrics.ConfusionMatrix(num_classes=constants.num_classes,
                                    normalize="true").to(constants.device)


# Load model et data

model = EfficientNet.from_pretrained('efficientnet-b0').to(constants.device)
dl = dataset.create_dataloader(batch_size=32,
                                shuffle=False,
                                pin_memory=False,
                                num_workers=4)

classes = dataset.load_json(constants.categories_file)

if __name__=="__main__":

    # Get preds
    y_true, logits = eval.get_preds(dl, model)

    print(f"Testing set contains {len(y_true)} valid samples.")

    # Print metrics
    for name, metric in zip(metrics_name, metrics):
        print(f"{name}: {metric(logits, y_true)}")

    # Separate accuracy

    sep_acc = sep_acc_fn(logits, y_true)
    best_acc, best_ind = torch.topk(sep_acc, k=5, largest=True)
    best_acc = best_acc.cpu()
    best_ind = best_ind.cpu()

    print("-"*20)
    print("Best accuracies:")
    for acc, ind in zip(best_acc, best_ind):
        # print(ind)
        # print(classes.iloc[ind, 0])
        print(f"Class {ind.item()} - {classes.iloc[ind.item(), 0]}: {acc}")

    worst_acc, worst_ind = torch.topk(sep_acc, k=5, largest=False)
    worst_acc = worst_acc.cpu()
    worst_ind = worst_ind.cpu()

    print("-"*20)
    print("Worst accuracies:")
    for acc, ind in zip(worst_acc, worst_ind):
        print(f"Class {ind.item()} - {classes.iloc[ind.item(), 0]}: {acc}")


    # Plot confusion matrix
    confusion_matrix = cm_fn(logits, y_true).cpu().numpy()
    plt.imshow(confusion_matrix, interpolation='none')
    plt.colorbar()
    plt.savefig("confusion_matrix")
    plt.show()




