"""
This script plot the distribution of the dataset.
"""

import torch
import torchmetrics

import numpy as np
import matplotlib.pyplot as plt

import constants
from utils import dataset


classes = dataset.load_json(constants.categories_file)

labels = dataset.load_json(constants.label_file).iloc[:, 0].values

if __name__=="__main__":
    print(len(labels))
    plt.hist(labels, bins=range(constants.num_classes))
    plt.show()

