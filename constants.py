import os
import torch

import config

image_folder = os.path.join(config.data_folder, "dataset")
label_file = os.path.join(config.data_folder, "labels.json")
categories_file = os.path.join(config.data_folder, "categories.json")

num_classes = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"
