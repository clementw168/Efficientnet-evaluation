import os
import json
from PIL import Image

import pandas as pd
import torch
import torchvision

import constants

def load_json(file):
    with open(file) as file:
        df = json.load(file)
        df = pd.DataFrame.from_dict(df, orient="index")
    return df

class Dataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, drop_non_valid=False):
        self.img_labels = load_json(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.drop_non_valid = drop_non_valid

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.index.values[idx])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 0]
        if image.mode != "RGB": # Filter if there is non rgb images
            if self.drop_non_valid:
                return None
            else:
                image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

def filtering_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def create_transform():
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    return transform

def create_dataloader(batch_size=16, shuffle=False, pin_memory=False, num_workers=4, drop_non_valid=False):
    transform = create_transform()
    dataset = Dataset(annotation_file=constants.label_file,
                        img_dir=constants.image_folder,
                        transform=transform,
                        drop_non_valid=drop_non_valid)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            collate_fn=filtering_collate,
                                            pin_memory=pin_memory,
                                            num_workers=num_workers)
    return dataloader



if __name__=="__main__":

    ### test dataloader

    dataloader = create_dataloader(batch_size=32,
                        shuffle=False,
                        pin_memory=False,
                        num_workers=4,
                        drop_non_valid=False)
    print(len(dataloader.dataset))

    for X, y in dataloader:
        print(X.shape)
        print(y)

    ### test get_classes

    # ind = 1
    # classes = load_json(constants.categories_file)
    # print(classes.iloc[ind, 0])
