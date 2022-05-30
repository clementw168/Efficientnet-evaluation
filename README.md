# Efficientnet-evaluation

This repository aims to evaluate a pre-trained `Efficientnet` on a test set of 50,000 images.


## Dataset

The dataset is a balanced set of 50,000 images over 1,000 classes. Some images are in `gray scale` or in `RGBA` format. `gray scale` images are repeated to have 3 channels and `RGBA` images are converted to `RGB`.



## Results

| Metric | Value |
| :----- | :----- |
|  Accuracy       |  0.743 |
|  Top 5 Accuracy |  0.919 |
|  F1 Score       |  0.743 |
|  Precision      |  0.743 |
|  Recall         |  0.743 |
|  Specificity    | 0.9997 |




| Class ID | Class name | F1 score | Accuracy | Precision | Recall | Specificity |
| :----- | :-----  | :-----: | :-----: | :-----: | :-----: | :-----: |
| 986 | Yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum    |  1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
|  24 | Great grey owl, great gray owl, Strix nebulosa | 0.99 | 0.98 |  1.0 | 0.98 |  1.0 |
| 135 | Limpkin, Aramus pictus                         | 0.98 | 1.0  | 0.96 |  1.0 |  1.0 |
|. . . |
| 885 | Velvet                      | 0.24 | 0.16 | 0.47 | 0.16 |  1.0  |
| 681 | Notebook, notebook computer | 0.21 | 0.20 | 0.22 | 0.20 |  0.99 |
| 744 | Projectile, missile         | 0.20 | 0.14 | 0.33 | 0.14 |  1.0  |



|![Confusion matrix](https://github.com/clementw168/Efficientnet-evaluation/blob/main/confusion_matrix.png)|
|:----:|
| <b>Confusion matrix</b>|
