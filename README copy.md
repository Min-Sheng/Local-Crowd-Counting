# Adaptive Mixture Regression Network with Local Counting Map for Crowd Counting

Anonymous ECCV submission, paper ID:4582.

PyTorch code for local crowd counting.


## Getting Started

### Preparation
- Prerequisites
  - Python >= 3.5
  - Pytorch >= 1.0.1
  - other libs in ```requirements.txt```, run ```pip install -r requirements.txt```.

- Data Preparation
  - Download ```ShanghaiTech, UCF-QNRF, UCF_CC_50``` datasets from the official websites 
    and unzip them in ```./ProcessedData```.
  - Run ```cd ./datasets/XXX/``` and ```python prepare_XXX_mod64.py``` to resize images and generate training labels.
  
- Pretrained Model
  - Some Counting Networks (such as VGG, CSRNet and so on) adopt the pre-trained models on ImageNet.
    Download ```vgg16-397923af.pth``` from ```torchvision.models```.
  - Place the pre-trained model to ```./models/Pretrain_model/```. 

- Training Model

    | Model         | Backbone |  Lable   | Other Modules |
    |---------------|:--------:|:--------:|:-------------:|
    | VGG16_DM      | VGG16    | DM       | --            |
    | VGG16_LCM     | VGG16    | LCM      | --            | 
    | VGG16_LCM_REG | VGG16    | LCM      | local regression framework | 
    | CSRNet_DM     | CSRNet   | DM       | --            |
    | CSRNet_LCM    | CSRNet   | LCM      | --            | 
    LCM: local counting map; DM: density map.

- Folder Tree
    ```
    +-- source_code
    |   +-- datasets
        |   +-- SHHA
        |   +-- ......
    |   +-- misc     
    |   +-- models
        |   +-- Prerain_Model
        |   +-- SCC_Model
        |   +-- ......
    |   +-- ProcessedData
        |   +-- shanghaitech_part_A
        |   +-- ......
    ```

### Training

- set the parameters in ```train_config.py``` and ```./datasets/XXX/setting.py```.
- run ```python train.py```.
- run ```tensorboard --logdir=exp --port=6006```.

### Testing

- set the parameters in ```test_config.py```
- run ```python test.py```.
