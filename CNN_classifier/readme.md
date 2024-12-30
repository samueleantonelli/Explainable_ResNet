# Fine Tune ResNet on CUB-200-2011 Dataset

# Introduction

This repo contains codes for fine tuning ResNet on CUB_200_2011 datasets.
Only the ResNet models provided by torchvision are available.

## Dataset
CUB200-2011: 11,788 images of 200 bird species. 
- Format of images.txt: <image_id> <image_name>
- Format of train_test_split.txt: <image_id> <is_training_image>
- Format of classes.txt: <class_id> <class_name>
- Format of iamge_class_labels.txt: <image_id> <class_id>

## How to use
git clone https://github.com/zhangyongshun/resnet_finetune_cub.git
cd resnet_finetune_cub
pip install requirements.txt

#You need to modify the paths of model and data in utils/Config.py
python train.py --net_choice ResNet --model_choice 50 #ResNet50, use default setting to get the Acc reported in readme
```

## Results

There are some results as follows:  

![result](https://github.com/zhangyongshun/resnet_finetune_cub/raw/master/imgs/results.png)
