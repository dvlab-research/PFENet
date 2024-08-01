# PFENet
This is the implementation of our paper [**PFENet: Prior Guided Feature Enrichment Network for Few-shot Segmentation**](http://arxiv.org/abs/2008.01449) that has been accepted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). 

### Pinned
Our latest works are available at:

Hierarchical Dense Correlation Distillation for Few-Shot Segmentation (CVPR 2023): https://github.com/Pbihao/HDMNet.

Generalized Few-shot Semantic Segmentation (CVPR 2022): https://github.com/dvlab-research/GFS-Seg.

# Get Started

### Environment
+ torch==1.4.0 (torch version >= 1.0.1.post2 should be okay to run this repo)
+ numpy==1.18.4
+ tensorboardX==1.8
+ cv2==4.2.0


### Datasets and Data Preparation

Please download the following datasets:

+ PASCAL-5i is based on the [**PASCAL VOC 2012**](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and [**SBD**](http://home.bharathh.info/pubs/codes/SBD/download.html) where the val images should be excluded from the list of training samples.

Images are available at: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

annotations: https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing

Note: If you wish to reproduce the results presented in the paper, please follow the provided [**datalist**](https://github.com/dvlab-research/PFENet/tree/master/lists) to use the specified data. As we followed Shaban's OSLSM work, we only utilized a subset of the complete dataset, which is not the entire 12,000 entries. However, different studies may have different usage requirements. To ensure fair comparison, we kindly request you to select the data according to your specific needs.

+ [**COCO 2014**](https://cocodataset.org/#download).

This code reads data from .txt files where each line contains the paths for image and the corresponding label respectively. Image and label paths are separated by a space. Example is as follows:

    image_path_1 label_path_1
    image_path_2 label_path_2
    image_path_3 label_path_3
    ...
    image_path_n label_path_n

Then update the train/val/test list paths in the config files.

#### [Update] We have uploaded the lists we use in our paper.
+ The train/val lists for COCO contain 82081 and 40137 images respectively. They are the default train/val splits of COCO. 
+ The train/val lists for PASCAL5i contain 5953 and 1449 images respectively. The train list should be **voc_sbd_merge_noduplicate.txt** and the val list is the original val list of pascal voc (**val.txt**).

##### To get voc_sbd_merge_noduplicate.txt:
+ We first merge the original VOC (voc_original_train.txt) and SBD ([**sbd_data.txt**](http://home.bharathh.info/pubs/codes/SBD/train_noval.txt)) training data. 
+ [**Important**] sbd_data.txt does not overlap with the PASCALVOC 2012 validation data.
+ The merged list (voc_sbd_merge.txt) is then processed by the script (duplicate_removal.py) to remove the duplicate images and labels.

### Run Demo / Test with Pretrained Models
+ Please download the pretrained models.
+ We provide **8 pre-trained models**: 4 ResNet-50 based [**models**](https://1drv.ms/u/c/af3d5d777523123a/EU_IQ0BZn0lErngawUbZ3UIBhYuJ1F8DJMoacP1zPJ-Hlg?e=mIKR0n) for PASCAL-5i and 4 VGG-16 based [**models**](https://1drv.ms/u/c/af3d5d777523123a/EU_IQ0BZn0lErngawUbZ3UIBhYuJ1F8DJMoacP1zPJ-Hlg?e=mIKR0n) for COCO.
+ Update the config file by specifying the target **split** and **path** (`weights`) for loading the checkpoint.
+ Execute `mkdir initmodel` at the root directory.
+ Download the ImageNet pretrained [**backbones**](https://drive.google.com/drive/folders/1Hrz1wOxOZm4nIIS7UMJeL79AQrdvpj6v) and put them into the `initmodel` directory.
+ Then execute the command: 

    `sh test.sh {*dataset*} {*model_config*}`

Example: Test PFENet with ResNet50 on the split 0 of PASCAL-5i: 

    sh test.sh pascal split0_resnet50


### Train

Execute this command at the root directory: 

    sh train.sh {*dataset*} {*model_config*}


# Related Repositories

This project is built upon a very early version of **SemSeg**: https://github.com/hszhao/semseg. 

Other projects in few-shot segmentation:
+ OSLSM: https://github.com/lzzcd001/OSLSM
+ CANet: https://github.com/icoz69/CaNet
+ PANet: https://github.com/kaixin96/PANet
+ FSS-1000: https://github.com/HKUSTCV/FSS-1000
+ AMP: https://github.com/MSiam/AdaptiveMaskedProxies
+ On the Texture Bias for FS Seg: https://github.com/rezazad68/fewshot-segmentation
+ SG-One: https://github.com/xiaomengyc/SG-One
+ FS Seg Propagation with Guided Networks: https://github.com/shelhamer/revolver


Many thanks to their great work!

# Citation

If you find this project useful, please consider citing:
```
@article{tian2020pfenet,
  title={Prior Guided Feature Enrichment Network for Few-Shot Segmentation},
  author={Tian, Zhuotao and Zhao, Hengshuang and Shu, Michelle and Yang, Zhicheng and Li, Ruiyu and Jia, Jiaya},
  journal={TPAMI},
  year={2020}
}

@InProceedings{peng2023hierarchical,
  title={Hierarchical Dense Correlation Distillation for Few-Shot Segmentation},
  author={Peng, Bohao and Tian, Zhuotao and Wu, Xiaoyang and Wang, Chenyao and Liu, Shu and Su, Jingyong and Jia, Jiaya},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}

@InProceedings{tian2022gfsseg,
    title={Generalized Few-shot Semantic Segmentation},
    author={Zhuotao Tian and Xin Lai and Li Jiang and Shu Liu and Michelle Shu and Hengshuang Zhao and Jiaya Jia},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```
