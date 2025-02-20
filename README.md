# 简介
本仓库提供了一个基于 PyTorch 的 [FCN-8s](https://openaccess.thecvf.com/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf
) 网络实现，并对Portrait Image数据集进行图像语义分割。

## 概述
FCN-8s（Fully Convolutional Networks-8s）是图像语义分割领域中的一种重要网络架构。
它基于全卷积网络（Fully Convolutional Networks, FCN）框架，通过引入跳级（skip）结构，
将深层的全局信息与浅层的局部信息相结合，从而实现了对图像中每个像素点的精确分类。本文档旨在详细介绍FCN-8s网络的基本原理、架构特点以及实现细节
此实现包含了模型定义、数据加载、训练评估和推理的基本框架。

## 网络架构
FCN-8s网络的主要特点包括：
- **全卷积层**：网络中的所有层均为卷积层，没有全连接层。这使得网络可以接受任意尺寸的输入图像。
- **跳级结构**：通过跳级结构将深层和浅层的特征图进行融合，从而结合了全局信息和局部细节，提高了分割精度。
- **上采样操作**：使用双线性插值等上采样技巧对特征图进行上采样，以得到与原图大小相同的分割结果。

## 目录结构：
### FCN_8s/
### ├── README.md             #本文件
### ├── data/                 #数据集文件夹
### │   ├── Portrait-dataset-2000 #人像数据集文件夹
### ├── PortraitDataset.py         #数据预处理和加载脚本
### ├── FCNnet.py           #模型定义
### ├── FCN_train.py             #训练脚本
### ├── FCN_predict.py           #推理脚本
### ├── my_utils.py                #实用工具函数(数据增强、可视化、评估指标等)
### ├── requirements.txt      #项目依赖包
### └── seg_loss_fn.py          #损失函数脚本

## 环境配置
### 1. 安装依赖
使用 requirements.txt 文件安装项目所需的 Python 包
```  JavaScript
pip install -r requirements.txt
```
### 2. 数据集准备
请将您的数据集放置data目录中，对齐数据集目录结构，并在对应数据加载脚本以适应您的数据格式

### [Portrait-dataset-2000.zip数据集下载](https://pan.baidu.com/s/1rsZH297UZPjNlJsmZEtiiA?pwd=ueye)
# 训练模型

### --max_epoch：训练的轮数。
### --batch_size：每批处理的数据量。
### --lr：学习率。
### --loss_function：损失函数，默认交叉熵损失函数
### 运行 FCN_train 脚本开始训练模型(在该文件中需手动修改超参，不能通过命令行)

# 模型推理
### 给定图像文件存放目录，可以逐个图像前景分割抠图，下图是训练过程中相应验证集的评估结果

### 或对给定视频文件名，可以对视频进行图像前景抠图

## [更多详见B站](https://www.bilibili.com/video/BV1PRCLYeErb/)
### 希望这个 README 文件能帮助您快速上手和理解本仓库中的 FCN-8s PyTorch 实现。祝您使用愉快！
### 如果您有任何问题或建议，请通过 GitHub Issues 与我们联系

