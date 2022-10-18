# FPNdemo

> 环境：
>
> NVIDIA GeForce RTX 3090 One
>
> cityscapesscripts==2.2.0
>
> torch==1.10.2+cu113
>
> torch-tb-profiler==0.4.0
>
> torchaudio==0.10.2+cu113
>
> torchsummary==1.5.1
>
> torchvision==0.11.3+cu113

## 0 数据生成

首先去cityspaces下载数据集

mkdir Cityscapes

下载getFine_trainvaltest数据集放在Cityscapes/getFine_trainvaltest下

下载leftImg8bit 11G数据集放在Cityscapes/leftImg8bit下

### 0.1 脚本介绍

数据集生成脚本文件说明如下图：

> SemanticSegmentationUsingPFPN/cityscapesscripts
>
> 文件说明
>
> ![img](image/README/intro_script.png)

### 0.2 如何使用

> cityscapesscripts/helpers/labels.py
>
> 更改其中的trainId 为255即为忽略的类别
>
> 训练时需要把255作为忽略类
>
> 运行
>
> `python3 cityscapesscripts/preparation/createTrainIdLabelImgs.py`
>
> 即可标记我们需要的数据 这里采用论文的19类

### 0.3 生成和拷贝数据

**将代码单独生成在Cityscapes/trainImg目录下**

> 运行数据转移代码

`python3 utils/dataMove.py`

> 运行标签转移代码

`python3 utils/labelMove.py`

> 编写dataset类 生成训练格式的数据

`python3 utils/dataset.py`

## 1 模型文件

详见

> model

## 2 训练脚本

### 2.1 loss

### 2.2 Metrics

### 2.3 traner
