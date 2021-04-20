该存储库为识别指针式仪表，见下图

<img src="/media/cq/data/public/hibiki/meter/data/images/train/831.jpg" alt="831" style="zoom:33%;" />

输出结果为归一化后的指针与刻度之间的相对位置。

理论上可已处理多指针多表盘的针表，只需要将不同的指针与刻度分配至不同的通道中即可。



**所有代码和模型都在积极开发中，如有更改或删除，恕不另行通知。**使用风险自负。



## 讲解

- 语义分割模型，在不同通道上分割出表盘和指针
- 环形的表盘展开为矩形图像
- 二维图像转换为一维数组
- 对刻度数组用均值滤波
- 定位指针相对刻度的位置
- 输出相对位置

语义分割模型采用的是**U2Net**



## 要求

安装了所有依赖项的Python 3.8或更高版本，包括`torch>=1.7`。要安装运行：



## 环境环境

Meter可以在以下任何经过验证的最新环境中运行（[已](https://pytorch.org/)预安装所有依赖项，包括[CUDA](https://developer.nvidia.com/cuda) / [CUDNN](https://developer.nvidia.com/cudnn)，[Python](https://www.python.org/)和[PyTorch](https://pytorch.org/)）：

- 带有免费GPU的**Google Colab和Kaggle**笔记本：[![在Colab中打开](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb) [![在Kaggle中打开](https://camo.githubusercontent.com/a08ca511178e691ace596a95d334f73cf4ce06e83a5c4a5169b8bb68cac27bef/68747470733a2f2f6b6167676c652e636f6d2f7374617469632f696d616765732f6f70656e2d696e2d6b6167676c652e737667)](https://www.kaggle.com/ultralytics/yolov5)
- **Google Cloud**深度学习VM。请参阅[GCP快速入门指南](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart)
- **Amazon**深度学习AMI。请参阅[AWS快速入门指南](https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart)
- **Docker映像**。请参阅《[Docker快速入门指南》](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) [![码头工人拉](https://camo.githubusercontent.com/280faedaf431e4c0c24fdb30ec00a66d627404e5c4c498210d3f014dd58c2c7e/68747470733a2f2f696d672e736869656c64732e696f2f646f636b65722f70756c6c732f756c7472616c79746963732f796f6c6f76353f6c6f676f3d646f636b6572)](https://hub.docker.com/r/ultralytics/yolov5)



## 训练

运行以下命令以在Data文件夹下的数据集上重现结果。在一台GTX2080TI上。使用`--batch-size`为您的GPU允许的最大容量（为11 GB设备显示的批量大小）。

```python
$ python train.py
```



## 推理

运行read_meter.py可以计算data/文件下的val图片，输出的指针数值为归一化后的数值。



## 权重

链接: https://pan.baidu.com/s/1wTPo1wJXrNyEFSu8RrD8Xw  密码: t0p4



将文件放置于weight下
