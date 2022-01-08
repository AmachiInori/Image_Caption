

Image Caption

方法

本项目使用[Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555#:~:text=Show%20and%20Tell%3A%20A%20Neural%20Image%20Caption%20Generator.,that%20combines%20recent%20advances%20in%20computer%20vision%20)中的图像标注生成方法，即标准的CNN-LSTM，编码器-解码器框架：

首先使用CNN将输入图片映射为一内部表示，

既可以分开训练，也可以联合训练


模型

框架

由于较新版的Keras对concat模型合并方法的兼容性较差，使用Concatenate拼接模型时会报出不明错误，故使用Keras-2.0.8框架，并由Tensorflow-1.10与CUDA-11.5提供支持，运行于Python-3.5上。

|依赖项|版本号|
|:---:|:---:|
|certifi            |      2020.6.20     |
|cudatoolkit        |      9.0           |
|cudnn              |      7.6.5         |
|keras              |      2.0.8         |
|numpy              |      1.14.2        |
|pandas             |      0.20.3        |
|pillow             |      4.2.1         |
|pip                |      9.0.1         |
|python             |      3.5.0         |
|scipy              |      1.1.0         |
|tensorboard        |      1.10.0        |
|tensorflow         |      1.10.0        |
|tensorflow-base    |      1.10.0        |
|tensorflow-gpu     |      1.10.0        |
|tqdm               |      4.62.3        |

数据集

由于硬件的限制，无法使用较大的Microsoft COCO数据集，转而使用[Flickr30k数据集](https://paperswithcode.com/dataset/flickr30k)，但即使如此，在我算力只有6.1的显卡(NVIDIA Geforce GTX 1050Ti)上训练一个Epoch所需的时间仍为30小时。

由于体积过大，数据集没有包含在提交的代码中，以空目录代替之。
