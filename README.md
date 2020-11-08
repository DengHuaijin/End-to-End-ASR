# End-to-End-ASR 
## 以NVIDIA的OpenSeq2Seq为基础构建自己的端到端语音识别模型训练框架

2020/10/23 从model的基类开始写起，开发中...

2020/11/08 模型的训练和评估框架基本实现

## **class Model**
初始版本不需要interence模式 horovod模式 以及interactive模式，后面有需求再添加

同时与horvod相关的一些参数也没有添加进去，比如 **iter_size** 

### EncoderDecoderModel

### Encoder

### Decoder

### Speech2Text
plot attention暂时不支持

## **class Loss**

### CrossEntropyLoss

### CTCLoss

## **class Optimizer**
无视on_horovod的判断条件,直接调用optimizer.apply_gradients()

## train
取消对hook的支持

## Speech2TextDataLayer

### speech_utils
特征文件存储格式：h5py. npy

backend: librosa

支持数据增强augmentation，即语音速率(采样率)的变化

语音特征类型：mfcc, spectrogram, logfbank
