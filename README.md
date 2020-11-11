# End-to-End-ASR 
## 以NVIDIA的OpenSeq2Seq为基础构建自己的端到端语音识别模型训练框架

2020/10/23 从model的基类开始写起，开发中...

2020/11/08 模型的训练和评估框架基本实现

## **class Model**
初始版本不需要interence模式 horovod模式 以及interactive模式，后面有需求再添加

同时与horvod相关的一些参数也没有添加进去，比如 **iter_size** 

model.compile()在utils中的create_model实现，compile中会调用_build_forward_graph

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

optimizer.py中的optimize_loss是梯度更新的核心部分，其流程如下：

**global_step()** -> **update_ops** -> **optimizer** -> **opt.compute_gradients** -> **opt.apply_gradients**

## train
取消对hook的支持

## Speech2TextDataLayer

data_split() 在train模式下应该对数据按GPU数量进行分割，反之在eval模式下不分割

### speech_utils
特征文件存储格式：h5py. npy

backend: librosa

支持数据增强augmentation，即语音速率(采样率)的变化

语音特征类型：mfcc, spectrogram, logfbank
