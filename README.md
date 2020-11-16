# End-to-End-ASR 
## 以NVIDIA的OpenSeq2Seq为基础构建自己的端到端语音识别模型训练框架

2020/10/23 从model的基类开始写起，开发中...

2020/11/08 模型的训练和评估框架基本实现

2020/11/17 模型compile实现

## **class Model**
初始版本不需要interence模式 horovod模式 以及interactive模式，后面有需求再添加

同时与horvod相关的一些参数也没有添加进去，比如 **iter_size** 

model.compile()在utils中的create_model实现，compile中会调用_build_forward_graph

构建train_op时，不用get_regularization_loss()

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

2020/11/12 发现了一个很奇怪的bug，没有hook的话变量无法保存，训练无法启动，可以尝试不使用estimator

这个问题其实并不是hooks导致的，是因为在Saver的输入变量中没有var_list，看了下OpenSeq2Seq的官方代码，他直接在Saver这里
用了一个#noninspection 跳过检查，很骚好吧。不给我没用PyCharm，tf源码中的ValueError没法跳过，有点僵硬

实际上和跳过检查与否也没有关系，因为可以在直接在linux终端执行OpenSeq2Seq的代码，并不会报出ValueError的错误，对比了一下
官方代码的执行结果，发现他在训练之前就已经把graph构建好了，而我的代码在训练之前并没有构建出graph，这应该是问题关键所在

很蠢好吧，create_model里面没有compile，很僵硬

2020/11/16 加入对hook的支持，在hooks.py中继承官方的tf.train.SessionRunHook类自定义hook，因为现阶段还是用estimator更方便一些

## Speech2TextDataLayer

data_split() 在train模式下应该对数据按GPU数量进行分割，反之在eval模式下不分割

### speech_utils
特征文件存储格式：h5py. npy

backend: librosa

支持数据增强augmentation，即语音速率(采样率)的变化

语音特征类型：mfcc, spectrogram, logfbank
