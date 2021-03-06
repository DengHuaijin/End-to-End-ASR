# End-to-End-ASR 
## 以NVIDIA的OpenSeq2Seq为基础构建自己的端到端语音识别模型训练框架

2020/10/23 从model的基类开始写起，开发中...

2020/11/08 模型的训练和评估框架基本实现

2020/11/17 模型compile实现

2020/11/25 模型训练实现

2020/12/24 模型evaluate+infer实现(WER仍然很高，明显训练出了问题)

2021/01/02 实现对单一语音文件的识别（后续追加录音功能）

2021/2/05 通过google的bazel编译kenlm语言模型用于解码

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

编写optimizer测试用例时，最好从optimize_loss函数中返回grads_and_vars，提取里面的梯度grad，然后对比sess.run(grad)
和手动计算出的梯度值，官方是在horovod和iter_size的基础上测试的，这里暂时不需要。

## train

2020/11/12 发现了一个很奇怪的bug，没有hook的话变量无法保存，训练无法启动，可以尝试不使用estimator

这个问题其实并不是hooks导致的，是因为在Saver的输入变量中没有var_list，看了下OpenSeq2Seq的官方代码，他直接在Saver这里
用了一个#noninspection 跳过检查，很骚好吧。不给我没用PyCharm，tf源码中的ValueError没法跳过，有点僵硬

实际上和跳过检查与否也没有关系，因为可以在直接在linux终端执行OpenSeq2Seq的代码，并不会报出ValueError的错误，对比了一下
官方代码的执行结果，发现他在训练之前就已经把graph构建好了，而我的代码在训练之前并没有构建出graph，这应该是问题关键所在

很蠢好吧，create_model里面没有compile，很僵硬

2020/11/16 加入对hook的支持，在hooks.py中继承官方的tf.train.SessionRunHook类自定义hook，因为现阶段还是用estimator更方便一些

2020/11/23 fetches中混入了一个int类型的变量，导致run不起来，需要找到变量源。不过暂时没找到，缓兵之计是在tf的源码中把这个判断无视掉，
不让他报错，具体是在
```
/usr/local/lib/python3.5/dist-packages/tensorflow/python/client/session.py
```
里面实现。
目前是能train起来，先训练一遍看看结果和log吧

2020/12/15 WER和Loss都降不下来，尤其是WER，长期在0.9徘徊甚至超过1，需要检查一下距离计算代码，同时观察tensorboard

2020/12/25 发现encoder里面少了rnn最后的top_layer, 所以相当于只训练了cnn，重新训练一次看看效果

2020/12/26 WER有所下降，但仍然较高，需要再对比一下tensorboard

## infer_interactive

2021/01/12 GPU训练的模型，用CPU识别单一句子的时候会报出 **conv2D only supports "NHWC"** 的错误，初步推测是因为
在把feed_dict数据送入conv layer之前没有用**expand_dims**对 **source_tensor** 进行升维处理。但GPU模式下却没有问题

## Speech2TextDataLayer

data_split() 在train模式下应该对数据按GPU数量进行分割，反之在eval模式下不分割

### speech_utils
特征文件存储格式：h5py. npy

backend: librosa

支持数据增强augmentation，即语音速率(采样率)的变化

语音特征类型：mfcc, spectrogram, logfbank

### language model
推荐用bazel编译，因为有现成的BUILD文件可用。如果直接用g++编译的话会导致用tensorflow载入时缺少相关动态链接库

编译生成ctc_decoder_with_kenlm.so时注意tensorflow目录下ctc的版本，需要tf-1.14对应的版本
