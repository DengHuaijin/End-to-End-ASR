# End-to-End-ASR

## 以NVIDIA的open sequence to sequence 为基础构建自己的端到端语音识别模型训练框架

2020/10/23 从model的基类开始写起，开发中...

### class model
初始版本不需要interence模式 horovod模式 以及interactive模式，后面有需求再添加

同时与horvod相关的一些参数也没有添加进去，比如 **iter_size** 

### optimizer
无视on_horovod的判断条件,直接调用optimizer.apply_gradients()
