# RNN-Research

**实验标题**: 循环神经网络与长程依赖问题探究

**实验目的**: 比较 RNN、LSTM 和 GRU 在处理长序列依赖问题上的性能，理解梯度消失/爆炸问题。

**实验内容**: 分别实现 RNN、LSTM 和 GRU 单元。在“加法问题”或特定文本分类任务上训练这些模型。监控和分析训练过程中梯度的范数变化。

**实验数据集**: Penn Treebank(文本)，或自构造的序列任务(如加法问题)

**评价指标**: 测试集准确率/困惑度(Perplexity)、训练时间、梯度范数曲线。

**相关经典论文**: 
- Hochreiter,S.,&Schmidhuber,J.(1997).Long short-termmemory.Neural computation.// Cho,K., et al.(2014). Learning phraserepresentations usingRNNencoder-decoder forstatistical machinetranslation.EMNLP
