## Transformer

## Link:
https://zhuanlan.zhihu.com/p/338817680

## Transformer整体结构
[![2023-06-09-00-51-28.png](https://i.postimg.cc/2ygCMQTb/2023-06-09-00-51-28.png)](https://postimg.cc/YLfcmm5M)
1. Transformer 由 Encoder 和 Decoder 两个部分组成
2. Encoder 和 Decoder 都包含 6 个 block。Transformer 的工作流程大体如下：
   1. 第一步：获取输入句子的每一个单词的表示向量 X，X由单词的 Embedding（Embedding就是从原始数据提取出来的Feature） 和单词位置的 Embedding 相加得到。
   [![2023-06-09-00-52-47.png](https://i.postimg.cc/KvnYwVSm/2023-06-09-00-52-47.png)](https://postimg.cc/N5jYrdjn)
   2. 第二步：将得到的单词表示向量矩阵 (如上图所示，每一行是一个单词的表示 x) 传入 Encoder 中，经过 6 个 Encoder block 后可以得到句子所有单词的编码信息矩阵 C，如下图。单词向量矩阵用 表示， n 是句子中单词个数，d 是表示向量的维度 (论文中 d=512)。每一个 Encoder block 输出的矩阵维度与输入完全一致。
   [![2023-06-09-00-59-51.png](https://i.postimg.cc/gkHrT583/2023-06-09-00-59-51.png)](https://postimg.cc/TphRytkP)
   3. 第三步：将 Encoder 输出的编码信息矩阵 C传递到 Decoder 中，Decoder 依次会根据当前翻译过的单词 1~ i 翻译下一个单词 i+1，如下图所示。在使用的过程中，翻译到单词 i+1 的时候需要通过 Mask (掩盖) 操作遮盖住 i+1 之后的单词。
   [![2023-06-09-01-01-10.png](https://i.postimg.cc/SjF8fM0q/2023-06-09-01-01-10.png)](https://postimg.cc/Cn7zFdL6)
3. 