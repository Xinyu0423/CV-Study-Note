# BiSeNet V2
1. https://zhuanlan.zhihu.com/p/141692672?utm_id=0

## 和BiSeNet的相同点和区别
1. 和BiSeNet相同，都是通过双边分割网络，即Spatial Path（细节分支）和Context Path（语义分支）来实现高精度和高效率的实时语义分割。
   1. 其中细节分支记录了较多图片原本的信息，具有宽通道和浅层，用于捕获低层细节并生成高分辨率的特征表示
   2. 语义分支则保留了图片的抽象能力，最后卷积形成的 feature map拥有感受野，通道窄，层次深，获取高层次语义语境。
2. 与BiSeNet不同的是，BiSeNet使用FFM作为Spatial Path和Context Path的聚合策略，而在BiseNet V2中，使用了一个引导聚合层(Aggreation Layer)来增强相互连接和融合这两种类型的特征表示。此外，还设计了一种增强型训练策略(Booster)，在不增加任何推理代价的情况下提高分割性能。
   [![2023-03-21-20-02-36.png](https://i.postimg.cc/nLvm3Z6F/2023-03-21-20-02-36.png)](https://postimg.cc/3yxWRM8V)


