## Link
https://www.bilibili.com/video/BV1FT4y1E74V?p=27&spm_id_from=pageDriver&vd_source=1cae1771f5cc18363af9b40168893ecd\

## 计算神经网络的输出
[![2023-04-03-10-15-35.png](https://i.postimg.cc/rsGbXXzH/2023-04-03-10-15-35.png)](https://postimg.cc/0KQZD4YG)
1. 如图是一个逻辑回归的单层神经网络，在计算神经网络的输出是，当参数进入layer后，需要进行2次计算，一次是z的计算，第二步是以sigmoid（z）为激活函数得出a
[![2023-04-03-10-28-58.png](https://i.postimg.cc/nMb3Ld9D/2023-04-03-10-28-58.png)](https://postimg.cc/LgDtC3T9)
2. 在进行神经网络计算时，为了增加计算的效率，我们将这些等式向量化。向量化的过程是将所有w纵向堆积起来。如图，因为有3个输入，因此最终会得到一个4x3的矩阵，将这些和x堆积成的矩阵（1*3）和b的矩阵（1*4）计算后，得到z


## 激活函数
1. Sigmoid
   1. [![2023-04-03-11-15-27.png](https://i.postimg.cc/MpdvQQ4F/2023-04-03-11-15-27.png)](https://postimg.cc/K13GXRmP)
   2. Sigmoid函数的导数
      1. [![2023-04-05-12-33-10.png](https://i.postimg.cc/T3yfSZw1/2023-04-05-12-33-10.png)](https://postimg.cc/QFr2KYcZ)
      2. 因此也可以写成g‘(a)=a(1-a)
2. tanh
   1. [![2023-04-03-11-16-22.png](https://i.postimg.cc/YqxS5FpC/2023-04-03-11-16-22.png)](https://postimg.cc/crrZYrgp)
   2. tanh函数在多数情况下都要比sigmoid函数更好，因为它有将数据聚合在0之间的作用，除了当y的值要在0和1之间时
   3. tanh的导数
      1. g'(z)=1-(tanh(z))^2
3. Relu
   1. [![2023-04-03-11-21-53.png](https://i.postimg.cc/jjnJNhfS/2023-04-03-11-21-53.png)](https://postimg.cc/gwdJb84C)
   2. 通常不确定用什么激活函数时，可以使用Relu
   3. Relu的导数
      1. [![2023-04-05-12-35-52.png](https://i.postimg.cc/5txpjMCN/2023-04-05-12-35-52.png)](https://postimg.cc/k2pWhzsk)
      2. 因为z=0的情况概率太小，所以通常忽视z=0的情况
4.  Leaky Relu
    1.  [![2023-04-03-11-25-11.png](https://i.postimg.cc/hGhTPBs6/2023-04-03-11-25-11.png)](https://postimg.cc/QBGH0vRJ)
    2.  Leaky Relu的导数
        1.  [![2023-04-05-12-37-25.png](https://i.postimg.cc/nc3w7rn9/2023-04-05-12-37-25.png)](https://postimg.cc/hf7rRSzK)
5. 为什么需要激活函数
   1. 在不使用激活函数时$a=(w^[2]*w^[1])x+(w^[2]*b^[1]+b^[2])$即$a=w'x+b'，因此神经网络只是在进行线性计算，并没有意义。
6. 激活函数通常是非线性的

## 反向传播 （Back Propagation）
1. 反向传播公式（左侧是正向传播公式）
   1. [![2023-04-05-12-46-59.png](https://i.postimg.cc/zfNbrnc9/2023-04-05-12-46-59.png)](https://postimg.cc/w1WT5sHV)
2. 损失函数的公式（Logistic Regression）
   1. [![2023-04-05-12-49-00.png](https://i.postimg.cc/760qYcJ9/2023-04-05-12-49-00.png)](https://postimg.cc/KkG600N3)
3. 梯度下降公式和向量化
   1. [![2023-04-05-13-01-37.png](https://i.postimg.cc/8kyr8fDC/2023-04-05-13-01-37.png)](https://postimg.cc/hhdGTjXF)

## 初始化参数
1. 为什么要做初始化
   1. 在不做初始化时，即weight=0时，所有的hidden unit都是在做相同的计算，因此所有的hidden unit都是一样的
   2. [![2023-04-05-13-11-56.png](https://i.postimg.cc/Qxft92Vx/2023-04-05-13-11-56.png)](https://postimg.cc/D8bFRYqt)
   3. 在做初始化时，通常将参数设置的比较小，如图，参数为0.01。因为在参数特别大时,假设使用的是sigmoid函数，得到的结果z值就会特别大，因此斜率就会特别小，这样不利于做梯度下降

## Bias和Variance
1. 当Train Data的错误率高时，我们称这个模型有很高的Bias，即这个模型是underfitting的
2. 当Train Data和Dev data的差距很大时，我们称这个模型有很高的Variance，即这个模型时overfitting的
3. 因此Bias为模型的输出与正确答案的误差
4. Variance为针对不同的输入资料，模型输出的变化分布（变化性）
   1. 例子：[![2023-04-18-07-12-38.png](https://i.postimg.cc/0jtZFG8C/2023-04-18-07-12-38.png)](https://postimg.cc/30vgDv0y)
   2. 如上图，在第二个例子中，train的error为15%，dev的error为16%因为train的error和dev相差不大，所以不是high variance的情况，而第三个例子因为train和d ev的error相差较大，所以为high variance
5. 当Bias很高时，解决措施：
   1. Bigger Network
   2. Train longer
   3. （NN architecture search）
6. 当Variance很高时
   1. more Data
   2. 正则化
   3. （NN architecture search）
7. 在当前big data的时代，Bias variance trade基本不适用，即可以做到在减小Bias的同时，不增加Variance

## 正则化
1. L1正则和L2正则
   1. [![2023-04-22-04-04-32.png](https://i.postimg.cc/65BgwW76/2023-04-22-04-04-32.png)](https://postimg.cc/7bQmnyMj)
   2. L2正则和L1正则的区别：
      1. 让某些没有用的特征便成0，即剔除了某些特征

## Grad check
1. 当grad check结果小于10**-7时，back prop结果通常是正确的
2. 小于10**-5时，可能是错误的
3. 当小于10**-3，通常back prop是错的
4. 使用Grad Check的建议
   1. [![2023-04-22-05-00-11.png](https://i.postimg.cc/VNw7w1LB/2023-04-22-05-00-11.png)](https://postimg.cc/18W00kFf)

## What is L2-regularization actually doing?:
1. L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.
2. **What you should remember** -- the implications of L2-regularization on: - The cost computation: - A regularization term is added to the cost - The backpropagation function: - There are extra terms in the gradients with respect to weight matrices - Weights end up smaller ("weight decay"): - Weights are pushed to smaller values.

## Dropout
1. A common mistake when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training.
Deep learning frameworks like tensorflow, PaddlePaddle, keras or caffe come with a dropout layer implementation. Don't stress - you will soon learn some of these frameworks.
2. **What you should remember about dropout:** - Dropout is a regularization technique. - You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time. - Apply dropout both during forward and backward propagation. - During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when keep_prob is other values than 0.5.

## 梯度检验
1. 梯度检验可验证反向传播的梯度与梯度的数值近似值之间的接近度（使用正向传播进行计算）。
2. 梯度检验很慢，因此我们不会在每次训练中都运行它。通常，你仅需确保其代码正确即可运行它，然后将其关闭并将backprop用于实际的学习过程。

## 动量梯度下降法
1. [![2023-04-27-07-13-35.png](https://i.postimg.cc/FzGGf2Yj/2023-04-27-07-13-35.png)](https://postimg.cc/YLGQZDf9)

## RSMProp(Root Square Mean Prop)
1. [![2023-04-28-20-08-53.png](https://i.postimg.cc/c4HnxzbC/2023-04-28-20-08-53.png)](https://postimg.cc/Btr6Nmsf)

## Adam优化算法(Adoption Moment Estimation)
1. [![2023-04-28-20-15-44.png](https://i.postimg.cc/mDpFCNcB/2023-04-28-20-15-44.png)](https://postimg.cc/DJsZKGdM)
2. [![2023-04-28-20-16-26.png](https://i.postimg.cc/3w7pTw55/2023-04-28-20-16-26.png)](https://postimg.cc/K1q47x8J)

## 学习率衰减(Learning Rate Decay)
1. [![2023-04-28-20-35-40.png](https://i.postimg.cc/C1WYDz4z/2023-04-28-20-35-40.png)](https://postimg.cc/Fdg2XF84)
2. [![2023-04-28-20-36-37.png](https://i.postimg.cc/GtGRjdMz/2023-04-28-20-36-37.png)](https://postimg.cc/wRqnq8Ys)

## 优化算法的总结
1. 冲量通常会有所帮助，但是鉴于学习率低和数据集过于简单，其影响几乎可以忽略不计。同样，你看到损失的巨大波动是因为对于优化算法，某些小批处理比其他小批处理更为困难。
2. 另一方面，Adam明显胜过小批次梯度下降和冲量。如果你在此简单数据集上运行更多epoch，则这三种方法都将产生非常好的结果。但是，Adam收敛得更快。
3. Adam的优势包括：
   1. 相对较低的内存要求（尽管高于梯度下降和带冲量的梯度下降）
   2. 即使很少调整超参数，通常也能很好地工作（alpha除外）

## ML的各种参数
1. Learning Rate $\alpha$
   1. 当$\alpha$较小时，随机给Learning Rate取值方法
      1. r=-4*np.random.rand(-4,0)
      2. $\alpha$=10^r
2. Momentum $\beta$
   1. 通常initialize为0.9
3. Adam优化器中的$\beta1$ $\beta2$ $\epsilon$
   1. 通常$\beta1$=0.9，$\beta2$=0.999 $\epsilon$=10^-8
4. number of layers
5. number of hidden units
6. learning rate deceay
7. mini-batch size

## BN的优势
1. 通过归一化隐藏单元的激活值来加速运行效率
2. 因为对mini-batch中的z_l添加了noise，对输入有一些归一化的作用（但不要把BN作为归一化的工具）

## Tensorflow
1. Tensorflow是深度学习中经常使用的编程框架
   1. Tensorflow中的两个主要对象类别是张量和运算符
   2. 在Tensorflow中进行编码时，你必须执行以下步骤：
   3. 创建一个包含张量（变量，占位符...）和操作（tf.matmul，tf.add，...）的计算图      - 创建会话      - 初始化会话      - 运行会话以执行计算图
   4. 你可以像在model（）中看到的那样多次执行计算图
   5. 在“优化器”对象上运行会话时，将自动完成反向传播和优化。
2. English Version
   1. **What you should remember**: - Tensorflow is a programming framework used in deep learning - The two main object classes in tensorflow are Tensors and Operators. - When you code in tensorflow you have to take the following steps: - Create a graph containing Tensors (Variables, Placeholders ...) and Operations (tf.matmul, tf.add, ...) - Create a session - Initialize the session - Run the session to execute the graph - You can execute the graph multiple times as you've seen in model() - The backpropagation and optimization is automatically done when running the session on the "optimizer" object.

## Padding
1. feather map大小公式:
   1. 不使用Padding时
      1. `f=n-f+1`
   2. 使用Padding后
      1. `f=n+2p-f+1`
2. 在CNN中，卷积核的大小通常是奇数的

# Stride
1. 当使用Stride后，feather map大小公式：
   1. `((n+2p-f)/s  + 1) * ((n+2p-f)/s + 1)`
   2. 当结果不是整数时，向下取整

## 三维图像上的卷积
1. 在卷积时，原图片的通道数(channel)必须match filter的channel的通道数
2. 每一个在kernel上的数字都是parameter
3. 在卷积操作中，通常随着卷积次数的增多，图片越来越小，但channel变得越来越多
4. 通常在做Max Pooling时，不使用Padding
5. 通常只有带有权重(weight)的才作为layer，比如当同时做conv和pooling操作时,指算为1层
6. 卷积的操作包括，卷积层，池化层，全联接层
7. 为什么卷积在图片中好用
   1. 因为卷积操作大大减少了参数的数量
   2. Parameters Sharing
      1. 在卷积操作中，将原图片和 filter进行计算，因此所使用的参数经过多次计算，即共享了参数
   3. Sparsity of connection(稀疏连接)
      1. 在卷积操作中，最后得到的feather map的每一个值都只和原图片中的每一个filter的位置对应，原图片中的其它位置对feather map中的该值没有影响，因此减少了过拟合的发生的概率

## LeNet-5
1. 结构
   1. 28*28*6 conv
   2. avg pool=14*14*6
   3. 10*10*16
   4. avg pool=5*5*16
   5. fc=120
   6. fc=85
   7. softmax
2. 通用结构：
   1. conv-pool-conv-pool-fc-fc-output

## AlexNet
1. Mutiple GPU
2. 使用relu
3. local respose normalization
4. similar with lenet, but much bigger 60 million parameter

## VGG- 16
1. VGG揭示了，随着网络的加深，height and width goes down，每次池化height和width刚好缩小一半，而通道数在不断增加，通道数每次增加一倍，因此图像缩小的比例和channel增加的比例是有规律的

## ResNet
1. 对于传统的深度学习网络应用来说，网络越深，所能学到的东西越多。当然收敛速度也就越慢，训练时间越长。但是如果简单地增加深度，会导致梯度弥散或梯度爆炸。对于该问题的解决方法是正则化初始化和在中间加入Batch Normalization，这样的话可以训练几十层的网络。
2. 虽然通过上述方法 深层网络能够训练了，但是又会出现另一个问题，就是网络退化问题，网络层数增加，但是在训练集上的准确率却饱和甚至下降了。这个不能解释为overfitting，因为overfit应该表现为在训练集上表现更好才对。退化问题说明了深度网络没有很好地被优化。这种现象并不是由于过拟合导致的，过拟合是在训练集中把模型训练的太好，但是在新的数据中表现却不尽人意的情况。
3. 深度残差网络(ResNet)引入了残差块的设计，克服了这种由于网络深度的加深而产生的学习率变低、准确率无法有效提升的问题。
4. 网络太深，模型就会变得不敏感，不同的图片类别产生了近似的对网络的刺激效果，这时候网络均方误差的减小导致最后分类的效果往往不会太好，所以解决思路就是引入这些相似刺激的“差异性因子”。
5. ResNet是由残差块(Residual Block)构成的
   1. 残差块的原理为将前面网络层的输出 直接跳过多层 与后面网络层的输出进行相加。
   2. 简单来说就是，前面较为“清晰”的数据和后面被“有损压缩”的数据 共同作为后面网络数据的输入。
   3. 残差的思想都是去掉相同的主体部分，从而突出微小的变化。
   4. 这个残差块往往需要两层以上，单单一层的残差块并不能起到提升作用。这种残差学习结构可以通过前向神经网络+shortcut连接实现
6. [![2023-06-01-02-01-03.png](https://i.postimg.cc/k5Stc46b/2023-06-01-02-01-03.png)](https://postimg.cc/qtkRBpGJ)
7. 如上图的紫色部分，我们直接将a^[l]向后，到神经网络的深层，在ReLU非线性激活函数前加上a^[l]，将激活值a^[l]的信息直接传达到神经网络的深层，不再沿着主路进行
8. 加上a^[l]后产生了一个残差块（residual block）。插入的时机是在线性激活之后，ReLU激活之前。除了捷径（shortcut），你还会听到另一个术语“跳跃连接”（skip connection），就是指a^[l]跳过一层或者好几层，从而将信息传递到神经网络的更深层。
9. 用残差块能够训练更深的神经网络。所以构建一个ResNet网络就是通过将很多这样的残差块堆积在一起，形成一个很深神经网络
10. 变成ResNet的方法是加上所有跳跃连接，每两层增加一个捷径，构成一个残差块。如下图所示，5个残差块连接在一起构成一个残差网络。
11. [![2023-06-01-02-01-03.png](https://i.postimg.cc/k5Stc46b/2023-06-01-02-01-03.png)](https://postimg.cc/qtkRBpGJ)

## Transfer Learning迁移学习
1. 在网上下载别人的代码和weight
2. 在做分类任务时，如果源代码的分类数和目标分类数不一致，可以删掉源代码地softmax层，重新构建softmax
3. 之后冷冻所有的不相关参数，只使用目标分类任务有关的参数进行训练
4. 以上为当数据量比较小时可以进行的操作
5. 当数据量较大时，可以freeze前面的layer，训练后面的layer，之后构建自己的output unit
6. 通常数据量越多，需要freeze的layer越少，训练的layer越多

## Data Argumentation数据增强
1. Mirroring翻转图像
2. Random Cropping随机剪裁
3. Color shifting通过改变RGB的值改变图片的颜色
   1. PCA Color Argumentation

## Batch Normalization
1. Batch Norm 是一种神经网络层，在许多架构中都普遍使用。 通常作为线性或卷积的一部分添加，有助于在训练期间稳定网络。
   1. 假设输入数据由几个特征 x1、x2、…xn 组成。 每个特征具有不同的值范围。 如特征 x1 的值可能介于 1 到 5 之间，而特征 x2 的值可能介于 1000 到 99999 之间。
   2. 对于每个特征列，分别取数据集中所有样本的值并计算均值和方差。 然后使用下面的公式对值进行标准化。
      1. `X_i=X_i-Mean_i/StdDev_i`
   3. 如果特征在相同的尺度上，损失函数的图像像碗一样均匀。 然后梯度下降可以平稳地下降到最小值。
   4. Batch Norm 层也有自己的参数： beta 和 gamma。
   5. 不可学习的参数: Moving Avg(Mean), Moving Avg(Variance)
   6. 这一步是 Batch Norm 引入的创新点。与要求所有归一化值的均值和单位方差为零的输入层不同，Batch Norm 允许将其值移动（到不同的均值）和缩放（到不同的方差）。它通过将归一化值乘以因子 gamma 并添加因子 beta 来实现此目的。这里是逐元素乘法，而不是矩阵乘法。
   7. 创新点在于，这些因素不是超参数（即模型设计者提供的常数），而是网络学习的可训练参数。每个 Batch Norm 层都能够为自己找到最佳因子，因此可以移动和缩放归一化值以获得最佳预测。
2. 正规化的方法主要是通过计算平均值和方差，然后使得数据分布为均值为0，方差为1的分布。
3. Batch normalization（批归一化）算法的核心是标准化数据（将输入的每个特征分别规范到均值为零，方差为一）。
   1. 在使用Batch normalization层时，模型通常会使用指数移动平均方法来维护每个特征的均值和方差，以便在训练过程中不断进行记录，并使用这些移动平均值来标准化数据。
   2.  指数移动平均线可以看作是平均值或方差的“历史记录”，它通过对权重进行加权平均来修正当前的平均值，使其更加稳定。在Batch normalization中，指数移动平均线分别对每个特征的均值和标准差进行记录和更新，它的更新过程可以描述为： moving_mean = momentum* moving_mean + (1-momentum)*batch_mean moving_var = momentum* moving_var + (1-momentum)*batch_var 其中，momentum是一个参数，通常设置成接近于1的小数，比如0.9或者0.99，它起到了平滑移动平均的作用，让前面历史数据的影响越来越小，当前批次的统计数据的影响更大。
   3. 同时，由于批次的数据是动态变化的，这样的平均值会随着新批次的数据更新而不断改变，以适应数据的变化。 因此，在使用Batch normalization时，可以采用指数移动平均线来保持均值和方差的更新，使得归一化后的数据保持一个更加稳定、一致的标准，从而加速模型的训练和提升模型的泛化性能。
3. Batch Norm的计算发生在计算Z和a之间
   1. 在计算完Z后，进行Batch Norm，然后进行a激活
4. Batch Norm通常和 Mini- Batch一起使用
5. `z_telda=A_l*gamma+beta`

## Word Embedding
1. 使用One-Hot代表每个词汇
   1. 即假设共有1000个词汇，当一个词进入后，所有剩余999位都是0，进入的词汇的位置在One-Hot中的对应位置置为1
   2. 缺点：这样忽略了词与词之间的关联性，因为任意两个不同的词进行相乘后都为0
2. 使用Featurized Representation表示每个词之间的关系
   1. 即：在二维矩阵中计算每个词汇和目标的关联程度，如：横坐标为词语King和Queen，纵坐标目标为Man和Woman，可以预见King和Man的相关性会很高，Queen和Woman的相关性会很高
3. T-SNE算法
   1. 这是一种二维展示每个词汇的相关性的算法，会将多维的信息放在二维可视化的展现，语义类似的词语在二维上互相接近
4. 这种Featurized Representation被称为Embedding
   1. 即每个词都在一个多维的矩阵内，意思相近的词会互相靠近
5. Word Embedding分析两个相近词汇
   1. 如：当知道了Man和Woman之间的关系后，要比较King和某个词汇的相关性
   2. 使用e_man-e_woman=e_king-e_?，即为e_?=e_king-e_man+e_woman
   3. 因为当e_man-e_woman后，除了在某个feature位置，如Gender为某个非0数字外，其它feature都为0，因此只要找到在相同feature位置接近的词汇即可
   4. 因此函数为word e_w=argmax(e_w,e_king-e_man+e_woman)
   5. 在比较两个词汇的接近程度时，通常使用cosine similarity，公式为`sim(u,v)=u^T*v/abs(u)*abs(v)`
6. Word2Vec算法
   1. 随机取一些词作为context word，然后随机取一些词作为target word
   2. 将target word向量化后，将其放入一个softmax层中，softmax会预测出y_hat
   3.  softmax的函数：[![2023-06-27-00-02-25.png](https://i.postimg.cc/vTrNXym8/2023-06-27-00-02-25.png)](https://postimg.cc/K3YfvdLC)
   4.  
