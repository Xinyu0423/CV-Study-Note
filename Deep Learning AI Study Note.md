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



