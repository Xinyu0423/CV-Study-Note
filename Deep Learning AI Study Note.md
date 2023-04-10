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
