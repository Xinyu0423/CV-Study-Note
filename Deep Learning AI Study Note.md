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
2. tanh
   1. [![2023-04-03-11-16-22.png](https://i.postimg.cc/YqxS5FpC/2023-04-03-11-16-22.png)](https://postimg.cc/crrZYrgp)
3. tanh函数在多数情况下都要比sigmoid函数更好，因为它有将数据聚合在0之间的作用，除了当y的值要在0和1之间时
4. Relu
   1. [![2023-04-03-11-21-53.png](https://i.postimg.cc/jjnJNhfS/2023-04-03-11-21-53.png)](https://postimg.cc/gwdJb84C)
5. 通常不确定用什么激活函数时，可以使用Relu
6.  Leaky Relu
    1.  [![2023-04-03-11-25-11.png](https://i.postimg.cc/hGhTPBs6/2023-04-03-11-25-11.png)](https://postimg.cc/QBGH0vRJ)