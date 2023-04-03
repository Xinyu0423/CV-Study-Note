# Questions

## Questions from 04/01/2023
1. 图片上为什么会存在负数
   1. 在批归一化(batch normalization)中，可能存在负数
   2. 均值方差归一化（归一化的一种）：`X_归一化后 = （X_归一化前 - 均值）/ 标准差`
   3. Batch Normalization指使用一个Batch的均值和标准差进行归一化
2. 越是接近-1，表示对应位置和feature的反面匹配越完整（https://zhuanlan.zhihu.com/p/27908027）这句话怎么理解
   1. 越是接近-1，表示对应位置和目标图形没有匹配上。
3. Softmax
   1. Softmax后所有的概率相加为1
4. Sigmoid
   1. 公式：[![2023-04-02-00-38-17.png](https://i.postimg.cc/yd5gdjP5/2023-04-02-00-38-17.png)](https://postimg.cc/gXVktv1q)
   2. 主要做2分类问题，任何数在经过Sigmoid后得到的值都在0到1之间
5. FC部分没懂
   1. 在进行上采样后，通过上采样结果和其它feature map结合，可以得到更好的结果
6. 梯度下降
   1. 梯度的方向永远指向函数增长率最大的方向（或者说函数值增长中最快的方向
   2. 为什么使用梯度下降减少loss：梯度方向的是最陡（最陡是指函数变化率最大）的方向，因此使用梯度下降可以快速减少loss

## Question from 04/04/2023
1. 神经网络计算的z是什么
2. 为什么x=a[0]，y_hat=a[2]
3. [![2023-04-03-11-11-10.png](https://i.postimg.cc/Dy1vqXWc/2023-04-03-11-11-10.png)](https://postimg.cc/3dJQhWFy)这里w和x为什么是矩阵