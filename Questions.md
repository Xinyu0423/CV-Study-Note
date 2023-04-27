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
   1. z记录了输入的特征融合
2. w为什么是参数
   1. w是通过bp通过计算梯度更新
3. 为什么x=a[0]，y_hat=a[2]
   1. 
4. [![2023-04-03-11-11-10.png](https://i.postimg.cc/Dy1vqXWc/2023-04-03-11-11-10.png)](https://postimg.cc/3dJQhWFy)这里w和x为什么是矩阵
5. 激活函数是非线性的吗
   1. 激活函数都是非线性的，引入激活函数的目的是加入非线性
6. Loss是什么，最后正向传播得到的结果吗
   1. loss是通过正向传播得到的结果，即y_hat
   2. 反向传播是通过loss计算w和b
7. 梯度下降和损失函数的关系
   1. 梯度下降永远是向loss最小的方向移动
8. n是什么
9.  `c2[:,-1]`这个逗号是什么

## Questions from 04/18/2023
1. 模型的泛化是什么
   1. 泛化能力即模型的学习能力
2. P48第2分钟的图片是什么意思
   1. [![2023-04-18-07-04-55.png](https://i.postimg.cc/gkLP0yzL/2023-04-18-07-04-55.png)](https://postimg.cc/Cz02Pqfw)
   2. 搜索一下方差的问题
3. 吴恩达C1Week3作业里的cost是什么
4.  Question: Implement the function backward_propagation() 左右两个图一个是for signle hideen unit，一个是for single hidden layer吗
    1.  具体说下实现流程
5. Prediction怎么做预测的
6. Week4
   1. Building your Deep Neural Network-Step by Step
      1. 4.2d为什么要再做一个，是要做一次sigmoid吗
      2. 6.1第21行为什么要np.matrix
      3. 6.3 40行为什么要-2
      4. 6.4 为什么parameter//2==layer数
7. Cost function是做什么的
   1. 即为Loss
8. P51为什么加入正则化后会简化这个NN网络
9.  为什么在数据量不大的情况下反而容易发生过拟合的情况
10. 为什么对图像进行翻转/resize可以防止overfitting
11. P54 07:24是什么意思
12. normalization是为了做梯度下降吗
13. Grad check