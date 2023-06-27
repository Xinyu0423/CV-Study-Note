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

## Questions from 04/25/2023
1. P61 5:46秒为什么X的shape是(n,m),Y.shape=(m,1)
   1. (n,m)中的n是数据的n个features，m是数据的个数
   2. Y.shape=(m,1)是指m跳训练数据中的一个
2. Week1 homework SGD和mini-batch的区别
   1. SGD不是每次用一个training example吗，Homework question1为什么是iteration through m
3. RMSprop是如何通过加入导数的平方取消训练中的oscillation的
4. Adam算法中的V-corrected_dw和S-corrected_dw是什么
   1. V-corrected_dw和S-corrected_dw都是通过数学期望修正V_dw和S_dw
5. 怎么理解NG在P69 1分20秒说的，在converage后会在一片区域内晃动
   1. 在一片区域内晃动是因为根据梯度，可能走过了optimization point
6. P70说的Decay Rate是什么
   1. Decay Rate是在达到optimization point的区域后，自动更新gradient，让每次走的步数变小，从而不走过optimization point
7. 比如week2 5.1中如何看是否训练到收敛了呢
8. 为什么mini-batch+momentum和mini-batch没有区别呢
9.  怎么理解Adam的优势

## Question from 05/02/2023
1. Batch Normilzation是对同一层的所有参数做批归一化吗
2. BN中的Z_delta=$\gamma$*z_norm+$\beta$是做什么的
3. pip和conda不是同一个python怎么解决,无法安装tensorflow
4. BN可以作用到每一层的layer的输入吗，怎么理解P76 7:25秒说的会加速学习的效率
5. 加入L2正则的作用是加入惩罚项，那为什么不能直接把learning rate调小
6. BN增加了噪音，这里Noise指的是什么
7. 为什么要在测试集上做BN
8. 解释一下P81的代码

## Question from 05/12/2023
1. 解释一下week3 tensorflow 2.6的代码是什么意思
2. 滤波器是什么，用来检测某种特征的吗
   1. 滤波器=filter
3. 鲁棒性(Robust)是什么，是指稳定性吗，怎么理解CNN中的鲁棒性
   1. 鲁棒性=稳定性，
   2. 在深度学习中，鲁棒性（Robustness）指的是模型在面对噪声、扰动、攻击等干扰因素时的稳定性和可靠性。一个鲁棒性强的模型能够在面对各种不确定性的情况下依然保持较好的准确性和稳定性。
   3. 在卷积神经网络（CNN）中，鲁棒性主要体现在两个方面：
      1. 平移不变性：由于卷积核的可移动性，CNN能够对于图片中的不同位置进行特征提取，并保持特征与位置无关，即具有平移不变性。这使得模型对于图片细微位置变化的变化具有一定的鲁棒性。
      2. CNN在面对图像对鲁棒性要求很高的场景下表现出了其优越的特性。比如在图像分类、目标检测、人脸识别等领域，CNN具有较好的鲁棒性、识别稳定性和泛化能力，能够在大量、拥挤、多变的图像数据中进行数据挖掘，提取出数据的内在规律。
4. P113 6分30秒说vertical edge detector是因为kernel都是vertical的吗
   1. vertical edge detector是指能识别出vertical的filter，是通过模型去学习出来的filter
5. 为什么FC层需要做多次
   1. 通过多次FC能让模型的特征更好的融合，从而学到更多的细节
6. P117 7分54秒为什么是84个parameter做 softmax操作
   1. 84个parameter之后会变为10个，然后做softmax
7. Convolution model - Step by Step - v1 3.3和4.1的代码看不明白
8. Convolution model - Application - v1 1.2结果和Answer不一样，是因为tf1和tf2的区别吗

## Questions from 05/30/2023
1. Convolution model - Step by Step - v1 3.3从39行到55行代码是什么意思，主要是vertical_start，vertical_end等的计算方式为什么这么计算
2. 解释一下ResNet中的残差块是什么，具体怎么使用/跳跃到下一层去做激活
   1. 残差块是指在计算时同时计算short cut后的和正常计算的，之后通过相加去融合
3. 怎么理解P122 4分11秒说的加了2个layer和简单的没有2个layer效果相同
   1. 因为在残差网络中，加入了之前的layer的结果，因此即使在后面出现了梯度消失的现象，之前的layer结果依然有效，即最差情况下也是和之前学习的相同
4. 1*1卷积的作用主要是用来减少图片的通道数（降维）吗
   1. 这是其中一个作用
5. 怎么理解P123说的1*1的卷积可以看为是一个FC层，和后面说的可以增加非线性
   1. 在做1*1的卷积时，将原图片和1*1卷积进行乘积和操作，就等于进行了FC操作，因此也就增加了非线性
6. 解释一下P125的Inception Net，这几个conv操作是并行的吗，为什么Inception block之间会有softmax操作
   1. 和并行无关
   2. softmax是在每一个模块后加入预测结果，使预测结果不断优化，保证当前位置之前的block也可以学的好一些
   3. Multitask Learning
7. 简单介绍一下framework，比如caffe
   1. Pytorch，Tensorflow
8.  简单介绍一下迁移学习，这个是做weight初始化的吗
    1.  在做迁移学习时，如果数据量较小，则将前面的layer全部freeze(即不更新前几层layer的参数)
    2.  当数据量较大时，可以少freeze几层layer
9.  `Residual Networks - v1`代码部分不明白

## Question from 06/12/2023  Transformer
1. 怎么理解这句话，因为transformer有position embedding
   1. 因为 Transformer 不采用 RNN 的结构，而是使用全局信息，不能利用单词的顺序信息，而这部分信息对于 NLP 来说非常重要
2. 公式中计算矩阵Q和K每一行向量的内积，为了防止内积过大，因此除以 的平方根
   1. 这里内积是什么
3. Decoder有一端接收的是encoder进行embedding后的信息，另一端直接接收的输入信息，这些信息是没有经过向量化的输入吗
4. 普通Multi-head Attention有Mask的操作吗
   