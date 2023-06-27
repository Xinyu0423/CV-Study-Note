# Tensorflow Note

## Tensorflow v1
1. Define constant and set it to 36
   1. `tf.constant(36, name='y_hat')`
2. When init is run later (session.run(init))
   1. `tf.global_variables_initializer()`
3. Create a session and print the output(as shown below)
   1. `with tf.Session() as session:`
   2. `session.run(init) `
      1. Initializes the variables
   3. `print(session.run(loss))`
      1. Prints the loss
4. Multiply tf variables
   1. `tf.multiply(a,b)`
5. 在初始化variable后，需要create a session并且运行这个session
   1. `sess = tf.Session()`
   2. `print(sess.run(c))`
6. A placeholder is an object whose value you can specify only later. To specify values for a placeholder, you can pass in values by using a "feed dictionary" (feed_dict variable).
   1. `x = tf.placeholder(tf.int64, name = 'x')`
   2. `print(sess.run(2 * x, feed_dict = {x: 3}))`
   3. `sess.close()`
   4. Here's what's happening: When you specify the operations needed for a computation, you are telling TensorFlow how to construct a computation graph. The computation graph can have some placeholders whose values you will specify only later. Finally, when you run the session, you are telling TensorFlow to execute the computation graph.
7. Computing the sigmoid
   1. `tf.sigmoid(...)`
8. Two ways create and use session in tf
   1. Method 1:
      1. `sess = tf.Session()`
        # Run the variables initialization (if needed), run the operations
       2. result = sess.run(..., feed_dict = {...})
       3. sess.close() # Close the session
    2. Method 2:
       1. with tf.Session() as sess: 
    # run the variables initialization (if needed), run the operations
        2. result = sess.run(..., feed_dict = {...})
    # This takes care of closing the session for you :)
9. Computing the Cost
   1.  `tf.nn.sigmoid_cross_entropy_with_logits(logits = ...,  labels = ...)`
   2.  $$- \frac{1}{m}  \sum_{i = 1}^m  \large ( \small y^{(i)} \log \sigma(z^{[2](i)}) + (1-y^{(i)})\log (1-\sigma(z^{[2](i)})\large )\small\tag{2}$$
   3.  Your code should input z, compute the sigmoid (to get a) and then compute the cross entropy cost  𝐽. All this can be done using one call to tf.nn.sigmoid_cross_entropy_with_logits, which computes
10. One Hot encodings
    1.  `tf.one_hot(labels, depth, axis)`
10. Initialize with zeros and ones¶
    1.  Now you will learn how to initialize a vector of zeros and ones. The function you will be calling is `tf.ones()`. To initialize with zeros you could use tf.zeros() instead. These functions take in a shape and return an array of dimension shape full of zeros and ones respectively.
11. tf.contrib.layers.xavier_initializer
    1.  该函数返回一个用于初始化权重的初始化程序 “Xavier” 。
    2.  这个初始化器是用来使得每一层输出的方差应该尽量相等。
12.  tf.reduce_mean函数
    1.   tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
    2.   `reduce_mean(input_tensor,`
                `axis=None,`
                `keep_dims=False,`
                `name=None,`
                `reduction_indices=None)`
         1. 第一个参数input_tensor： 输入的待降维的tensor;
         2. 第二个参数axis： 指定的轴，如果不指定，则计算所有元素的均值;
         3. 第三个参数keep_dims：是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;
         4. 第四个参数name： 操作的名称;
         5. 第五个参数 reduction_indices：在以前版本中用来指定轴，已弃用;
    3. 类似函数还有:
       1. tf.reduce_sum ：计算tensor指定轴方向上的所有元素的累加和;
       2. tf.reduce_max  :  计算tensor指定轴方向上的各个元素的最大值;
       3. tf.reduce_all :  计算tensor指定轴方向上的各个元素的逻辑和（and运算）;
       4. tf.reduce_any:  计算tensor指定轴方向上的各个元素的逻辑或（or运算）;
 13. Backward propagation
     1.  After you compute the cost function. You will create an "optimizer" object. You have to call this object along with the cost when running the tf.session. When called, it will perform an optimization on the given cost with the chosen method and learning rate.
     2.  `optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)`
     3.  To make the optimization you would do:
     4.  `_ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})`
 14. Padding
     1.   Implement the following function, which pads all the images of a batch of examples X with zeros. Use np.pad. Note if you want to pad the array "a" of shape  (5,5,5,5,5)with pad = 1 for the 2nd dimension, pad = 3 for the 4th dimension and pad = 0 for the rest, you would do:
     2.   `a = np.pad(a, ((0,0), (1,1), (0,0), (3,3), (0,0)), 'constant', constant_values = (..,..))`


