# Note for Tensorflow
1. Link: https://www.bilibili.com/video/BV1oi4y1L7NJ/?spm_id_from=333.999.0.0&vd_source=1cae1771f5cc18363af9b40168893ecd

## 视频1
1. Shape(3,)表示是一维的向量
2. Tensor里做相加时，数据类型（dtype）必须相同
   1. tf为了加快运行效率，不支持自动类型转换
   2. 修改变量类型`tf.cast(var, dtype=tf.float32)`
3. 切片，取2维数组里所有行的最后一列
   1. `c2[:,-1]`
4. 扩充维度
   1. `c2[:,-1, tf.newaxis]`
5. 不使用切片获取二维数组中第二列的数
   1. `c2[...,0]`