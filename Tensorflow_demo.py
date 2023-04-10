import tensorflow as tf
import sys
from tensorflow import keras
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
# print(sys.version)

# print(tf.__version__)
# c=tf.constant(10)
# print(c)

# c1=tf.constant([1,2,3])
# print(c1)

# c2=tf.constant([[1.,2.,3.],[4.,5.,6.]])
# print(c2)

# print("cut 1",c2[:,-1])
# print("cut 2",c2[:-1])

# print("second number",c2[...,0])

# c3=c2+tf.cast(c,dtype=tf.float32)
# print(c3)

(x_train, y_train), (x_test, y_test)=mnist.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train_real, x_valid, y_train_real, y_valid=train_test_split(x_train, y_train, test_size=0.2)
print(x_train_real.shape)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train_real.reshape(-1,1)).reshape(-1,28,28)
x_valid_scaled=scaler.transform(x_valid.reshape(-1,1)).reshape(-1,28,28)
x_test_scaled=scaler.transform(x_test.reshape(-1,1)).reshape(-1,28,28)

model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10,activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer ='adam', metrics=['accuracy'])
model.summary()

history=model.fit(x_train_scaled, y_train_real, epochs=10, validation_data=(x_valid_scaled, y_valid))

print(history.history)
model.evaluate(x_test_scaled, y_test)

predicted_model=tf.keras.Sequential([
    model,
])
img=x_test_scaled[0]
print(img.shape)

img=np.expand_dims(img, 0)
predictions_single_img=predicted_model(img)

print(predictions_single_img)

res=np.argmax(predictions_single_img[0])
print(res, y_test[0])
