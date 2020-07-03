# MNIST Dataset (손글씨 이미지 분류 예측)

import tensorflow as tf 
import sys

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(x_train.shape, ' ', y_train.shape) # (60000, 28, 28)   (60000,)
print(x_test.shape, ' ', y_test.shape) #(10000, 28, 28)   (10000,)

#print(x_train[0])
# for i in x_train[1]:
#     for j in i :
#         sys.stdout.write('%s '%j)
#     sys.stdout.write('\n')
#     
# print(y_train[0])

import matplotlib.pyplot as plt 
plt.imshow(x_train[0].reshape(28,28), cmap = 'Greys')
plt.show()