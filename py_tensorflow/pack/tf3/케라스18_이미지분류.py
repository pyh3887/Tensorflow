# MNIST Dataset (손글씨 이미지 분류 예측)

import tensorflow as tf 
import sys

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

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

x_train = x_train.reshape(60000,784).astype('float32')
x_test = x_test.reshape(10000,784).astype('float32')

print(x_train[0])
x_train /= 255 # 정규화
x_test /= 255

print(x_train[0])

print(y_train[0])
print(set(y_train)) # label {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

y_train = tf.keras.utils.to_categorical(y_train,10) #label 을 원핫 인코딩 
y_test = tf.keras.utils.to_categorical(y_test,10)
print(y_train[0])

# train data 의 일부를 validation data 로 사용 
x_val = x_train[50000:60000]
y_val = y_train[50000:60000]
x_train = x_train[0:50000]
y_train = y_train[0:50000]
print(x_val.shape,' ',x_train.shape) # (10000, 784)   (50000, 784)

print()

# 모델 작성 후 분류 
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(512, input_shape=(784,)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))


#print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(patience=3)
history = model.fit(x_train, y_train, epochs=1000,batch_size=128, verbose=2, validation_data=(x_val,y_val),callbacks = [es])

print(history.history)
print(history.history.keys())

import matplotlib.pyplot as plt 

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], 'r--',label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


score = model.evaluate(x_test,y_test, batch_size = 128 , verbose=1)
print('evaluate loss:',score[0])
print('evaluate acc:', score[1])

model.save('mnist_model.hdf5')

del model 
model = tf.keras.models.load_model('mnist_model.hdf5')

#pred 
import numpy as np

print(x_test[:1], x_test[:1].shape)
plt.imshow(x_test[:1].reshape(28,28),cmap = 'Greys')
plt.show()


pred = model.predict(x_test[:1])
print('pred : ', pred)
print('pred : ', np.argmax(pred,1))
print('pred : ', np.argmax(y_test[:1],1))


# 내가 그린 숫자 이미지 분류 예측

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 

im = Image.open('7.png')
img = np.array(im.resize((28,28),Image.ANTIALIAS).convert('L'))
#print(img)

data = img.reshape([1,784])
data = data/255,

#print(data)
#plt.show(data.reshape(28,28) , cmap='Greys')
#plt.show()


new_pred = model.predict(data)
print('new_pred :' , np.argmax(new_pred ,1 ))

         






















