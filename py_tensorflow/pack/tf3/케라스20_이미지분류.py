# CNN(Convoluation) 
# convolution : feature extraction 역할



import tensorflow as tf
from tensorflow.keras import datasets,layers,models 
import sys

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(x_train.shape, ' ', y_train.shape) # (60000, 28, 28)   (60000,)
print(x_test.shape, ' ', y_test.shape) #(10000, 28, 28)   (10000,)

print(x_train[0])


# import matplotlib.pyplot as plt 
# plt.imshow(x_train[0].reshape(28,28), cmap = 'Greys')
# plt.show()

# 3차원을 4차원으로 구조 변경 . (흑백(1), 칼라(3) 여부 확인용 채널 추가 )

x_train = x_train.reshape((60000,28,28,1)) 
print(x_train.ndim)
train_images = x_train / 255.0 # 정규화
print(train_images[:1])

x_test = x_test.reshape((10000,28,28,1)) # 구글 이외의 제품일 경우 채널의 위치가 다를수 있다.
print(x_test.ndim)
test_test = x_test /255.0
print(test_test[:1])

# model 
model = models.Sequential()


#CNN
model.add(layers.Conv2D(64, kernel_size=(3,3),padding='valid',
                        activation='relu',input_shape=(28,28,1)))
#valid는 패딩을 두지 않겠다는 의미. 
model.add(layers.MaxPool2D(pool_size=(2,2),strides=None))  #strides=(2,2)
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())  #fully connected layer :이미지의 주요 특징만  추출한 CNN 결과를 1차원으로 변경

#분류기로 분류 작업 
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(10,activation='softmax'))

model.summary() # 설정된 구조 화깅ㄴ 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy']) #sparse_categorical > onehot encoding 내부적으로가능

from tensorflow.keras.callbacks import  EarlyStopping

early_stop = EarlyStopping(monitor='loss',patience='5')

history = model.fit(x_train,y_train,batch_size=128,verbose=2,validation_split=0.2,epochs=100,callbacks = [early_stop])

history = history.history
print(history)

#evaluate
train_loss, train_acc = model.evaluate(x_test,y_test,batch_size=128,verbose=2)
test_loss, test_acc = model.evaluate(x_test,y_test,batch_size=128,verbose=2)
print('train_loss, train_acc : ',train_loss, train_acc)
print('test_loss, test_acc : ',test_loss,test_acc)

#모델 저장 후 읽기
model.save('fashion.hdf5')

del model

model = tf.keras.models.load_model('fashion.hdf5')

#predict
import numpy as np
print('예측값  :',np.argmax(model.predict(x_test[:1])))
print('예측값  :',np.argmax(model.predict(x_test[[0]])))
print('실제값 : ',y_test[0])

print('예측값  :',np.argmax(model.predict(x_test[[1]])))
print('실제값 : ',y_test[1])

import matplotlib.pyplot as plt 

def plot_acc(title=None):
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    if title is not None: 
        plt.title(title)
    plt.xlabel('epchos')
    plt.xlabel('acc')
    plt.legend(['train data', 'validation data'],loc =4)

plot_acc('accuracy')
plt.show()

plt.show()















 