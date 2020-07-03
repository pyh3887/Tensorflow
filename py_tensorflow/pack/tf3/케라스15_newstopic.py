# 로이터 뉴스 토픽 분류 
# 46가지 토픽으로 라벨이 달린 11,228개의 로이터 뉴스로 이루어진 데이터셋.
# IMDB 데이터셋과 마찬가지로 , 각 뉴스는(같은 방식을 사용한) 단어 인덱스의 시퀀스로 인코딩되어 있다. 

from tensorflow.keras.datasets import reuters


(train_data,train_label),(test_data,test_label) = reuters.load_data(num_words=10000)
print(train_data.shape,train_label.shape,test_data.shape) # (8982,) (8982,) (2246,)
print(train_data[:2])
print(train_label[:2])

# 로이터 실제 데이터 보기 -----------

word_index = reuters.get_word_index()
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
decode_re = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
print(decode_re)
#==========================

# feature : list >  vector 

import numpy as np 

def vector_seq(sequence, dim= 10000):
    results = np.zeros((len(sequence), dim))
    for i , seq in enumerate(sequence):
        results[i,seq] = 1.
    return results
x_train = vector_seq(train_data) # train_data를 벡터화 
x_test = vector_seq(test_data) # test_Data 를 벡터화 

# print(x_train)
# print(x_test)

# 원핫 인코딩 

def to_onehot(labels, dim = 46):
    res = np.zeros((len(labels),dim))
    for i , label in enumerate(labels):
        res[i,label] = 1
    
    return res

one_hot_train_label = to_onehot(train_label) # trian_label 원핫 처리
one_hot_test_label = to_onehot(test_label) #test_label 원핫 처리

print(one_hot_train_label[0])

from tensorflow.keras.utils import to_categorical
one_hot_train_label = to_categorical(train_label)
one_hot_test_label = to_categorical(test_label)
print(one_hot_train_label[0]) 

# model 

from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(64,input_shape = (10000,), activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['acc'])

#모델 훈련 시 검증데이터(validation data ) 를 사용 

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
print(len(x_val),len(partial_x_train))

y_val = one_hot_train_label[:1000]
partial_y_train = one_hot_train_label[1000:]
print(len(y_val), len(partial_y_train))


history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=128,validation_data=(x_val,y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
print(loss)

import matplotlib.pyplot as plt 

plt.plot(epochs, loss , 'bo' , label='train loss')
plt.plot(epochs, val_loss , 'r' , label='train val_loss')
plt.xlabel('epchos')
plt.ylabel('loss')
plt.show()


acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1,len(loss)+1)
print(acc)

import matplotlib.pyplot as plt 

plt.plot(epochs, acc , 'bo' , label='train acc')
plt.plot(epochs, val_acc , 'r' , label='train val_acc')
plt.xlabel('epchos')
plt.ylabel('acc')
plt.show()


m_eval = model.evaluate(x_test,one_hot_test_label)
print('model evaluate: ' , m_eval)

pred = model.predict(x_test)
print(pred[0])
print(np.sum(pred[0]))
print(np.argmax(pred[0]))
