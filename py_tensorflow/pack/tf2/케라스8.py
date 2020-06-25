
#선형회귀
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
from sklearn.preprocessing import MinMaxScaler,minmax_scale
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from tensorflow.python.keras.layers.core import Activation

data = pd.read_csv('../testdata/Advertising.csv')
del data['no']
print(data.head(3))
print(data.corr())

# 정규화 : 0 ~ 1 사이로 scaling 
scaler = MinMaxScaler()
xy = scaler.fit_transform(data)

#scaler = MinMaxScaler(feature_range = (0,1)) #기본이 0서부터 1

#xy = scaler.fit_transform(data)

xy = minmax_scale(data,axis=0,copy=True) 
print(xy[:2])

#
from sklearn.model_selection import train_test_split

x_train,x_test, y_train,y_test = train_test_split(xy[:,0:-1], xy[:,-1], test_size=0.3, random_state = 123)
print(x_train[:2], ' ', x_train.shape)
print(y_train[:2], ' ', y_train.shape)

model = Sequential()
model.add(Dense(20,input_dim = 3))
model.add(Activation('linear'))
model.add(Dense(10))
model.add(Activation('linear'))
model.add(Dense(1))
model.add(Activation('linear'))

model.summary()

import tensorflow as tf
tf.keras.utils.plot_model(model,'abc.png') #GraphBiz 가 설치되어야 가능 
model.compile(optimizer= Adam(lr=0.01), loss ='mse', metrics=['mse'])

history = model.fit(x_train,y_train,batch_size = 32, epochs=100 , verbose= 2 , validation_split=0.3) #batchsize ↓ 정확도는 느려지지만 속도가 느려짐
# train data 와 test data 를 7:3 으로 나눈것에  7을 학습데이터로 나머지 3을 검정데이터로 씀
print('train: ',history.history['loss']) 

loss = model.evaluate(x_test , y_test,batch_size = 32)
print('test_loss :' , loss)

#설명력 

from sklearn.metrics import r2_score

print('r2_score :' , r2_score(y_test,model.predict(x_test))) #설명력

pred = model.predict(x_test)

print('실제값 :' , y_test[:5])
print('예측값 :', pred[:5].flatten())












