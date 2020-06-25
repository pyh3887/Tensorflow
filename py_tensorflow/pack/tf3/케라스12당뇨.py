# 당뇨병 관련 자료로 이항분류 

import tensorflow as tf 
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


np.random.seed(123)


dataset = np.loadtxt('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/diabetes.csv',delimiter =',') #마지막열 당뇨병 환자다 아니다.
print(type(dataset))
print(dataset.shape)
print(dataset[:1])

# train / test 
x_train = dataset[:700,0:8]
y_train = dataset[:700,8]
x_test = dataset[700:,0:8]
y_test = dataset[700:,8]

print(np.unique(y_train)) #[0. 1.]

# train/test 분리 2

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dataset[:,0:8], dataset[:,-1],test_size=0.3,random_state =123)

# 모델 구성 1 
# model = Sequential()
# model.add(Dense(64,input_dim=8,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))

# 모델 구성2 
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

inputs = Input(shape=(8,))
output1 = Dense(64,activation='relu')(inputs)
output2 = Dense(32,activation='relu')(output1)
output3 = Dense(16,activation='relu')(output2)
output4 = Dense(1,activation='sigmoid')(output3)

model = Model(inputs,output4)                  

model.compile(optimizer='adam' , loss='binary_crossentropy',metrics=['acc'])

history = model.fit(x_train,y_train,epochs=500,batch_size=64, verbose=1) # 딥러닝 시작 

scores = model.evaluate(x_test,y_test) 

print('%s : %.2f%%'%(model.metrics_names[1],scores[1] * 100))

print(x_test[:1])
new_x = [[0.05,0.84,0.39,-0.69,0,-0.10,-0.03,-0.10]]
pred = model.predict(new_x)

#print('에측결과: ', pred)
print("예측결과:" ,np.where(pred.flatten() > 0.5,1,0))

import matplotlib.pyplot as plt 
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.show()















