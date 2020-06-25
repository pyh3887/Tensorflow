#다중 선형회귀 , 텐서보드 ( 모델의 구조 및 학습 진행 결과를 시각화하는 툴)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt 

x_data = np.array([[70,85,80],[71,89,88],[50,45,70],[99,90,90],[50,15,10]]) #연속형
y_data = np.array([80,85,55,95,20]) #연속형

model = Sequential()
#model.add(Dense(1,input_dim = 3, activation = 'linear')) #레이어 1개
model.add(Dense(10, input_dim = 3, activation = 'linear')) #레이어 복수    3가 들어옴 첫번째 레이어의 노드는 6개
model.add(Dense(10, activation = 'linear')) #레이어 복수      x1,x2,x3 >> o o o o o o > o o o > o >>>>> 형탠
model.add(Dense(1, activation = 'linear')) #레이어 복수
print(model.summary())

from sklearn.metrics import r2_score

print('설명력:', r2_score(y_data,model.predict(x_data)))

opti = optimizers.Adam(lr=0.01) #adam optimizer
model.compile(optimizer = opti, loss = 'mse', metrics=['mse']) #모델 컴파일 
history = model.fit(x_data, y_data, batch_size= 1, epochs = 100, verbose= 1) # 피팅 
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
print(model.predict(x_data))
x_new = np.array([[20,30,70],[100,70,30]])
print('예상점수 : ', model.predict(x_new))

#텐서보드

from tensorflow.keras.callbacks import TensorBoard
tb = TensorBoard(
        log_dir = '.\\mylog',
        histogram_freq=True,
        write_graph =True,
             
        
    )

history = model.fit(x_data,y_data,batch_size=1,epochs=1000 , verbose= 1,callbacks=[tb]) #callback 자동호출


plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()




