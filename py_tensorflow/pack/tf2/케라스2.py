# 복수레이어 (deep learning)

import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD,Adam,RMSprop


#논리회로 (XOR gate) 모델 작성 

#1) 데이터 수집 및 가공 

x = np.array([[0,0],[0,1],[1,0],[1,1]]) #feature 2차원 
y = np.array([0,1,1,0]) #논리곱 [[0],[1],[1],[0]]

print(x)
print(y)

# 2) 모델 생성 (설정)

# 방법1
# model = Sequential([
#     Dense(input_dim = 2,units=5), #하나의 레이어에 뉴런을 하나씀 활성함수는 sigmoid를 사용 모델생성
#     Activation('relu'),
#     Dense(units=1), #하나의 레이어에 뉴런을 하나씀 활성함수는 sigmoid를 사용 모델생성
#     Activation('relu'),
#     
#     ])

# 방법2
model = Sequential()
#model.add(Dense(1,input_dim=2))
#model.add(Activation('relu'))

model.add(Dense(5,input_dim = 2,activation ='relu'))
model.add(Dense(5,activation ='relu'))
model.add(Dense(1,activation ='sigmoid')) #layer 두개사용

model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['acc'])

#print(model.summary())

history = model.fit(x,y,epochs=500,batch_size=1,verbose=2) # epochs = 훈련횟수 , verbose=1 진행과정이 보임(속도저하)

loss_metrics = model.evaluate(x,y) 
print(loss_metrics) # 분류 정확도 [0.75]

print('------------')
print(model.weights)  #dense/kernel , bias 확인
print('************')
print(history.history)
print('loss:', history.history['loss'])
print('acc :',history.history['acc'])

import matplotlib.pyplot as plt
plt.plot(history.history['loss']) #loss 값은 떨어지고 acc값은 올라감 
plt.plot(history.history['acc'])
plt.xlabel('epoch')
plt.show()



