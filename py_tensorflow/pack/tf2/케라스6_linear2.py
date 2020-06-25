#모델 세가지 방법으로 시험공부를 한 시간동안 성적이 얼마나 나올지 예측
#단순선형회귀 모델 : 작성방법 3가지
#공부시간에 따른 성적 예측 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np

x_data = np.array([1,2,3,4,5],dtype=np.float32)
y_data = np.array([11,32,53,64,70],dtype=np.float32)
print(np.corrcoef(x_data,y_data))  #상관관계가 매우 높음. [[1.        0.9743547]

#모델 작성 1 : 완전 연결 모델
model = Sequential()
model.add(Dense(1,input_dim = 1,activation='linear'))
# model.add(Dense(1,activation='linear')) #레이어 추가

opti = optimizers.SGD(lr=0.001)  #학습률 0.001
model.compile(opti,loss='mse',metrics='mse')  #mse = mean squared error(평균제곱오차) : 수치가 작을수록 정확성이 높아짐

model.fit(x=x_data, y=y_data, batch_size = 1, epochs=100, verbose=1)

loss_metrics = model.evaluate(x_data, y_data)
print(loss_metrics)

from sklearn.metrics import r2_score
print('설명력 : ',r2_score(y_data,model.predict(x_data)))

print('실제값 : ',y_data)
print('예측값 : ',model.predict(x_data).flatten())
print('새값 예측 : ',model.predict([6.5,2.1]).flatten())

# import matplotlib.pyplot as plt
# plt.plot(x_data,model.predict(x_data),'b',x_data,y_data,'ko')
# plt.show()

#모델 작성 2 : function api를 사용 : 방법1에 비해 유연한 모델을 작성
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
#모델 설계방법만 다르고 나머지 부분은 방법1과 동일
inputs = Input(shape=(1,))  #input 객체를 따로 만듦
# outputs = Dense(1, activation='linear')(inputs) #히든 레이어 1개
output1 = Dense(2, activation='linear')(inputs)
output2 = Dense(2, activation='linear')(inputs)  
outputs = Dense(1, activation='linear')(output1)  #히든 레이어 2개 , 첫번째 레이어에 객체를 줌

model2 = Model(inputs,outputs)

opti = optimizers.SGD(lr=0.001)  #학습률 0.001
model2.compile(opti,loss='mse',metrics='mse') 

model2.fit(x=x_data, y=y_data, batch_size = 1, epochs=100, verbose=1)

loss_metrics = model2.evaluate(x_data, y_data)
print(loss_metrics)

print('실제값 : ',y_data)
print('예측값 : ',model2.predict(x_data).flatten())

print('모델 작성 3-1 --------------------')
#모델 작성 3-1 : sub classing을 사용 : Model을 상속
class MyModel(Model): #모델을 상속받고 클래스 내부에서 레이어를 만듦
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(5,activation='linear')
        self.d2 = Dense(1,activation='linear')  #레이어 2개를 생성자로 만들었다
    
    def call(self, x):  # x : 입력 매개변수 <== 모델.fit(), 모델.evaluate(),모델.predict()
        x = self.d1(x)
        return self.d2(x)
    
model3 = MyModel()    #생성자 호출

opti = optimizers.SGD(lr=0.001)  #학습률 0.001
model3.compile(opti,loss='mse',metrics='mse') 

model3.fit(x=x_data, y=y_data, batch_size = 1, epochs=100, verbose=1)

loss_metrics = model3.evaluate(x_data, y_data)
print(loss_metrics)

print('실제값 : ',y_data)
print('예측값 : ',model3.predict(x_data).flatten())
print('모델 작성 3-2 --------------------')
#모델 작성 3-2 : sub classing을 사용 : Layer 상속
from tensorflow.keras.layers import Layer

class Linear(Layer):
    def __init__(self,units=1):
        super(Linear,self).__init__()
        self.units = units 
    
    def build(self, input_shape):  #call 호출
        self.w = self.add_weight(shape=(input_shape[-1],self.units),initializer='random_normal',trainable=True)  #입력의 크기를 잘 모르겠을 때는 -1을 적음
        self.b = self.add_weight(shape=(self.units),initializer='zeros',trainable=True)
        
    def call(self,inputs):
        return tf.matmul(inputs,self.w) + self.b
    
class MyMLP(Model):
    def __init__(self):
        super(MyMLP,self).__init__()
        self.linear1 = Linear(1)  #이름은 알아서 지어주면 됨. 레이어 1개 
#         self.linear1 = Linear(2)  #레이어 2개
#         self.linear2 = Linear(1)  
    def call(self,inputs):  #linear의 build 호출
#         x = self.linear1(inputs)  #레이어 2개일 경우에는 이렇게 함
#         return self.linear1(x)
        return self.linear1(inputs)  # 레이어 1개이므로 이렇게 해도됨

mlp = MyMLP()

opti = optimizers.SGD(lr=0.001)  #학습률 0.001
mlp.compile(opti,loss='mse',metrics='mse') 

mlp.fit(x=x_data, y=y_data, batch_size = 1, epochs=100, verbose=1)

loss_metrics = mlp.evaluate(x_data, y_data)
print(loss_metrics)

print('실제값 : ',y_data)
print('예측값 : ',mlp.predict(x_data).flatten())

