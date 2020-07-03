# 다항분류 : softmax 활성화 함수 사용 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np 
import matplotlib.pyplot as plt 


np.random.seed(1)

xdata = np.random.random((1000,12))
ydata = np.random.randint(10,size = (1000,1))
ydata = to_categorical(ydata, num_classes= 10)
print(xdata[:2],xdata.shape)
print(ydata[:2],ydata.shape) #레이블 10개

model= Sequential()

model.add(Dense(100,input_shape = (12,), activation='relu')) # 입력 12개 > 출력 100개  
model.add(Dense(50,activation='relu')) # 레이어 추가 100 > 50 > 10
model.add(Dense(10,activation='softmax')) #softmax > 확률값  

print(model.summary()) 
model.compile(optimizer = 'adam', loss ='categorical_crossentropy', metrics=['acc'])

hist = model.fit(xdata,ydata,epochs=500,batch_size = 32 , verbose=2)

model_eval = model.evaluate(xdata,ydata) #모델의 성능평가

print('model_eval : ' , model_eval) # model_eval :  [0.09353379160165787, 0.9990000128746033] > 정확도 0.99

print('예측값 : ',[np.argmax(i) for i in (model.predict(xdata[:5]))])
print('실제값 : ', [np.argmax(i) for i in ydata[:5]])


# 시각화 

plt.plot(hist.history['loss'])
plt.plot(hist.history['acc'])

plt.show()

# 새로운 값 예측 

x_new = np.random.random((1,12))
print(x_new)
pred = model.predict(x_new)
print('pred 합 : ', np.sum(pred))
print(pred)
print(np.argmax(pred))















 












