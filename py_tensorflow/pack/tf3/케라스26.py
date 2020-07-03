# RNN  이해를 위해 4개의 숫자를 입력하고 그다음 숫자 예측하기 

import tensorflow as tf 
import numpy as np 

x = [] # sequence data 기억장소 

y = [] 

for i in range(6):
    lst = list(range(i,i+4))
    x.append(list(map(lambda c:[c/10],lst)))
    y.append((i+4)/10)
    
x = np.array(x)
y = np.array(y)
print(x,np.shape(x))
print(y)

for i in range(len(x)):
    print(x[i],y[i])
    
print()
model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(units=10,activation='tanh',input_shape=[4,1]),
        tf.keras.layers.Dense(1)
    ])

model.compile(optimizer='adam', loss = 'mse')
model.summary()
model.fit(x,y,epochs=100,verbose=0)
print('예측값:' ,model.predict(x).flatten())
print('실제값: ' , y)



