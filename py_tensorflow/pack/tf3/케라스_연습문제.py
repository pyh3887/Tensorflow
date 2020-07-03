
import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd
from tensorflow.python.keras.layers.core import Activation, Dropout




data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/bmi.csv',delimiter=',')


data['height'] /= 200 # 정규화 (미작업시 분류 정확도가 현저히 낮아짐 )
data['weight'] /= 100
print(data)
x = data[['height','weight']].values # feature 마지막행 제외

# label은 원핫인코딩                  왜 ?? 
bclass = {'thin':[1,0,0],'normal':[0,1,0],'fat':[0,0,1]}
y = np.empty((20000,3))

for i, v in enumerate(data['label']):
    y[i] = bclass[v]

print(y)
print('---------')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test =train_test_split(x,y,test_size= 0.3 , random_state=0)
print(x_train.shape, x_test.shape, y_train.shape,y_test.shape)


#model 
model = Sequential()
model.add(Dense(128, input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dropout(0.2)) # 20% 데이터는 학습에서 제외(과적합 방지)
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(3)) # 출력 레이어
model.add(Activation('softmax'))

print(model.summary())
model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])

# model train 
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',mode='min',baseline=0.05, patience=5) # baseline 0.05 보다 작은 값이 연속적으로 5회 이상 나오면 학습 조기 종료
model.fit(x_train,y_train, batch_size= 64, epochs= 1000, validation_split=0.2,verbose=2, callbacks= [es])

#model 평가

m_score = model.evaluate(x_test,y_test)
print('loss :' ,m_score[0])
print('accuracy :' ,m_score[1]) 

#predict 

print('예측값 : ', np.argmax(model.predict(x_test[:1])))
print('실제값 :' , y_test[:1],np.argmax(y_test[:1]))

#new data 


print('예측값 : ', np.argmax(model.predict(np.array([[187/200,55/100]])),axis=1))
print('예측값 : ', np.argmax(model.predict(np.array([[167/200,75/100]])),axis=1))