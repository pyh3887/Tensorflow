# 와인의 맛 , 등급 , 산도 등을 측정해 레드와 화이트 와인 분류 모델 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import pandas as pd
import numpy as np 
import os 
import matplotlib.pyplot as plt
import tensorflow as tf

# seed 값 고정 

np.random.seed(3)
tf.random.set_seed(3)


wdf = pd.read_csv('../testdata/wine.csv')
df = wdf.sample(frac=0.5)
print(df.head(2))
print(df.info())
print(df.iloc[:,12].unique()) #[0 1]
dataset = df.values

x= dataset[:,0:12] # feature
y = dataset[:,-1] # label(class)

print(x,'\n')
print(y)
# 모델 설정

model = Sequential()

model.add(Dense(30,input_dim=12,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid')) #마지막 값은 sigmoid 

# model.add(Dense(30,input_dim=12,activation='elu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Dense(12,activation='elu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Dense(8,activation='elu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(Dense(1,activation='sigmoid')) #마지막 값은 sigmoid 


model.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics=['accuracy'])


# 모델 저장 폴더 설정 

# fit() 이전의 훈련되지 않은 모델 정확도 

loss, acc = model.evaluate(x,y,verbose = 0)
print('훈련되지 않은 모델 정확도 :{:5.2f}%'.format(acc * 100)) 

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
    
modelpath = 'model/{epoch:02d}-{val_loss:4f}.hdf5'

# 모델 학습시 모니터링 결과를 파일로 저장 

chkpoint =ModelCheckpoint(filepath=modelpath,monitor='val_loss',verbose=1,set_test_only = True) # 학습을 할때마다 

# 학습 조기종료 

early_stop = EarlyStopping(monitor='val_loss',patience= 5)


# 모델 실행 

history = model.fit(x,y,validation_split=0.3, epochs=100 , batch_size = 128,callbacks=[early_stop,chkpoint]) # 모델을

loss, acc = model.evaluate(x,y,verbose= 0)
print('훈련된 모델 정확도 :{:5.2f}%'.format(acc * 100))

# 시각화

y_vloss = history.history['val_loss']
y_acc = history.history['val_accuracy']


x_len = np.arange(len(y_acc))
plt.plot(x_len,y_vloss,'o',c='red', ms=3) #오차
plt.plot(x_len,y_acc,'x',c='blue', ms=3) # 정확도
plt.show()














