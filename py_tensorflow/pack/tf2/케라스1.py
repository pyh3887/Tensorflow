import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD,Adam,RMSprop


#논리회로 (OR gate) 모델 작성 

#1) 데이터 수집 및 가공 

x = np.array([[0,0],[0,1],[1,0],[1,1]]) #feature 2차원 
y = np.array([0,1,1,1]) #논리합

print(x)
print(y)

# 2) 모델 생성 (설정)

# 방법1
model = Sequential([
    Dense(input_dim = 2,units=1), #하나의 레이어에 뉴런을 하나씀 활성함수는 sigmoid를 사용 모델생성
    Activation('sigmoid')
    
    ])

# 방법2
model = Sequential()
model.add(Dense(1,input_dim=2))
model.add(Activation('sigmoid'))


#3) 학습 process 생성(컴파일)
#model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=SGD(lr=0.1), loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=SGD(lr=0.1,momentum=0.8), loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=RMSprop(lr=0.1), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(lr=0.1), loss='binary_crossentropy', metrics=['accuracy'])


#4) 모델 학습
model.fit(x,y,epochs=100,batch_size=1,verbose=1) # epochs = 훈련횟수 , verbose=1 진행과정이 보임(속도저하)

#5) 모델 평가
loss_metrics = model.evaluate(x,y) 
print(loss_metrics) # 분류 정확도 [0.75]

#6) 예측값 출력 

pred = model.predict(x)
print('pred : ', pred)
#pred2 = (model.predict(x) > 0.5).astype('int32')
#print('pred: ', pred2)
#pred3 = model.predict.classes(x)
#print('pred3 :', pred3)

#loss 값이 떨어질수록 정확도 올라감

#모델 저장 
model.save('test.hdf5s')
del model

#모델 읽기 
from tensorflow.keras.models import load_model
model2 = load_model('test.hdf5')









 

