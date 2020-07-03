# softmax 로 다항분류  - 동물 type 분류

import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
 

xy= np.loadtxt('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/zoo.csv',delimiter=',')
print(xy[0],xy.shape) #(101,17)


x_data = xy[:,0:-1] # feature 마지막행 제외
y_data = xy[:,[-1]] # label # type


print(x_data)
print(y_data[:1],set(y_data.ravel()))

nb_classes = 7  #label 7가지
y_one_hot = to_categorical(y_data,num_classes= nb_classes)
print(y_one_hot[:1]) # 가중치 부여

model = Sequential()
model.add(Dense(32,input_shape=(16,),activation='relu')) # 입력데이터(노드) 보단 출력데이터(유닛)이 더 많도록하자(병목현상 방지)
model.add(Dense(16, activation='relu')) 
model.add(Dense(nb_classes,activation='softmax')) # type의 7가지값 
model.summary()
model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics='accuracy')

history = model.fit(x_data,y_one_hot,epochs=100, batch_size = 10, validation_split=0.2, verbose=2)

print('eval : ' , model.evaluate(x_data,y_one_hot)) 

# 시각화 
hist_dict = history.history
print(hist_dict)
loss = hist_dict['loss']
val_loss = hist_dict['val_loss']
accuracy = hist_dict['accuracy']
vall_accuracy = hist_dict['val_accuracy']


import matplotlib.pyplot as plt 
plt.plot(loss,'b-',label = 'train_loss')
plt.plot(val_loss,'r--',label = 'train_val_loss')

plt.xlabel('epchos')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(accuracy,'b-',label = 'train_accuarcy')
plt.plot(vall_accuracy,'r--',label = 'train_val_accuarcy')

plt.xlabel('epchos')
plt.ylabel('accuarcy')
plt.legend()
plt.show()

print()
pred_data = x_data[:1]
pred = np.argmax(model.predict(pred_data),axis=-1)
print(pred)


pred_datas = x_data[:5]
preds = [np.argmax(i) for i in (model.predict(pred_data))]
print('예측값 들:', preds)
print('실제값 들:', y_data[:5].flatten())

# 새로운 데이터로 
print(x_data[:1])
new_data = np.array([[1.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,12.,0.,0.,0.]]) 
print(np.argmax(model.predict(new_data))) # 5번 동물로 분류 






