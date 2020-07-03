# 이미지 보강 

import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt 
import numpy as np 
import os 
from sympy.matrices.densetools import augment

np.random.seed(0)
tf.random.set_seed(0)

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255
x_test = x_test.reshape(-1,28,28,1).astype('float32') / 255
print(x_train.shape,' ',x_train[:1])
print(y_train[:5])
y_train = to_categorical(y_train)
print(y_train[:5])
y_test = to_categorical(y_test)

# 이미지 보강 예) 기존 이미지를 좌우대칭 ,약간 회전 , 기울기 , 확대/축소 , 평행 이동 등의 작업을 통해 다양한 이미지로 모델 학습  
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_generate = ImageDataGenerator(
        rotation_range =10,
        zoom_range = 0.1,
        shear_range= 0.5,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
        vertical_flip= True,        
    )

'''
print(img_genrate)
augument_size = 100
x_augment = img_generate.flow(np.tile(x_train[0].reshape(28,28),100).reshape(-1,28,28,1),
                             np.zeros(augument_size),
                             batch_size= augument_size,
                             shuffle=False).next()[0]
                             
print(x_augment)

plt.figure(figsize=(10,10))
for c in range(100):
    plt.subplot(10,10,c + 1)
    plt.axis('off')
    plt.imshow(x_augment[c].reshape(28,28),cmap='gray')
    
plt.show()
'''

augment_size = 30000
randidx = np.random.randint(x_train.shape[0], size = augment_size)
x_augment = x_train['randidx'].copy()
y_augment = y_train[randidx.copy()]

x_augment = img_generate.flow(x_augment,np.zeros(augment_size),batch_size=augment_size,shuffle=False).next()[0]

# 원래 x_train에 image augment된 x_augment 를 추가 

x_train = np.concatenate((x_train,x_augment))
y_train = np.concatenate((y_train,y_augment))
print(x_train.shape,' ', y_train.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1),
                           padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size =(2,2)),
    tf.keras.layers.Dropout(rate=0.3),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(units=128,activation='relu'),
    tf.keras.layers.Dropout(rate=0.25),
    tf.keras.layers.Dense,
    tf.keras.layers.Dropout,
    tf.keras.layers.Dense,
       
    
    ])

model.compile(optimizer = 'adam', loss= 'categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

#모델 최적화 설정 

MODEL_DIR = './mymnist/'

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
    


modelpath = "./mymnist/{epoch:02d}-{val_loss:.4f}.hdf5"

chkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss',verbose=1,save_best_only=True)

earlystop = EarlyStopping(monitor='val_loss',patience=3)

#train 
history = model.fit(x_train,y_train,validation_split=0.2)


#시각화 

plt.figure(figsize=(12.4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],marker='o',c='red',label='Test acc')
plt.plot(history.history['val_accuracy'],marker='+',c='blue', label='vali_acc')
plt.xlabel('epchos')
plt.ylim(0.5,1)
plt.legend(loc='lower right')


plt.figure(figsize=(12.4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],marker='o',c='red',label='Test acc')
plt.plot(history.history['val_loss'],marker='+',c='blue', label='vali_acc')
plt.xlabel('epchos')
plt.legend(loc='upper right')








