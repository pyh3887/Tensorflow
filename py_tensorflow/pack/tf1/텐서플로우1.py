from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

import tensorflow as tf

print(tf.__version__) #2.2.0

print(tf.keras.__version__) # 2.3.0 -tf

print('GPU 사용 가능 여부 :','가능' if tf.config.list_physical_devices('GPU') else '불가능')
# tensor 의 이해 : tensorflow의 기본 구성 요소. 데이터를 위한 컨테이너의 별개로 

# 임의의 차원 갯수를 가지는 행렬의 일반화된 객체.
# 상수정의 (상수 텐서를 생성)

print(tf.constant(1)) # scala 0차원 텐서
print(tf.constant([1])) #vector 1차원 텐서
print(tf.constant([[1]])) #matrix 2차원 텐서

print(tf.rank(tf.constant(1,)), ' ', tf.rank(tf.constant([[1]])))
print(tf.constant(1.).get_shape(), ' ', tf.constant([[1]]).get_shape())

print()

a= tf.constant([1,2])
b= tf.constant([3,4])

c = a+b
print(c,type(c))

print()
#d = tf.constant([3])
#d = 3 #텐서로 변환
d= tf.constant([[3]])
e= c+d # Broadcast 연산 

print(e)

#상수를 텐서화 
print(7)
print(tf.convert_to_tensor(7))
print(tf.convert_to_tensor(7,dtype=tf.float32))
print(tf.cast(7, dtype=tf.float32))

# numpy 의 ndarray와 tensor 사이에 자동 변환 
import numpy as np 

arr = np.array([1,2])
print(arr,' ', type(arr))
tfarr = tf.add(arr,5)
print(tfarr)
print(tfarr.numpy(),' ', type(tfarr))
print(np.add(tfarr,3))











