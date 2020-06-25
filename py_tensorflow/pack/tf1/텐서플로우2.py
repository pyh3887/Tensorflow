# 변수 
import tensorflow as tf
import numpy as np

f = tf.Variable(1.0) # 변수형 텐서에 scala 값 기억 
v = tf.Variable(tf.ones((2,))) # 변수형 텐서에 vector 값 기억 
m = tf.Variable(tf.ones((2,1))) # 변수형 텐서에 matrix 값 기억

print(f)
print(v,v.numpy())
print(m)


print()
v1 = tf.Variable(1)
print(v1)
v1.assign(10)
print(v1,' ',v1.numpy(), ' ', type(v1))

print()

v2 = tf.Variable(tf.ones(shape=(1))) # 1차원 텐서
v2.assign([20]) # 1차원 텐서이므로 배열값 할당
print(v2 , ' ', type(v2))

v3 = tf.Variable(tf.ones(shape=(1,2))) # 2차원 텐서
v3.assign([[30,40]]) # 2차원 텐서이므로 배열값 할당  차원을 맞추어야함
print(v3 , ' ', type(v3))

print()

v1 = tf.Variable([3])
v2 = tf.Variable([5])
v3 = v1 * v2 + 10
print(v3.numpy())

var = tf.Variable([1,2,3,4,5], dtype= tf.float32)
result1 = var + 10
print(result1)

w = tf.Variable(tf.ones(shape=(1,)))
b = tf.Variable(tf.ones(shape=(1,)))

w.assign([3])
b.assign([2])
def func1(x):   #python 함수
    return w*x+b

out_a1 = func1([3])
print('out_a1:' , out_a1)

print()
w = tf.Variable(tf.zeros(shape=(1,2)))
b = tf.Variable(tf.zeros(shape=(1,)))
w.assign([[2,3]])

@tf.function   # autograph 기능 (내부적으로 tf.Graph + tf.Session) : 속도가 빨라짐
def func2(x):
    return w*x+b

out_a2 = func2([3])
print('out_a2 :' ,out_a2)

print('-------------------------')
w = tf.Variable(tf.keras.backend.random_normal([5,5], mean=0 , stddev=0.3)) #난수 발생 
print(w.numpy().mean())
print(np.mean(w.numpy()))
print(w)

b = tf.Variable(tf.zeros([5]))
print(b * w)

print()
rand1 = tf.random.normal([4], 0 , 1) #정규분포를 따르는 텐서 # 평균 , 표준편ㄴ차  
print('rand1:', rand1)

rand2 = tf.random.uniform([4], 0 , 1) #정규분포를 따르는 텐서 #최소 0 최대 1 
print('rand2:', rand2)

aa = tf.ones((2,1)) #2행1열 1로 채움
print(aa.numpy())

m = tf.Variable(tf.zeros((2,1)))
m.assign(aa) #치환 
print(m.numpy())

m.assign(m+10)
print(m.numpy)

m.assign_add(aa) #aa 배열 더하기
print(m.numpy()) 

m.assign_sub(aa) #aa 배열 빼기
print(m.numpy())
