# constant() : 텐서(일반적인 상수 값) 를 직접 기억 
# variable() : 텐서가 저장된 주소를기억

import tensorflow as tf
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = 10 
print(a,type(a)) # >> int type
print('------')
b= tf.constant(10)
print(b,type(b))
print('------')
c= tf.Variable(10)
print(c,type(c))

print()
#node1 = tf.constant(3.0 , tf.float32)
#node2 = tf.constant(4.0) 
node1 = tf.Variable(3.0 , tf.float32)
node2 = tf.Variable(4.0) #numpy 4.0의 참조변수

print(node1)
print(node2)
node3 = tf.add(node1,node2) # 내부적으로 graph 를 이용해 연산한다.
print(node3)

print('----------------------------')

v = tf.Variable(1)

@tf.function  #autograph function > 속도향상
def find_next_odd():
    abc()   # autograph 지원 함수가 다른 함수를 호출하면 해당 함수도 autograph가 된다.
    v.assign(v+1) #tensor 값은 1+1 롤 
    if tf.equal(v % 2,0 ):
        v.assign(v+10)
        
def abc():
    print('abc')

find_next_odd()
print(v.numpy())

print('~1 ~ 3 까지 숫자 증가 ')

def func():
    imsi = tf.constant(0) # imsi = 0
    su = 1 
    
    for _ in range(3):
        #imsi = tf.add(imsi,su) # 누적
        imsi += su
    return imsi
        
kbs = func()
print(kbs.numpy(), ' ', np.array(kbs)) 

print('-------------')
imsi = tf.constant(0)
def func2():
    su = 1
    global imsi  # > 전역변수 사용 
    for _ in range(3):
        imsi = tf.add(imsi,su)
    return imsi

mbc = func2()
print(mbc.numpy())

print('-------------')

def func3():
    imsi = tf.Variable(0)
    su = 1     
    for _ in range(3):
        #imsi = tf.add(imsi,su)
        imsi.assign_add(su) # assgin 을 이용한 누적 
    return imsi

kbs = func3()
print(kbs.numpy())

print('---------------')

imsi = tf.Variable(0)
@tf.function # decorate functzon 이 있는경우  Variable 변수선언은 함수밖에서 한다.
def func4():
    #imsi = tf.Variable(0) #ValueError    
    su = 1     
    for _ in range(3):
        #imsi = tf.add(imsi,su)
        imsi.assign_add(su) # assgin 을 이용한 누적 
    return imsi

kbs = func4()
print(kbs.numpy())

print('구구단 출력 ---------------')
#@tf.functio
def gugu1(dan):
    su = 0 
    #aa = tf.constant(5)
    #print(aa.numpy()) # autograph 에서는 .numpy() X
    for _ in range(9):
        su = tf.add(su,1)
        print('{} X {} = {:2}'.format(dan,su,dan * su))
        # TypeError: unsupported format string passed to Tensor.__format__
        
gugu1(3)

print('---------------------------')
#@tf.functio >> tensor 의 연산외에는 사용불가능
def gugu2(arg):
    for i in range(1,10):
        result = tf.multiply(arg,i)
        print('{} X {} = {:2}'.format(arg,i,result))
gugu2(5)

        
        
        
        


    

    
  