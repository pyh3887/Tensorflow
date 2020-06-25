# tf 변수 구조

import tensorflow as tf

g1 = tf.Graph()

with g1.as_default(): # 특정 자원ㅇ르 처리하고 자동 close
    c1 = tf.constant(1,name='c_one')
    print(c1)
    print(type(c1))
    print(c1.op)
    print(g1.as_graph_def())
    
print(c1)

print('----------------------')

g2 = tf.Graph()

with g2.as_default():
    v1 = tf.Variable(initial_value = 1 ,name = "v1")
    print(v1)
    print(type(v1))
    print(v1.op)
    print(g2.as_graph_def())