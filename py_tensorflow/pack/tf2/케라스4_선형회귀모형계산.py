# 선형회귀 모형 계산

import tensorflow as tf
import numpy as np

x = [1.,2.,3.,4.,5.] #feature
y = [1.2,2.0,3.0,3.5,5.5] #

w= tf.Variable(tf.random.normal((1,)))
b= tf.Variable(tf.random.normal((1,)))

opti = tf.keras.optimizers.SGD()

def train_step(x,y):
    with tf.GradientTape() as tape:
        hypo = tf.add(tf.multiply(w,x),b)
        loss = tf.reduce_mean(tf.square(tf.subtract(hypo,y)))
    
    grad = tape.gradient(loss,[w,b])
    opti.apply_gradients(zip(grad,[w,b]))
    return loss

w_val = []
loss_vals = []
for i in range(100):
    loss_val = train_step(x, y)
    loss_vals.append(loss_val.numpy())
    w_val.append(w.numpy())
    #print(loss_val)
    
print(w_val)
print(loss_vals)
    
import matplotlib.pyplot as plt
plt.plot(w_val,loss_vals,'o')

plt.show()



    