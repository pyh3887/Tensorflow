# 계단 함수 : 거의 사용되지는 않지만, 퍼셉트론을 통해 처음으로 인공 신경망을 배울 때 가장 처음 접하게 되는 활성화 함수

import numpy as np 

import matplotlib.pyplot as plt 



def step_func(x):

    print(x)

    return np.array(x > 0, dtype=np.int)



x = np.arange(-5.0, 5.0, 0.1)    # -5.0부터 5.0까지 0.1 간격 생성

y = step_func(x)

print(y)



plt.title('Step Function')

#plt.scatter(x,y)

plt.plot(x,y)

plt.show()



print('***' * 20)

# 시그모이드 함수 : 가중치 곱의 합을 0과 1사이의 값으로 조정하여 반환하는 활성화 함수

def sigmoid_func(x):

    return 1 / (1 + np.exp(-x))



x = np.arange(-5.0, 5.0, 0.1)

y = sigmoid_func(x)

print(y)

plt.plot(x, y)

plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가

plt.title('Sigmoid Function')

plt.show()



print('---' * 20)

# 렐루 함수 : 은닉층에서 활성화 함수로 가장 많이 사용되는 활성화 함수.

def relu_func(x):

    return np.maximum(0, x)



x = np.arange(-5.0, 5.0, 0.1)

y = relu_func(x)

print(y)

plt.plot(x, y)

plt.plot([0,0],[5.0,0.0], ':')

plt.title('Relu Function')

plt.show()



print('~~~' * 20)

# 하이퍼볼릭탄젠트 함수 : 은닉층에서 활성화 함수로 자주 사용되는 활성화 함수. 

# 이미지 인식 분야에서 사용되는 CNN에서는 ReLu 함수가 주로 사용되고, 

# 자연어 처리 분야에서 사용되는 LSTM에서는 tanh 함수와 Sigmoid 함수가 주로 사용됨.

x = np.arange(-5.0, 5.0, 0.1) 

y = np.tanh(x)

print(y)

plt.plot(x, y)

plt.plot([0,0],[1.0,-1.0], ':')

plt.axhline(y=0, color='orange', linestyle='--')

plt.title('Tanh Function')

plt.show()



print('^^^' * 20)

# 소프트맥스 함수 : MultiClass Classification 에서 주로 사용

x = np.arange(-5.0, 5.0, 0.1) 

y = np.exp(x) / np.sum(np.exp(x))

print(y)

plt.plot(x, y)

plt.title('Softmax Function')

plt.show()