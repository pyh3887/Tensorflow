from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,LSTM


model = Sequential()

model.add(SimpleRNN(3,input_length=2, input_dim = 10)) # 2행 10열 자료로 출력 3개 수행 

model.add(LSTM(3,input_length=2))

print()

model = Sequential()
model.add(LSTM(3,batch_input_shape=(8,2,10)))
print(model.summary())

print()

from tensorflow.keras.layers import Dense
model = Sequential()
model.add(LSTM(3,batch_input_shape=(8,2,10),return_sequences=True))
model.add(Dense(2,activation='softmax'))
print(model.summary())
        