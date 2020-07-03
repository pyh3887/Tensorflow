import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from tensorflow.keras import Model

(x_train, y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(x_train.shape,' ',y_train.shape)    #(60000, 28,28)    (60000,)
print(x_test.shape,' ',y_test.shape)      #(10000,28,28)      (10000,)
# print(x_train[0])
# print(x_train[0],set(x_train))

# for i in x_train[0]:
#     for j in i:
#         sys.stdout.write('%s '%j)
#     sys.stdout.write('\n')
    
# print(y_train[0])
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0].reshape(28,28),cmap = 'Greys')
# plt.show()

#3차원을 4차원으로 구조 변경. 흑백(1) 칼라(3) 여부 확인용 채널 추가
x_train = x_train.reshape((60000,28,28,1))
print(x_train.ndim)  #차원 확인 가능
x_train = x_train / 255.0    #정규화
print(x_train[:1])

x_test = x_test.reshape((10000,28,28,1))  #구글 이외의 제품일 경우 채널의 위치가 다를 수 있다. 10000,1,28,28
print(x_test.ndim)  #차원 확인 가능
x_test = x_test / 255.0    #정규화
print(x_test[:1])


#데이터 섞기
import numpy as np
np.random.seed(1)
x= np.random.sample((5,2))
print(x)
dset =tf.data.Dataset.from_tensor_slices(x)
# print(dset)
dset = tf.data.Dataset.from_tensor_slices(x).shuffle(10000).batch(5)
print(dset)
for a in dset:
    print(a)
#--------------------------------------------

#train data를 shuffle
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(28)
test_ds = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(28)
print(train_ds,' ',test_ds)

#subclassing api로 모델 생성
class MyModel(Model):
    def __init__(self):
        super(MyModel,self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size = [3,3], padding='valid',activation='relu')
        self.pool1 = MaxPool2D((2,2))
        
        self.conv2 = Conv2D(filters=32, kernel_size = [3,3], padding='valid',activation='relu')
        self.pool2 = MaxPool2D((2,2))
        
        self.flatten = Flatten(dtype='float64')
        
        self.d1 = Dense(64, activation='relu')
        self.drop1 = Dropout(rate=0.3)
        self.d2 = Dense(10, activation='softmax')
        
    def call(self, inputs):  #init에서 선언한 레이어를 불러와 network 구성
        net = self.conv1(inputs)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.flatten(net)
        net = self.d1(net)
        net = self.drop1(net)
        net = self.d2(net)
        return net
        
model = MyModel()
temp_input = tf.keras.Input(shape=(28,28,1))
model(temp_input)
model.summary()  #설정된 구조 확인

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
'''
model.compile(optimizer=optimizer,loss=loss_obj, metrics=['acc'])

model.fit(x_train,y_train,batch_size=128, epochs=3, verbose=2,max_queue_size=10,workers=4, use_multiprocessing=True)
score = model.evaluate(x_test,y_test)
print('test loss : ',score[0])
print('test acc : ',score[1])

print('예측값 : ',np.argmax(model.predict(x_test[:1]),1))
'''


#GradientType을 사용해 모델 학습 방법: model.compile, model.fit을 대신해 아래와 같이 기술
train_loss = tf.keras.metrics.Mean() #Computes the (weighted) mean of the given values.
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()  #정확도 계산
test_loss = tf.keras.metrics.Mean() #Computes the (weighted) mean of the given values.
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()  #정확도 계산

def train_step(images,labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_obj(labels,predictions)
    gradients = tape.gradient(loss, model.trainable_variables) #loss를 최소화하기 위한 미분값을 계산
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss) #가중치 평균 계산
    train_accuracy(labels,predictions)

@tf.function
def test_step(images,labels):
    predictions = model(images)
    loss = loss_obj(labels,predictions)
    
    test_loss(loss)
    test_accuracy(labels, predictions)
    

EPOCHS = 3
for epoch in range(EPOCHS):
    for x_train, y_train in train_ds:
        train_step(x_train, y_train)
        
    for x_test, y_test in test_ds:
        test_step(x_test, y_test)
        
    kbs = 'epochs:{}, train_loss:{}, train_acc:{}, test_loss:{}, test_acc:{}'
    print(kbs.format(epoch + 1,train_loss.result(), train_accuracy.result() * 100,
                     test_loss.result(), test_accuracy.result() * 100))
    


print('예측값 : ',np.argmax(model.predict(x_test[:1]),1))

print('실제값 : ',y_test[:1].numpy())

