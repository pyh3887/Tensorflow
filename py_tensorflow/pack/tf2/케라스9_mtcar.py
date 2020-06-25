# keras 로 자동차 연비 예측

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow as tf 
from tensorflow.keras import layers

dataset = pd.read_csv('../testdata/auto-mpg.csv')

print(dataset.head(2))

del dataset['car name']  # 필요없는 칼럼 삭제 

print(dataset.head(2))

# 강제형 변환 valueError 를 무시하기 errors = 'coerce
dataset['horsepower'] = dataset['horsepower'].apply(pd.to_numeric, errors='coerce')

dataset = dataset.dropna()  # nan값 날림 
print(dataset.isna().sum())

# 시각화 

sns.pairplot(dataset[['mpg', 'weight', 'horsepower']], diag_kind='kde')
plt.show()

train_dataset = dataset.sample(frac=0.7, random_state=0)

test_dataset = dataset.drop(train_dataset.index)

print(train_dataset.shape)  # (279,8)
print(test_dataset.shape)  # (118,8) > test한 데이터 118개의 데이터 랜덤 추출 

train_stat = train_dataset.describe()
train_stat.pop('mpg')
train_stat = train_stat.transpose()
print(train_stat)

# label : 'mpg'
train_labels = train_dataset.pop('mpg')
print(train_labels[:2])
test_labels = test_dataset.pop('mpg')
print(train_labels[:2])


def st_func(x):  # 표준화 처리 함수 
    return ((x - train_stat['mean']) / train_stat['std'])
    
# print('st_func(10):' ,st_func(10))


st_train_data = st_func(train_dataset)
st_test_data = st_func(test_dataset)

print('-------------')


# 모델 작성 후 예측 
def build_model():
    model = tf.keras.Sequential([
        layers.Dense(units=64, activation=tf.nn.relu, input_shape=[7]),  # carname과 mpg가 빠짐
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear'),     
        
        ])

    # opti = tf.keras.optimizers.RMSprop(0.001)
    opti = tf.keras.optimizers.Adam(0.01)
    
    model.compile(optimizer=opti , loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])  # mean_squared_error , mean_absolute_error
    return model


model = build_model()
print(model.summary())

# fit() 전에 모델을 실행해 볼수도 있다.
print(model.predict(st_train_data[:1]))  # 결과는 신경쓰지 않음

# 모델 훈련 
epochs = 2000

#학습 조기 종료 
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(st_train_data, train_labels, epochs=epochs, validation_split=0.2, verbose=1,callbacks=[early_stop])

df = pd.DataFrame(history.history)
print(df.head(3))
print(df.columns)

# from Ipython.display import display jupyter 에서 실행하면 칼럼명 모두 보임 

import matplotlib.pyplot as plt

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure(figsize=(8, 12))
    plt.subplot(2, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
    plt.ylim([0, 5])
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


plot_history(history)

#모델 평가 

loss, mae, mse = model.evaluate(st_test_data, test_labels)
print('test dataset으로 모델을 평가 mae',mae)
print('test dataset으로 모델을 평가 mse',mse)
print('test dataset으로 모델을 평가 loss',loss)


#예측 : 주의 - 새로운 데이터로 예측을 원한다면 표준화 작업을  선행

# 데이터 분포와 모델에 의한 선형회귀선

test_pred = model.predict(st_test_data).flatten()
print(test_pred)

plt.scatter(test_labels,test_pred)
plt.xlabel('True value[mpg]')
plt.ylabel('pred value[mpg]')
plt.show()

# 오차 분포 확인 (정규성: 잔차항이 정규분포를 따라야함)

err = test_pred
plt.hist(err, bins = 20)
plt.xlabel('pred error[mpg]')
plt.show()













