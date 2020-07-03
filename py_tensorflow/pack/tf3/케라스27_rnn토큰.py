# 텍스트의 토큰화 


from keras_preprocessing.text import Tokenizer
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers.recurrent import LSTM

samples = ['The cat say on the mat.','The dog ate my homework']
#직접 토큰분리 
token_index = {}
for sam in samples:
    for word in sam.split():
        if word not in token_index:
            #print(word)
            token_index[word] =len(token_index)
            

print(token_index)


print()

# Tokenizer 로 분리 

tokenizer = Tokenizer()
tokenizer.fit_on_texts(samples)
token_seq = tokenizer.texts_to_sequences(samples) #텍스트 정수 인덱싱
print(token_seq)

print()
token_mat = tokenizer.texts_to_matrix(samples, mode='binary')#2진 mode ='binary','count','tfidf' 
print(token_mat)
word_index = tokenizer.word_index
print(word_index)
print('found %s unique tokens'%(len(word_index))) #found 9 unique tokens
print(tokenizer.word_counts)
print(tokenizer.document_count)
print(tokenizer.word_docs)


print()
docs = [
    '먼저 ㅓ텍스트의 각 단어를 나누어 토큰화 한다.'
    '텍스트의 단어로 토큰화 해야 딥러닝에서 인식된다.',
    '토큰화 한 결 과는 딥러닝에서 사용할수 있다'
    ]

token = Tokenizer()
token.fit_on_texts(docs)
print('단어 카운트:', token.word_counts)
print('문장 카운트:', token.document_count)
print('각 단어가 몇개의 문장에 포함되어 있는가 :', token.word_docs)
print('각 단어에 매겨진 인덱스 값 :', token.word_index)

print()
# 텍스트를 읽고 긍정 , 부정 분류 예측 

docs = ['너무 재밌네요', '최고에요','참 잘만든 영화예요','추천하고 싶은 영화네요','한번 더 보고싶네요',
        '글쎄요','별로네요','생각보다 지루합니다','연기가 좋지않아요','재미없어요']

import numpy as np 
classes = np.array([1,1,1,1,1,0,0,0,0,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

model = Sequential()
model.add(Embedding(word_size,8,input_length=4))
#model.add(Flatten())
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))

print(model.summary())
model.compile(optimizer='adam',loss='binary_crossentropy')




