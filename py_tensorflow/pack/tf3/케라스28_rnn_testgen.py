# RNN으로 텍스트 생성 : 기존 문서를 반영하여 다음 단어를 예측하고, 텍스트를 생성 

from tensorflow.keras.layers import Embedding,Dense,SimpleRNN,LSTM
from tensorflow.keras.models import Sequential
import numpy as np 
from keras_preprocessing.text import Tokenizer

text = '''경마장에 말이 뛰고 있다 
그의 말이 법이다 
가는 말이 고와야 오는 말이 곱다
'''

tok = Tokenizer()
tok.fit_on_texts([text])
encoded = tok.texts_to_sequences([text])[0]

print(tok.word_index)

vocab_size = len(tok.word_index)
print('단어집합의 크기 : %d'%vocab_size)

# train data 
sequences = list()
for line in text.split('\n'):
    