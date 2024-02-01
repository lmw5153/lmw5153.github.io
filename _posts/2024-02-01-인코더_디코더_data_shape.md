# 인코더-디코더 언어번역 lstm 데이터 shape 이해
---
- 언어번역 인코더-디코더 형식의 lstm 모델
- fitting 하기 위해선 3가지의 data가 필요하다
    - encoder input data
    - decoder input data
    - decoder output data
- 데이터 차원을 중심으로 어떻게 input과 output의 data shape이 구성되는지 확인해보자


---

### `-` import


```python
import numpy as np
import pandas as pd
from keras.layers import LSTM ,Dense, Bidirectional, Input, TimeDistributed
from keras.models import Sequential ,Model
from keras.callbacks import EarlyStopping
import keras.backend as K
```

---

### `-` 순서

1. data의 length
2. sequence의 length
3. feature의 개수
4. summary
5. model fitting

---

### `-` data의 length

`-` 예제


```python
# 예제 데이터"
input_data = ["Hello.", "How are you?", "What is your name?", "I'm hungry.", "How old are you?"]
target_data = ["안녕하세요.", "잘 지내니?", "너의 이름이 뭐니?", "나 배고파.", "너는 몇 살이니?"]
```


```python
input_data[0], target_data[0]
```




    ('Hello.', '안녕하세요.')



- 번역하고 싶은 문장은 영어로된 input data이고 번역된 문장이 target data이다
- 각 문장을 위와 같이 각각 대응시켜 학습하기 위해선 인코더와 디코더의 train data length가 일치해야한다. 마찬가지로 디코더의 test를 위한 data 또한 길이가 같아야한다

**하지만 각 문장이 요구하는 문장의 길이와 글자의 개수는 다를 수 밖에 없다**

---

### `-` sequence의 최대 길이

- 따라서 input data와 target data에 필요한 sequence의 최대 길이와 글자의 개수를 구해야함
- 언어마다 같은 의미를 지니어도 문장의 길이는 보통 달라질 수밖에 없기 때문


```python
max_encoder_seqlen = max([len(txt) for txt in input_texts])
max_decoder_seqlen = max([len(txt) for txt in target_texts])
```


```python
print("인코더 문장 최대 길이 :",max_encoder_seqlen ,"\n디코더 문장 최대 길이 :",max_decoder_seqlen)
```

    인코더 문장 최대 길이 : 18 
    디코더 문장 최대 길이 : 10
    

---

### `-` feature의 개수

- 이는 각 언어의 쓰여진 특성, 글자라는 기호가 중복되지 않고 얼마나 쓰였는지를 물어보는 것이다
- 마찬가지로 언어마다 필요한 글자 기호의 개수는 당연히 다르기 때문에 이또한 인코더와 디코더에서 다르게 입력된다


```python
# data 각 글자 집합 
input_set = set(" ".join(input_texts)) # 원래문장
target_set = set(" ".join(target_texts)) # 번역문장

# 각 글자에 대한 숫자 부여
input_token = dict([(char, i) for i, char in enumerate(input_set)])
target_token = dict([(char, i) for i, char in enumerate(target_set)])

# 데이터의 중복되지 않는 총 글자수
encoder_text_len = len(input_set)
decoder_text_len = len(target_set)
```


```python
print("인코더 글자 개수 :",encoder_text_len ,"\n디코더 글자 개수 :",decoder_text_len)
```

    인코더 글자 개수 : 23 
    디코더 글자 개수 : 24
    

---

### `-` 정리


```python
# 원핫 인코딩 zero 필드
encoder_inputdata= np.zeros((len(input_texts), max_encoder_seqlen, encoder_text_len ), dtype='float32') # encoder 입력데이터
decoder_inputdata= np.zeros((len(input_texts), max_decoder_seqlen, decoder_text_len ), dtype='float32') # decoder 입력데이터
decoder_target_data = np.zeros((len(input_texts), max_decoder_seqlen, decoder_text_len), dtype='float32') #encoder 출력데이터

# 원핫인코딩
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_inputdata[i, t, input_token[char]] = 1.0
    for t, char in enumerate(target_text):
        decoder_inputdata[i, t, target_token[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token[char]] = 1.0
            
print("인코더 input shape ",encoder_inputdata.shape,"\n디코더 input shape ",decoder_inputdata.shape,"\n디코더 target shape",decoder_target_data.shape)
```

    인코더 input shape  (5, 18, 23) 
    디코더 input shape  (5, 10, 24) 
    디코더 target shape (5, 10, 24)
    


- 3가지의 차원이 필요하다.
- data length, data sequence maximum length, feature(data의 중복되지 않은 글자 개수)
- 그중에서도 data의 length는 인코더 인풋과 디코더 인풋이 서로 일치해야한다
    - 번역을 하기 위해서 번역하고자하는 문장을 번역이 된 문장과 대응시켜야한다.
    - 따라서 data length를 맞춰야함
- sequence max length와 feature는 인코더 인풋과 디코더 인풋이 서로 다를 수 있음
    - 언어번역이기 때문에 각 언어마다 같은 의미여도 필요한 문장의 길이와 필요한 글자는 다르기 때문
    - sequence의 max length와 feature는 주어진 data에서 가지고 온다.

---

### `-` 모델

- 결국 인코더와 디코더는 data length 길이는 통일되어야한다.
- 나머지 차원이 다르지만 모델이 작동할 수 있는 것은 컨벡스트 벡터 과정이 있기 때문
     - 인코더의 output은 사용하지 않고 hidden과 cell state만을 가지고 디코더에 initial하기 때문에 인코더- 디코더의 sequence의 길이와 feature는 다를 수 있는 것이다
- **단 너무 당연한지만 unit의 개수는 일치해야한다. encoder의 hidden, cell의 정보를 받기 위해서**


```python
K.clear_session()
n= 32

encoder_input = Input(shape=(None,encoder_text_len))

encoder = LSTM(units=n, return_state=True) # return_state=True 출력,은닉,셀 반환옵션

# 아웃풋,히든스테이트,셀스테이트 중에 아웃풋 사용x
output, encoder_h, encoder_c = encoder(encoder_input) 

# decoder에서 입력할 state
encoder_state = [encoder_h, encoder_c]

decoder_input = Input(shape=(None,decoder_text_len))

decoder = LSTM(units=n, return_sequences=True, return_state=True)

# 컨텍스트 벡터 encoder_state를 decoder로 전달
decoder_output,decoder_h, decoder_c= decoder(decoder_input,initial_state=encoder_state) 

# decoder에서는 output만을 이용해 출력
decoder_dense = Dense(units=decoder_text_len,activation='softmax')
decoder_output = decoder_dense(decoder_output)

model = Model([encoder_input, decoder_input], decoder_output)
```


```python
model.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                Output Shape                 Param #   Connected to                  
    ==================================================================================================
     input_1 (InputLayer)        [(None, None, 23)]           0         []                            
                                                                                                      
     input_2 (InputLayer)        [(None, None, 24)]           0         []                            
                                                                                                      
     lstm (LSTM)                 [(None, 32),                 7168      ['input_1[0][0]']             
                                  (None, 32),                                                         
                                  (None, 32)]                                                         
                                                                                                      
     lstm_1 (LSTM)               [(None, None, 32),           7296      ['input_2[0][0]',             
                                  (None, 32),                            'lstm[0][1]',                
                                  (None, 32)]                            'lstm[0][2]']                
                                                                                                      
     dense (Dense)               (None, None, 24)             792       ['lstm_1[0][0]']              
                                                                                                      
    ==================================================================================================
    Total params: 15256 (59.59 KB)
    Trainable params: 15256 (59.59 KB)
    Non-trainable params: 0 (0.00 Byte)
    __________________________________________________________________________________________________
    
