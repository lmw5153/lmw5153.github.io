# seq to seq LSTM model 이해

---


```python
import numpy as np
import pandas as pd
from keras.layers import LSTM ,Dense, Bidirectional, Input, TimeDistributed
from keras.models import Sequential ,Model
from keras.callbacks import EarlyStopping
import keras.backend as K
```

---

# seq2seq 

- input data : sequence
- output data : sequence
- 자연어 처리 분야에서 많이 활용
- data에 대한 변환이 매우 중요
- 자연어 처리를 위한 인코더와 디코더라는 모듈이 필요
    - Encoder : 임의의 길이를 가진 문장을 고정길이 벡터로 변환하는 작업
    - Decoder : 인코더의 수치 벡터를 통한 모델에 출력 시퀀스를 생성, 활성화함수는 softmax

---

# 데이터 변환

`-` 예제


```python
# 예제 데이터"
input_texts = ["Hello.", "How are you?", "What is your name?", "I'm hungry.", "How old are you?"]
target_texts = ["안녕하세요.", "잘 지내니?", "너의 이름이 뭐니?", "나 배고파.", "너는 몇 살이니?"]

```

`-` 데이터처리, 아래의 데이터를 생성

- encoder 입력데이터
- decoder 입력데이터
- encoder 출력데이터


```python
# data 각 글자 집합 
input_set = set(" ".join(input_texts)) # 원래문장
target_set = set(" ".join(target_texts)) # 번역문장
# 각 글자에 대한 숫자 부여
input_token = dict([(char, i) for i, char in enumerate(input_set)])
target_token = dict([(char, i) for i, char in enumerate(target_set)])

# 시퀀스의 최대 길이
max_encoder_seqlen = max([len(txt) for txt in input_texts])
max_decoder_seqlen = max([len(txt) for txt in target_texts])

# 데이터의 중복되지 않는 총 글자수
encoder_text_len = len(input_set)
decoder_text_len = len(target_set)

# 원핫 인코딩 zero 
encoder_inputdata= np.zeros((len(input_texts), max_encoder_seqlen, encoder_text_len ), dtype='float32') # encoder 입력데이터
decoder_inputdata= np.zeros((len(input_texts), max_decoder_seqlen, decoder_text_len ), dtype='float32') # decoder 입력데이터
decoder_target_data = np.zeros((len(input_texts), max_decoder_seqlen, decoder_text_len), dtype='float32') #encoder 출력데이터

encoder_inputdata.shape, decoder_inputdata.shape, decoder_target_data.shape
```




    ((5, 18, 23), (5, 10, 24), (5, 10, 24))



- 5개의 각각의 문장을 최대 시퀀스 길이로 확장을 시키고 해당되는 글자에 1을 부여할 것임
- encoder와 decoder의 input data의 shape은 서로 충분히 달라질 수 있음
- model fitting 에서 encoder, decoder 두가지 모델을 만들고 각각에 입력을 할 것임
- 또한 encoder model에서 output은 가져오지 않고 hidden, cell state만 가지고 온다

`-` 원핫인코딩(gpt도움 좀 받았습니다)


```python
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_inputdata[i, t, input_token[char]] = 1.0
    for t, char in enumerate(target_text):
        decoder_inputdata[i, t, target_token[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token[char]] = 1.0
```

- decoder의 타겟 데이터는 입력데이터보다 스텝이 한칸 더 빠름
    - seq2seq 모델의 훈련 과정을 위한 방법
    - 미리 정답을 주어 다음 타임스탭에서 예측에활용

---

# encoder

- return_state=True 출력,은닉,셀 반환옵션
- 아웃풋,히든스테이트,셀스테이트 중에 아웃풋 사용x
- units은 인코더, 디코더 동일해야한다.


```python
K.clear_session()
n= 32

encoder_input = Input(shape=(None,encoder_text_len))

encoder = LSTM(units=n, return_state=True) # return_state=True 출력,은닉,셀 반환옵션

# 아웃풋,히든스테이트,셀스테이트 중에 아웃풋 사용x
output, encoder_h, encoder_c = encoder(encoder_input) 

# decoder에서 입력할 state
encoder_state = [encoder_h, encoder_c]
```

---

# decoder
- 컨텍스트 벡터 : initial_state=encoder_state $\rightarrow$ encoder의 정보를 decoder에게 전달
- decoder에서는 output만을 이용해 출력


```python
decoder_input = Input(shape=(None,decoder_text_len))

decoder = LSTM(units=n, return_sequences=True, return_state=True)

# 컨텍스트 벡터 encoder_state를 decoder로 전달
decoder_output,decoder_h, decoder_c= decoder(decoder_input,initial_state=encoder_state) 

# decoder에서는 output만을 이용해 출력
decoder_dense = Dense(units=decoder_text_len,activation='softmax')
decoder_output = decoder_dense(decoder_output)

model = Model([encoder_input, decoder_input], decoder_output)
```

---


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
    

---

# Model fitting


```python
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
```


```python
model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_inputdata, decoder_inputdata], decoder_target_data, 
          batch_size=1, epochs=200,  verbose=0, callbacks=[early_stop])
```

    Epoch 68: early stopping
    




    <keras.src.callbacks.History at 0x1b985fc4c40>



---
