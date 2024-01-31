# encoder -decoder LSTM model 구현 예제

- 지난번 many2many lstm 언어모델 구현에서 tiem series forecasting 문제로 변경

---

# import 


```python
import numpy as np
import pandas as pd
from datasetsforecast.m4 import M4, M4Evaluation
import matplotlib.pyplot as plt
import time
from keras.layers import LSTM ,Dense, Bidirectional, Input, TimeDistributed
from keras.models import Sequential ,Model
from keras.callbacks import EarlyStopping
import keras.backend as K
```

---

# source data


```python
df,*_= M4.load(directory='data',group = 'Hourly')

lst = list(set(df.loc[:,'unique_id']))

tt = [df[df.loc[:,'unique_id']==lst[i]].iloc[:,1:4].set_index(['ds']) 
      for i in range(len(lst))]

for l in range(len(tt)):
    for s in range(1, 25):
        tt[l]['shift_{}'.format(s)] = tt[l]['y'].shift(s)
        tt[l]['shift_{}'.format(s)] = tt[l]['y'].shift(s)
        
tt=[tt[i].dropna(axis=0) for i in range(len(tt))]

train = np.concatenate([np.array(tt[i].iloc[:,1:]) for i in range(len(tt))])
y = np.concatenate([np.array(tt[i].iloc[:,0]) for i in range(len(tt))]).reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler
min_max_scaler1 = MinMaxScaler()

X_scale = min_max_scaler1.fit_transform(train)
y_scale = min_max_scaler1.fit_transform(y)
```

# target data


```python
df = pd.read_csv('C:/Users/default.DESKTOP-2ISHQBS/Documents/R/time_ele/train.csv')

arr = df.iloc[:,9] # 전력소비량
date=  pd.to_datetime(df.iloc[:,2]) # 일시

df_= pd.DataFrame({'date':date,
              'ele': arr})

df_ = df_.set_index('date')

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

df_['mmele'] = min_max_scaler.fit_transform(df_.iloc[:].to_numpy().reshape(-1,1))

df_ = df_.drop(['ele'],axis=1)

for s in range(1, 25):
    df_['shift_{}'.format(s)] = df_['mmele'].shift(s)
    df_['shift_{}'.format(s)] = df_['mmele'].shift(s)

df_ = df_.dropna()

y = df_.iloc[:,[0]].values #scaled 

X = df_.iloc[:,1:]

from sklearn.model_selection import train_test_split

x_train,x_test, y_train , y_test = train_test_split(X.iloc[-2400:,:],y[-2400:],shuffle=False, test_size=0.1)
```


```python
class WINdow:
    def __init__(self,df,timestep):
        self.df = df
        self.timestep=timestep+1 # 예상한 timestep보다 1적기 때문에 +1
        
    def window(self):
        for i in range(1, self.timestep):
            df['shift_{}'.format(i)] = df.iloc[:,0].shift(i)
            df['shift_{}'.format(i)] = df.iloc[:,0].shift(i)
        window_df = df.dropna(axis=0) # 결측치 공간 제거
        self.window_df = window_df.iloc[:,::-1] # 좌우 반전
        
                
        self.feature= self.window_df.iloc[:,:-1].values
        self.y_label= self.window_df.iloc[:,-1].values
        
        return self. window_df 
```

`-` sourece data를 target data 와 data size 통일


```python
a = ((X_scale.shape[0] // x_train.shape[0])*x_train.shape[0])
X_ad_scale= X_scale.flatten()[:int(a)].reshape(x_train.shape[0],-1)
y_ad_scale = y_scale[:int(a)].reshape(x_train.shape[0],-1)
```


```python
X_ad_scale.shape,x_train.shape,y_train.shape, x_test.shape
```

`-` 비교군 lstm fitting data shape


```python

df =pd.DataFrame(X_scale.flatten()[:int(a)])#.reshape(-1,24).shape
dfwindow= WINdow(df,24)
dfwindow.window()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>shift_24</th>
      <th>shift_23</th>
      <th>shift_22</th>
      <th>shift_21</th>
      <th>shift_20</th>
      <th>shift_19</th>
      <th>shift_18</th>
      <th>shift_17</th>
      <th>shift_16</th>
      <th>shift_15</th>
      <th>...</th>
      <th>shift_9</th>
      <th>shift_8</th>
      <th>shift_7</th>
      <th>shift_6</th>
      <th>shift_5</th>
      <th>shift_4</th>
      <th>shift_3</th>
      <th>shift_2</th>
      <th>shift_1</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>0.002205</td>
      <td>0.001980</td>
      <td>0.001962</td>
      <td>0.001935</td>
      <td>0.001841</td>
      <td>0.001674</td>
      <td>0.001279</td>
      <td>0.001273</td>
      <td>0.001159</td>
      <td>0.000721</td>
      <td>...</td>
      <td>0.000357</td>
      <td>0.000745</td>
      <td>0.000832</td>
      <td>0.001627</td>
      <td>0.002091</td>
      <td>0.002114</td>
      <td>0.002121</td>
      <td>0.002122</td>
      <td>0.002184</td>
      <td>0.002162</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.001980</td>
      <td>0.001962</td>
      <td>0.001935</td>
      <td>0.001841</td>
      <td>0.001674</td>
      <td>0.001279</td>
      <td>0.001273</td>
      <td>0.001159</td>
      <td>0.000721</td>
      <td>0.000464</td>
      <td>...</td>
      <td>0.000745</td>
      <td>0.000832</td>
      <td>0.001627</td>
      <td>0.002091</td>
      <td>0.002114</td>
      <td>0.002121</td>
      <td>0.002122</td>
      <td>0.002184</td>
      <td>0.002162</td>
      <td>0.002205</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.001962</td>
      <td>0.001935</td>
      <td>0.001841</td>
      <td>0.001674</td>
      <td>0.001279</td>
      <td>0.001273</td>
      <td>0.001159</td>
      <td>0.000721</td>
      <td>0.000464</td>
      <td>0.000484</td>
      <td>...</td>
      <td>0.000832</td>
      <td>0.001627</td>
      <td>0.002091</td>
      <td>0.002114</td>
      <td>0.002121</td>
      <td>0.002122</td>
      <td>0.002184</td>
      <td>0.002162</td>
      <td>0.002205</td>
      <td>0.001980</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.001935</td>
      <td>0.001841</td>
      <td>0.001674</td>
      <td>0.001279</td>
      <td>0.001273</td>
      <td>0.001159</td>
      <td>0.000721</td>
      <td>0.000464</td>
      <td>0.000484</td>
      <td>0.000616</td>
      <td>...</td>
      <td>0.001627</td>
      <td>0.002091</td>
      <td>0.002114</td>
      <td>0.002121</td>
      <td>0.002122</td>
      <td>0.002184</td>
      <td>0.002162</td>
      <td>0.002205</td>
      <td>0.001980</td>
      <td>0.001962</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.001841</td>
      <td>0.001674</td>
      <td>0.001279</td>
      <td>0.001273</td>
      <td>0.001159</td>
      <td>0.000721</td>
      <td>0.000464</td>
      <td>0.000484</td>
      <td>0.000616</td>
      <td>0.000580</td>
      <td>...</td>
      <td>0.002091</td>
      <td>0.002114</td>
      <td>0.002121</td>
      <td>0.002122</td>
      <td>0.002184</td>
      <td>0.002162</td>
      <td>0.002205</td>
      <td>0.001980</td>
      <td>0.001962</td>
      <td>0.001935</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>362875</th>
      <td>0.000008</td>
      <td>0.000010</td>
      <td>0.000012</td>
      <td>0.000015</td>
      <td>0.000017</td>
      <td>0.000017</td>
      <td>0.000017</td>
      <td>0.000018</td>
      <td>0.000018</td>
      <td>0.000017</td>
      <td>...</td>
      <td>0.000005</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000003</td>
      <td>0.000003</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>0.000005</td>
      <td>0.000007</td>
    </tr>
    <tr>
      <th>362876</th>
      <td>0.000010</td>
      <td>0.000012</td>
      <td>0.000015</td>
      <td>0.000017</td>
      <td>0.000017</td>
      <td>0.000017</td>
      <td>0.000018</td>
      <td>0.000018</td>
      <td>0.000017</td>
      <td>0.000016</td>
      <td>...</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000003</td>
      <td>0.000003</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>0.000005</td>
      <td>0.000007</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <th>362877</th>
      <td>0.000012</td>
      <td>0.000015</td>
      <td>0.000017</td>
      <td>0.000017</td>
      <td>0.000017</td>
      <td>0.000018</td>
      <td>0.000018</td>
      <td>0.000017</td>
      <td>0.000016</td>
      <td>0.000015</td>
      <td>...</td>
      <td>0.000002</td>
      <td>0.000002</td>
      <td>0.000003</td>
      <td>0.000003</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>0.000005</td>
      <td>0.000007</td>
      <td>0.000008</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>362878</th>
      <td>0.000015</td>
      <td>0.000017</td>
      <td>0.000017</td>
      <td>0.000017</td>
      <td>0.000018</td>
      <td>0.000018</td>
      <td>0.000017</td>
      <td>0.000016</td>
      <td>0.000015</td>
      <td>0.000013</td>
      <td>...</td>
      <td>0.000002</td>
      <td>0.000003</td>
      <td>0.000003</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>0.000005</td>
      <td>0.000007</td>
      <td>0.000008</td>
      <td>0.000010</td>
      <td>0.000012</td>
    </tr>
    <tr>
      <th>362879</th>
      <td>0.000017</td>
      <td>0.000017</td>
      <td>0.000017</td>
      <td>0.000018</td>
      <td>0.000018</td>
      <td>0.000017</td>
      <td>0.000016</td>
      <td>0.000015</td>
      <td>0.000013</td>
      <td>0.000011</td>
      <td>...</td>
      <td>0.000003</td>
      <td>0.000003</td>
      <td>0.000004</td>
      <td>0.000005</td>
      <td>0.000005</td>
      <td>0.000007</td>
      <td>0.000008</td>
      <td>0.000010</td>
      <td>0.000012</td>
      <td>0.000015</td>
    </tr>
  </tbody>
</table>
<p>362856 rows × 25 columns</p>
</div>




```python
lstm_x= dfwindow.feature
lstm_y= dfwindow.y_label

lstm_x.shape, lstm_y.shape
```




    ((362856, 24), (362856,))



---

# model bulid

- encoder, decoder 형식
- 각 모델마다 두개의 lstm 레이어층 사용
- 각 층의 cell = 64


```python
n= 64
encoder_feature = 1
decoder_feature = 1
```


```python
K.clear_session()

####################################################################################
# encoder

encoder_input = Input(shape=(None,encoder_feature))
# encoder layer1 
encoder1 = LSTM(units=n, return_sequences=True, return_state=True) # return_state=True 출력,은닉,셀 반환옵션
output, encoder_h, encoder_c = encoder1(encoder_input) 
# encoder layer2
encoder2 = LSTM(units=n, return_state=True)
output2, encoder_h2, encoder_c2 = encoder2(output)

# decoder에서 입력할 state
encoder_state = [encoder_h2, encoder_c2]

####################################################################################
# decoder 
decoder_input = Input(shape=(None,decoder_feature))

# decoder layer1
decoder1 = LSTM(units=n, return_sequences=True, return_state=True)

# 컨텍스트 벡터 encoder_state를 decoder로 전달
decoder_output,decoder_h, decoder_c= decoder1(decoder_input,initial_state=encoder_state) 

# decoder layer2
decoder2 = LSTM(units=n, return_state=True)
decoder_output2,decoder_h2, decoder_c2= encoder2(decoder_output)

# decoder에서는 output2만을 이용해 출력
decoder_dense = Dense(units=decoder_feature)
decoder_final = decoder_dense(decoder_output2)

model = Model([encoder_input, decoder_input], decoder_final)
```

---


```python
model.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                Output Shape                 Param #   Connected to                  
    ==================================================================================================
     input_1 (InputLayer)        [(None, None, 1)]            0         []                            
                                                                                                      
     lstm (LSTM)                 [(None, None, 64),           16896     ['input_1[0][0]']             
                                  (None, 64),                                                         
                                  (None, 64)]                                                         
                                                                                                      
     lstm_1 (LSTM)               [(None, 64),                 33024     ['lstm[0][0]',                
                                  (None, 64),                            'lstm_2[0][0]']              
                                  (None, 64)]                                                         
                                                                                                      
     input_2 (InputLayer)        [(None, None, 1)]            0         []                            
                                                                                                      
     lstm_2 (LSTM)               [(None, None, 64),           16896     ['input_2[0][0]',             
                                  (None, 64),                            'lstm_1[0][1]',              
                                  (None, 64)]                            'lstm_1[0][2]']              
                                                                                                      
     dense (Dense)               (None, 1)                    65        ['lstm_1[1][0]']              
                                                                                                      
    ==================================================================================================
    Total params: 66881 (261.25 KB)
    Trainable params: 66881 (261.25 KB)
    Non-trainable params: 0 (0.00 Byte)
    __________________________________________________________________________________________________
    

---

# encoder-decoder model fitting


```python
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

model.fit([X_ad_scale,x_train],y_train, epochs=100,
          batch_size=1, verbose=1, callbacks=[early_stop])
```

    Epoch 1/100
    2160/2160 [==============================] - 78s 34ms/step - loss: 5.4695e-05
    Epoch 2/100
    2160/2160 [==============================] - 74s 34ms/step - loss: 2.2603e-05
    Epoch 3/100
    2160/2160 [==============================] - 74s 34ms/step - loss: 1.8553e-05
    Epoch 4/100
    2160/2160 [==============================] - 74s 34ms/step - loss: 1.7918e-05
    Epoch 5/100
    2160/2160 [==============================] - 74s 34ms/step - loss: 1.9142e-05
    Epoch 5: early stopping
    




    <keras.src.callbacks.History at 0x2419255ed40>




```python
pred = model.predict([X_ad_scale,x_train])
```

    68/68 [==============================] - 3s 24ms/step
    


```python
plt.plot(pred)
plt.plot(y_train)
```




    [<matplotlib.lines.Line2D at 0x2419c8dafe0>]




    
![png](output_26_1.png)
    


---

# 비교군 모델

- 타겟데이터를 피팅하고 예측하는 모델
- 소스데이터를 학습하고 타겟데이터를 예측하는 모델

`-` non pretrained model


```python
K.clear_session()
modelnon0 = Sequential() # Sequeatial Model
modelnon0.add(LSTM(64, return_sequences=True,input_shape=(24, 1)))# (timestep, feature)
modelnon0.add(LSTM(64))
modelnon0.add(Dense(1)) # output = 1
modelnon0.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
start_timenon0 = time.time()
modelnon0.fit(x_train,y_train, epochs=100,
          batch_size=1, verbose=1, callbacks=[early_stop])
end_timenon0 = time.time()
```

    Epoch 1/100
    2160/2160 [==============================] - 14s 5ms/step - loss: 5.4045e-05
    Epoch 2/100
    2160/2160 [==============================] - 12s 5ms/step - loss: 2.3120e-05
    Epoch 3/100
    2160/2160 [==============================] - 12s 5ms/step - loss: 1.9951e-05
    Epoch 4/100
    2160/2160 [==============================] - 12s 5ms/step - loss: 1.8749e-05
    Epoch 5/100
    2160/2160 [==============================] - 12s 5ms/step - loss: 1.8339e-05
    Epoch 6/100
    2160/2160 [==============================] - 12s 5ms/step - loss: 1.8754e-05
    Epoch 6: early stopping
    

---

`-` using pretrain model(no freezing)


```python
K.clear_session()
model01 = Sequential() # Sequeatial Model
model01.add(LSTM(64, return_sequences=True, input_shape=(24, 1))) # (timestep, feature)
model01.add(LSTM(64, return_sequences=True)) # 연결한 모델의 차원을 맞추기 위해 시퀀스 반환을 함
model01.add(Dense(1)) # output 사용 x
model01.compile(loss='mean_squared_error', optimizer='adam')
np.random.seed(1)
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

model01.fit(lstm_x, lstm_y, epochs=100,
          batch_size=128, verbose=1, callbacks=[early_stop])
```

    Epoch 1/100
    2835/2835 [==============================] - 71s 24ms/step - loss: 8.4261e-04
    Epoch 2/100
    2835/2835 [==============================] - 70s 25ms/step - loss: 8.4190e-04
    Epoch 3/100
    2835/2835 [==============================] - 71s 25ms/step - loss: 8.4190e-04
    Epoch 4/100
    2835/2835 [==============================] - 72s 25ms/step - loss: 8.4205e-04
    Epoch 4: early stopping
    




    <keras.src.callbacks.History at 0x240f2ca9600>




```python
pretrained_layers = model01.layers[:-1]
for layer in model01.layers:
    layer.trainable = True # freezing
    

model012 = Sequential(pretrained_layers)
model012.add(LSTM(64, input_shape=(24,64)))
model012.add(Dense(1))
model012.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0)
start_time22 = time.time()
model012.fit(x_train, y_train, epochs=100,
          batch_size=128, verbose=0, callbacks=[early_stop])
end_time22 = time.time()
```

---

# test

- 인코더 디코더 모델을 쓰기 위해서 data size를 맞춰주는 것이 매우 중요하다.


```python
a = ((X_scale.shape[0] // x_test.shape[0])*x_test.shape[0])
X_ad_test= X_scale.flatten()[:int(a)].reshape(x_test.shape[0],-1)
#y_ad_test = y_scale[:int(a)].reshape(x_test.shape[0],-1)
```


```python
X_ad_test.shape, x_test.shape, y_test.shape
```




    ((240, 1514), (240, 24), (240, 1))



---


```python
pred2 = model.predict([X_ad_test,x_test])
pred3 = modelnon0.predict(x_test)
pred4 = model012.predict(x_test)
```

    8/8 [==============================] - 2s 176ms/step
    8/8 [==============================] - 1s 4ms/step
    8/8 [==============================] - 0s 5ms/step
    

`-` y plot


```python
plt.plot(pred2, color = 'red',label = 'en-de fitting')
plt.plot(pred3,label = 'normal fitting')
plt.plot(pred4,label = 'pretrain fitting')
plt.plot(y_test,label = 'y', color = 'black')
plt.legend(loc='lower left')
plt.show()
```


    
![png](output_42_0.png)
    


`-` **MSE**


```python
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mse1= mean_squared_error(y_test, pred2)
mse2= mean_squared_error(y_test, pred3)
mse3= mean_squared_error(y_test, pred4)
mae1= mean_absolute_error(y_test, pred2)
mae2= mean_absolute_error(y_test, pred3)
mae3= mean_absolute_error(y_test, pred4)
print('en-de fitting   ',f"MSE : {mse1:.8f}",f"MAE : {mae1:.8f}")
print('normal fitting  ',f"MSE : {mse2:.8f}",f"MAE : {mae2:.8f}")
print('pretrain fitting',f"MSE : {mse3:.8f}",f"MAE : {mae3:.8f}")
```

    en-de fitting    MSE : 0.00000691 MAE : 0.00212697
    normal fitting   MSE : 0.00000632 MAE : 0.00196587
    pretrain fitting MSE : 0.00000689 MAE : 0.00205653
    

`-` 정리/보완
- 해당 실험에서 normal fitting이 MSE 지표가 가장 좋게 나왔지만 유의한 수준이라고 보기는 어렵다
- 소스데이터의 크기가 기존 pretrain 모델은 많이 커야하지만 encoder- decoder 방법 모델은 target 데이터의 크기에 data size가 따라가기 때문에 source data의 크기가 비교적 작아도 될 것이다
- encoder-decoder 방법 모델에서 소스데이터를 처리하는 방법을 더 고민해야할 것이다. 현재는 단순히 data size에 맞추어 shape를 잘랐기 때문이다.
