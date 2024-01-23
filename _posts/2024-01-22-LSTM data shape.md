# LSTM imput data shape


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from keras.layers import LSTM ,Dense, Bidirectional, Input, TimeDistributed
from keras.models import Sequential
import keras.backend as K
from keras.callbacks import EarlyStopping
```

---

# 1.1. 단변량 데이터


```python
np.random.seed(2)
z = np.random.randn(100)
t=np.arange(0,100)
X = 10*z+t

X.shape
```




    (100,)




```python
plt.plot(X)
```




    [<matplotlib.lines.Line2D at 0x1516cc7dde0>]




    
![png](output_5_1.png)
    


`-` input data 
- 기본적으로 LSTM input data는 3차원의 모습을 가짐
- 단변량인 데이터인 경우 단순 벡터이기 때문에 이를 reshape 해주어야 한다.
- 단변량인 경우 time step에 따라 window 열이 증가하며,feature는 1이다.

`-` 3차원의 구조로 재생성
- 첫번째 차원은 데이터 크기
- 두번째 차원은 time step(input dim)
- 세번째 차원은 feature(모형적합 때 따로 입력 가능)
- 2차원 모형일지라도 적합할 때 feature을 개별적으로 입력하면 3차원 구조가 된다. 

`-` data window


```python
df = pd.DataFrame({'y':X})
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


```python
wd= WINdow(df,4)
```


```python
wd.window().head()
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
      <th>shift_4</th>
      <th>shift_3</th>
      <th>shift_2</th>
      <th>shift_1</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>-4.167578</td>
      <td>0.437332</td>
      <td>-19.361961</td>
      <td>19.402708</td>
      <td>-13.934356</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.437332</td>
      <td>-19.361961</td>
      <td>19.402708</td>
      <td>-13.934356</td>
      <td>-3.417474</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-19.361961</td>
      <td>19.402708</td>
      <td>-13.934356</td>
      <td>-3.417474</td>
      <td>11.028814</td>
    </tr>
    <tr>
      <th>7</th>
      <td>19.402708</td>
      <td>-13.934356</td>
      <td>-3.417474</td>
      <td>11.028814</td>
      <td>-5.452881</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-13.934356</td>
      <td>-3.417474</td>
      <td>11.028814</td>
      <td>-5.452881</td>
      <td>-2.579522</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = wd.y_label
X = wd.feature
```

- 임의로 time step = 4, feature =1
- 단변량이어서 마지막 5열의 값이 각 시퀀스의 y_label이 된다
- 결국 시퀀스가 하나의 값을 예측해야하는 many to one 문제가 된다


```python
X.shape , y.shape
```




    ((96, 4), (96,))



---

# 1.2. 단변량 데이터 모형적합

- 단층 단방향 LSTM 모형
- many to one


```python
unit = 4
input_dim = X.shape[1]
feature = 1
```


```python
model1 = Sequential()
model1.add(LSTM(unit, input_shape =(input_dim,feature)))
model1.add(Dense(1))
model1.compile(loss='mean_squared_error', optimizer='adam')

np.random.seed(1)
early_stop = EarlyStopping(monitor='loss', patience=7, verbose=1)

history = model1.fit(X, y, epochs=2000,
          batch_size=1, verbose=0, callbacks=[early_stop])

```

    Epoch 447: early stopping
    


```python
model1.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm_5 (LSTM)               (None, 4)                 96        
                                                                     
     dense_5 (Dense)             (None, 1)                 5         
                                                                     
    =================================================================
    Total params: 101 (404.00 Byte)
    Trainable params: 101 (404.00 Byte)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    

`-` 예측


```python
pred = model1.predict(X)

from sklearn.metrics import mean_squared_error

mean_squared_error(y,pred.flatten())
```

    3/3 [==============================] - 0s 1ms/step
    




    144.12090399862112




```python
plt.plot(y)
plt.plot(pred.flatten())
```




    [<matplotlib.lines.Line2D at 0x1517fab3970>]




    
![png](output_22_1.png)
    


---

# 2.1 다변량 데이터

- 위 데이터 활용
- 2개의 설명변수와 y


```python
df2 = pd.DataFrame({'z':z,
             't':t,
             'y':X})

df2.head()
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
      <th>z</th>
      <th>t</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.416758</td>
      <td>0</td>
      <td>-4.167578</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.056267</td>
      <td>1</td>
      <td>0.437332</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.136196</td>
      <td>2</td>
      <td>-19.361961</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.640271</td>
      <td>3</td>
      <td>19.402708</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.793436</td>
      <td>4</td>
      <td>-13.934356</td>
    </tr>
  </tbody>
</table>
</div>



`-` input data

- 설명변수만큼 feature 증가
- time step은 단변량과 마찬가지로 데이터의 성질에 따라 아니면 임의로 조절
- 마찬가지로 시퀀스에 따른 단일 값에 대한 문제이므로 seq to one문제이다


```python
X2 = df2.iloc[:,0:2]
y1 = df2['y'].values

X2.shape, y1.shape

```




    ((100, 2), (100,))



---

# 2.2. 모형적합

- 동일한 단층 단방향 LSTM 모델
- many to one


```python
unit = 5
timestep = 5
feature = 2
```


```python
K.clear_session()

model2 = Sequential()
model2.add(LSTM(unit, input_shape =(timestep ,feature)))
model2.add(Dense(1))
model2.compile(loss='mean_squared_error', optimizer='adam')

np.random.seed(1)
early_stop = EarlyStopping(monitor='loss', patience=7, verbose=1)

history = model1.fit(X2.values, y1, epochs=2000,
          batch_size=1, verbose=0, callbacks=[early_stop])

```

    Epoch 263: early stopping
    


```python
model2.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm (LSTM)                 (None, 5)                 160       
                                                                     
     dense (Dense)               (None, 1)                 6         
                                                                     
    =================================================================
    Total params: 166 (664.00 Byte)
    Trainable params: 166 (664.00 Byte)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    

`-` 예측


```python
pred = model1.predict(X2.values)

from sklearn.metrics import mean_squared_error

mean_squared_error(y1,pred.flatten())
```

    4/4 [==============================] - 0s 1ms/step
    




    12.527921541767805




```python
plt.plot(y1)
plt.plot(pred.flatten())
```




    [<matplotlib.lines.Line2D at 0x15103778070>]




    
![png](output_35_1.png)
    


---

# 정리

- 단변량과 다변량의 input data 형식의 큰 차이는 time step과 feature에 있어 보임
- time step(input_dim)같은경우 단변량은 해당 데이터의 주기나 성질에 따라 결정되는 반면 다변량 데이터는 임의로 time step을 쪼갤 수 있어보인다
- feature는 단변량 데이터인 경우 1로 고정이지만 다변량은 해당 설명변수에 따라 feature가 달라진다.
