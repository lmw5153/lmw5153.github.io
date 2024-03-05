# N-beats model의 이해

---


```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, Adam
import keras.backend as K
from keras.callbacks import EarlyStopping
```

---

## contributions

![image.png](attachment:image.png)

- 순수한 딥러닝 방법이 통계적 접근 방식보다 타임시리즈 예측의 벤치마크가 더 높았음 따라서 time series의 예측을 위한 순수 ML 연구의 동기을 얻을 수 있음
- time seires를 전통적인 분해법과 같은 해석가능한 유연한 딥러닝 아키텍쳐를 제공해주었음

---

## n-beats

![image.png](attachment:image.png)

- n-beats의 모델 디자인은 몇가지의 원칙에 의존
    1. 기초 아키텍쳐는 간단하고 generic함
    2. 이 아키텍쳐는 time series의 특정 변수 엔지니어링이나 input scaling에 의존하지 않음ㅈ
    3. 이 아키텍쳐는 해석이 가능하게끔 확장될 수 있음

---

## basic block

![image.png](attachment:image.png)
![image-2.png](attachment:image-2.png)

- basic block은 포크 형태의 아키텍쳐를 가짐
- frist block은 특정 길이의 관측된 window만큼의 $x$를 반영함
- multiple of the forecast horizon H만큼의 input window 길이를 설정함 
- block은 input x를 받아들이고 $\hat{x}, \hat{y}$를 output으로 내놓는다
    - $\hat{x}$는 실제 $x$와 가장 잘 추정한 것을 내놓고
    - $\hat{y}$는 블락에서 설정한 H의 길이만큼을 내놓음
- 나머지 블락은 앞서 블락이 내놓은 residual output을 input으로 삼는다.

`-` input x가 first block을 통과하고 다음 블락에게 전달하는 과정
![image-2.png](attachment:image-2.png))

`-` basic모델 구성

![image.png](attachment:image.png)

- 베이직 모델은 크게 2개의 part로 구분되어 있다
    - first part(RELU) : fully connnected network로 구성되었고 expansion coefficients의 예측자인 forward $\theta$와 backward $\theta$를 내놓는다
    - second part(Linear) : backward g함수와 forward g함수로 구성되어있어 앞서 첫번째 파트에서 나온 \theta를 받아들이고 최종적으로 output을 내놓는다

---

## generic block 구현

`-` 연습용 데이터


```python
t = np.linspace(1,10,1000)
X =  t+ np.sin(4*t*np.pi)
y = X+np.cos(t)
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
scale_X= minmax.fit_transform(X.reshape(-1, 1)) # input
scale_y= minmax.fit_transform(y.reshape(-1, 1)) # putput

from sklearn.model_selection import train_test_split
train_x, test_x, train_y,test_y = train_test_split(scale_X,scale_y,shuffle=False)

```


```python
train_x.shape, train_y.shape,test_x.shape,test_y.shape
```




    ((750, 1), (750, 1), (250, 1), (250, 1))



---

### `-` generic block model


```python
backcast_length,horizen = 1,1
feature1 = 1
unit = 128
theta_dim1,theta_dim2= 1,1
forecast_length = horizen

K.clear_session()
d0 = Input(shape=(horizen , feature1))
d1 = Dense(unit,activation = 'relu')(d0)
d2 = Dense(unit,activation = 'relu')(d1)
d3 = Dense(unit,activation = 'relu')(d2)
d4 = Dense(unit,activation = 'relu')(d3)

theta_b = Dense(theta_dim1, activation='linear', name='theta_b')(d4)
theta_f = Dense(theta_dim2, activation='linear',  name='theta_f')(d4)

backcast = Dense(backcast_length, activation='linear', use_bias=False,name='backcast')(theta_b)
forecast = Dense(forecast_length, activation='linear', use_bias=False,name='forecast')(theta_f)

block = Model(inputs=d0, outputs = [backcast,forecast])
optimizer = Adam(learning_rate=0.0001)
block.compile(optimizer=optimizer, loss='mean_squared_error')
block.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_1 (InputLayer)           [(None, 1, 1)]       0           []                               
                                                                                                      
     dense (Dense)                  (None, 1, 128)       256         ['input_1[0][0]']                
                                                                                                      
     dense_1 (Dense)                (None, 1, 128)       16512       ['dense[0][0]']                  
                                                                                                      
     dense_2 (Dense)                (None, 1, 128)       16512       ['dense_1[0][0]']                
                                                                                                      
     dense_3 (Dense)                (None, 1, 128)       16512       ['dense_2[0][0]']                
                                                                                                      
     theta_b (Dense)                (None, 1, 1)         129         ['dense_3[0][0]']                
                                                                                                      
     theta_f (Dense)                (None, 1, 1)         129         ['dense_3[0][0]']                
                                                                                                      
     backcast (Dense)               (None, 1, 1)         1           ['theta_b[0][0]']                
                                                                                                      
     forecast (Dense)               (None, 1, 1)         1           ['theta_f[0][0]']                
                                                                                                      
    ==================================================================================================
    Total params: 50,052
    Trainable params: 50,052
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
epochs = 50
batch_size = 2

block.fit(train_x,[train_x,train_y], epochs=epochs, batch_size=2,callbacks=[early_stop])
          
```

    Epoch 1/50
    375/375 [==============================] - 1s 856us/step - loss: 0.1457 - backcast_loss: 0.1185 - forecast_loss: 0.0272
    Epoch 2/50
    375/375 [==============================] - 0s 861us/step - loss: 0.0055 - backcast_loss: 0.0022 - forecast_loss: 0.0033
    Epoch 3/50
    375/375 [==============================] - 0s 864us/step - loss: 0.0028 - backcast_loss: 2.7359e-04 - forecast_loss: 0.0025
    Epoch 4/50
    375/375 [==============================] - 0s 863us/step - loss: 0.0028 - backcast_loss: 5.3695e-05 - forecast_loss: 0.0028
    Epoch 5/50
    375/375 [==============================] - 0s 859us/step - loss: 0.0026 - backcast_loss: 4.2857e-05 - forecast_loss: 0.0026
    Epoch 6/50
    375/375 [==============================] - 0s 869us/step - loss: 0.0024 - backcast_loss: 3.9441e-05 - forecast_loss: 0.0024
    Epoch 7/50
    375/375 [==============================] - 0s 859us/step - loss: 0.0026 - backcast_loss: 4.0449e-05 - forecast_loss: 0.0026
    Epoch 8/50
    375/375 [==============================] - 0s 866us/step - loss: 0.0026 - backcast_loss: 4.0790e-05 - forecast_loss: 0.0026
    Epoch 9/50
    375/375 [==============================] - 0s 861us/step - loss: 0.0026 - backcast_loss: 3.5510e-05 - forecast_loss: 0.0025
    Epoch 10/50
    375/375 [==============================] - 0s 861us/step - loss: 0.0026 - backcast_loss: 3.4233e-05 - forecast_loss: 0.0026
    Epoch 11/50
    375/375 [==============================] - 0s 867us/step - loss: 0.0024 - backcast_loss: 3.5760e-05 - forecast_loss: 0.0024
    Epoch 12/50
    375/375 [==============================] - 0s 862us/step - loss: 0.0026 - backcast_loss: 3.2765e-05 - forecast_loss: 0.0025
    Epoch 13/50
    375/375 [==============================] - 0s 865us/step - loss: 0.0026 - backcast_loss: 3.4796e-05 - forecast_loss: 0.0025
    Epoch 14/50
    375/375 [==============================] - 0s 862us/step - loss: 0.0025 - backcast_loss: 3.0483e-05 - forecast_loss: 0.0025
    Epoch 15/50
    375/375 [==============================] - 0s 867us/step - loss: 0.0023 - backcast_loss: 2.8703e-05 - forecast_loss: 0.0023
    Epoch 16/50
    375/375 [==============================] - 0s 867us/step - loss: 0.0023 - backcast_loss: 2.4671e-05 - forecast_loss: 0.0023
    Epoch 17/50
    375/375 [==============================] - 0s 869us/step - loss: 0.0023 - backcast_loss: 2.4180e-05 - forecast_loss: 0.0023
    Epoch 18/50
    375/375 [==============================] - 0s 865us/step - loss: 0.0024 - backcast_loss: 2.3800e-05 - forecast_loss: 0.0023
    Epoch 19/50
    375/375 [==============================] - 0s 877us/step - loss: 0.0023 - backcast_loss: 2.1466e-05 - forecast_loss: 0.0023
    Epoch 20/50
    375/375 [==============================] - 0s 867us/step - loss: 0.0023 - backcast_loss: 1.9968e-05 - forecast_loss: 0.0022
    Epoch 21/50
    375/375 [==============================] - 0s 864us/step - loss: 0.0025 - backcast_loss: 2.8001e-05 - forecast_loss: 0.0025
    Epoch 22/50
    375/375 [==============================] - 0s 873us/step - loss: 0.0024 - backcast_loss: 2.1017e-05 - forecast_loss: 0.0024
    Epoch 23/50
    375/375 [==============================] - 0s 866us/step - loss: 0.0023 - backcast_loss: 2.1234e-05 - forecast_loss: 0.0023
    Epoch 24/50
    375/375 [==============================] - 0s 864us/step - loss: 0.0024 - backcast_loss: 2.0107e-05 - forecast_loss: 0.0024
    Epoch 25/50
    375/375 [==============================] - 0s 864us/step - loss: 0.0023 - backcast_loss: 1.9770e-05 - forecast_loss: 0.0023
    Epoch 26/50
    375/375 [==============================] - 0s 867us/step - loss: 0.0024 - backcast_loss: 2.4273e-05 - forecast_loss: 0.0024
    Epoch 27/50
    375/375 [==============================] - 0s 864us/step - loss: 0.0023 - backcast_loss: 1.3454e-05 - forecast_loss: 0.0023
    Epoch 28/50
    375/375 [==============================] - 0s 862us/step - loss: 0.0023 - backcast_loss: 1.9195e-05 - forecast_loss: 0.0023
    Epoch 29/50
    375/375 [==============================] - 0s 861us/step - loss: 0.0022 - backcast_loss: 1.7036e-05 - forecast_loss: 0.0022
    Epoch 30/50
    375/375 [==============================] - 0s 864us/step - loss: 0.0022 - backcast_loss: 2.0086e-05 - forecast_loss: 0.0022
    Epoch 31/50
    375/375 [==============================] - 0s 861us/step - loss: 0.0022 - backcast_loss: 1.5170e-05 - forecast_loss: 0.0022
    Epoch 32/50
    375/375 [==============================] - 0s 864us/step - loss: 0.0024 - backcast_loss: 2.6870e-05 - forecast_loss: 0.0023
    Epoch 33/50
    375/375 [==============================] - 0s 864us/step - loss: 0.0022 - backcast_loss: 1.6357e-05 - forecast_loss: 0.0022
    Epoch 34/50
    375/375 [==============================] - 0s 869us/step - loss: 0.0024 - backcast_loss: 2.2703e-05 - forecast_loss: 0.0023
    Epoch 35/50
    375/375 [==============================] - 0s 875us/step - loss: 0.0024 - backcast_loss: 3.5373e-05 - forecast_loss: 0.0023
    Epoch 36/50
    375/375 [==============================] - 0s 869us/step - loss: 0.0022 - backcast_loss: 1.3268e-05 - forecast_loss: 0.0022
    Epoch 37/50
    375/375 [==============================] - 0s 865us/step - loss: 0.0023 - backcast_loss: 1.8128e-05 - forecast_loss: 0.0023
    Epoch 38/50
    375/375 [==============================] - 0s 859us/step - loss: 0.0022 - backcast_loss: 2.2329e-05 - forecast_loss: 0.0022
    Epoch 39/50
    375/375 [==============================] - 0s 856us/step - loss: 0.0023 - backcast_loss: 2.2981e-05 - forecast_loss: 0.0023
    Epoch 40/50
    375/375 [==============================] - 0s 857us/step - loss: 0.0022 - backcast_loss: 1.5427e-05 - forecast_loss: 0.0022
    Epoch 41/50
    375/375 [==============================] - 0s 858us/step - loss: 0.0023 - backcast_loss: 2.0651e-05 - forecast_loss: 0.0023
    Epoch 42/50
    375/375 [==============================] - 0s 853us/step - loss: 0.0022 - backcast_loss: 2.5133e-05 - forecast_loss: 0.0022
    Epoch 43/50
    375/375 [==============================] - 0s 859us/step - loss: 0.0023 - backcast_loss: 2.2727e-05 - forecast_loss: 0.0022
    Epoch 44/50
    375/375 [==============================] - 0s 859us/step - loss: 0.0022 - backcast_loss: 1.9512e-05 - forecast_loss: 0.0022
    Epoch 45/50
    375/375 [==============================] - 0s 868us/step - loss: 0.0023 - backcast_loss: 2.6689e-05 - forecast_loss: 0.0023
    Epoch 46/50
    375/375 [==============================] - 0s 876us/step - loss: 0.0023 - backcast_loss: 2.3922e-05 - forecast_loss: 0.0022
    Epoch 46: early stopping
    




    <keras.callbacks.History at 0x2088a8ab5e0>




```python
pred = block.predict(train_x)
```

    24/24 [==============================] - 0s 783us/step
    

### `-` backcast


```python
plt.plot(train_x, color = 'red',linewidth=0.9, linestyle='--')
plt.plot(pred[0].flatten())
```




    [<matplotlib.lines.Line2D at 0x2088df88b80>]




    
![png](output_30_1.png)
    


### `-` forecast


```python
plt.plot(train_y, color = 'red',linewidth=0.9, linestyle='--')
plt.plot(pred[1].flatten())

```




    [<matplotlib.lines.Line2D at 0x2088e0a9990>]




    
![png](output_32_1.png)
    

