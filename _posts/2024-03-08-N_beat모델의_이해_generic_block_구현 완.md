# N-beats model의 이해

- 코드참고 : https://github.com/philipperemy/n-beats/tree/master
- 관련논문 : https://arxiv.org/pdf/1905.10437.pdf?trk=public_post_comment-text


---


```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate, Dropout,Subtract,Add
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, Adam
import keras.backend as K
from keras.callbacks import EarlyStopping
from nbeats_keras.model import NBeatsNet as NBeatsKeras
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
    2. 이 아키텍쳐는 time series의 특정 변수 엔지니어링이나 input scaling에 의존하지 않음
    3. 이 아키텍쳐는 해석이 가능하게끔 확장될 수 있음

---

## basic block

![image.png](attachment:image.png)
![image-2.png](attachment:image-2.png)

- basic block은 포크 형태의 아키텍쳐를 가짐
- frist block은 특정 길이의 관측된 window만큼의 $x$를 반영함
- the forecast horizon H의 배수만큼의 input window 길이를 설정함 
- block은 input x를 받아들이고 $\hat{x}, \hat{y}$를 output으로 내놓는다
    - $\hat{x}$는 실제 $x$와 가장 잘 추정한 것을 내놓고(backcast)
    - $\hat{y}$는 블락에서 설정한 H의 길이만큼을 내놓음(forecast)
- 나머지 블락은 앞서 블락이 내놓은 residual output을 input으로 삼는다.

## `-` basic block

![image.png](attachment:image.png)

- 베이직 모델은 크게 2개의 part로 구분되어 있다
    - first part(RELU) : fully connnected network로 구성되었고 expansion coefficients의 예측자인 forward $\theta$와 backward $\theta$를 내놓는다
    - second part(Linear) : backward g함수와 forward g함수로 구성되어있어 앞서 첫번째 파트에서 나온 $\theta$를 받아들이고 최종적으로 output을 내놓는다

![image.png](attachment:image.png)

- 선형 레이어
    - $g^f_l$에 의해 제공된 기저벡터를 적절하게 혼합하여 forward expasion coff를 예측
    - 더불어 하위 네트워크는 $g^b_l$에 의해 사용되는 backward expasion coff를 예측하여 x를 추정
    - the downstream blocks에 의해 예측에 도움외 되지 않는 입력 구성 요소를 제거하는 것을 목표

![image.png](attachment:image.png)

- $v^f_i,v^b_i$ : forecast, backcast basis vectors
- $g^b,g_f $는 풍부한 기저벡터 set를 제공
    - 그들의 반영된 아웃풋은 적절한 다양한 편향을 가진 coff를 잘 나타낼 것임
    - $g^b,g_f $는 배울 수 있게 선택할 수 있고 이는 출력 구조를 적절하게 제약하기 위해 특정 문제에 특화된 귀납적 편향을 반영하기 위한 것

---

## `-` doubly residual stacking

![image.png](attachment:image.png)

- 고전적인 잔차 네트워크는 레이어의 스택의 인풋에 출력을 더한후 결과를 다음 스택에 전달
- denseNet 아키텍쳐는 각 스택의 출력에서 이를 따르는 모든 스택의 입력으로 추가적인 연결을 도입, 그러나 이런 것은 해석하기 어렵다
- 새로운 계층적 이중 잔차 topology를 제안
    - one : 각 레이어의 backcast 예측
    - two : 각 레이어의 forecast 예측

![image.png](attachment:image.png)

- 첫번째 블락에서 input x는 $x_1 \equiv x$  
- 다른 모든 블락에서 backcast 잔여 브랜치 x는 인풋 신호에 순차적인 분석이라고 볼 수 있다
- 이전 블락에서 the sigmal $\hat{x_{l-1}}$을 잘 근사하는 신호 일부가 삭제됨
- 따라서 하류 블락의 예측 작업이 단순하게 된다
- 이런 구조는 유연한 그래디언트 역전파를 용이하게 함

- **더욱 중요한 것은 각 블록이 먼저 스택 수준에서, 그리고 나중에 전체 네트워크 수준에서 집계한 일부 예측 $\hat{y}$를 출력**
- 최종 예측 $\hat{y}$는 모든 부분 예측의 합계이다.

---

## `-` interpretability

![image.png](attachment:image.png)

- 두가지를 제안
    - 일반적인 딥러닝
    - 해석가능한 특정한 귀납적 편향이 더해진 구조 모델
- generic achitecture는 일반적인 타임시리즈의 지식에 의존하지 않음

![image.png](attachment:image.png)

- 이 모델의 해석은 네트워크에서 학습한 기저 벡터의 일부인 $\hat{y}$ 예측의 분해를 학습함
- matrix 기저벡터는 $H \times dim(\theta^f)$의 차원을 가짐
    - 첫번째 차원은 예측 도메인에서 이산 시간의 순서의 해석이라고 볼 수 있다
    - 두번째 차원은 기저 함수들의 지수,시간 도메인에서 파형이라고 볼 수 있다
- 기저벡터의 형태에 대한 추가적인 제약이 없기 때문에 딥모델은 내재된 구조를 갖지않고 파형을 학습함
- 따라서 예측값에 대한 해석이 불가능하다

---
## `-` 해석가능한 block
### `-` trend model

![image.png](attachment:image.png)


- 가장 단조로운 함수이거나 천천히 변화하는 함수
- 이를 흉내내기위해 작은 차수의 p의 다항식이 되게끔 제한할 것이다.
- forecast에 $t^i$추가 
    - 이는 horizen 만큼의 길이를 가지고 그 값은 horizen로 나눈 값이다
- T는 t의 거듭제곱으로 이루어진 행렬 차수가 낮다면 이는 trend를 흉내낸다

### `-`Seasonality model.

![image.png](attachment:image.png)

- 계절 주기 성분을 모델에 넣기위해 g함수를 주기 함수와 함께 제한조건을 걸 것임
- 이는 전형적인 계절 패턴을 모방하는 주기적인 함수
- 이렇게 주어진 표현은 시계열 데이터의 계절성을 설명하고, 
- 주기적인 패턴을 표현하기 위해 삼각파 기저 함수를 사용하는 것으로 이해된다.

![image.png](attachment:image.png)

- 해석가능한 아키텍쳐는 traned stack 뒤로 계절성스택이 이어짐
- 이중 잔차 스택은 orecast/ backcast의 원칙과 결합
    1. trend 요소는 계절 스택에 도달하기전에 input window x로부터 삭제됨
    2. 추세와 계절의 부분 예측들은 별도의 해석가능한 output으로 사용가능하다
- 구조적으로 각 스택은 그림 1과 같이 잔차 연결이 연결된 여러 블록으로 구성되며 각 스택은 학습할 수 없는 $g^b_s$' 및 $g^f_s$를 공유
- 블록 수는 추세 및 계절성 모두에 대해 3 
- 우리는 gbs;' 및 gfs;를 공유하는 것 외에도 스택에서 블록 간의 모든 가중치를 공유하면 더 나은 검증 성능을 얻을 수 있음을 발견했다고한다

---

---

`-` 예시 데이터
- 1000개의 index를 가진 가상 데이터
- $y = t+sin(10t \pi) + \epsilon$


```python
t = np.linspace(1,10,1000)
X =  t+ np.sin(10*t*np.pi) +np.random.normal(1,0.5,1000)

class WINdow:
    def __init__(self,df,timestep):
        self.df = df
        self.timestep=timestep+1 # 예상한 timestep보다 1적기 때문에 +1
        
    def window(self):
        for i in range(1, self.timestep):
            self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
            self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
        window_df = self.df.dropna(axis=0) # 결측치 공간 제거
        self.window_df = window_df.iloc[:,::-1] # 좌우 반전
        
                
        self.feature= self.window_df.iloc[:,:-1].values
        self.y_label= self.window_df.iloc[:,-1].values
        
        return self. window_df 
df = pd.DataFrame(X)
x_window = WINdow(df,10)
x_window.window()

from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
scale_X= minmax.fit_transform(x_window.feature) # input
scale_y= minmax.fit_transform(x_window.y_label.reshape(-1, 1)) # putput

from sklearn.model_selection import train_test_split
train_x, test_x, train_y,test_y = train_test_split(scale_X,scale_y,shuffle=False)

train_x.shape, train_y.shape,test_x.shape,test_y.shape
```




    ((742, 10), (742, 1), (248, 10), (248, 1))




```python
plt.plot(X)
plt.title("example train data")
plt.show()
```


    
![png](output_46_0.png)
    


---

## 단일 generic block 구현

![image.png](attachment:image.png)


```python
# time step 
backcast_length = 10
# horizen
forecast_length = 1
feature1 = 1
# block setting
unit = 128
theta_dim1,theta_dim2= 1,1

K.clear_session()
#############################################################################################
#### blcok 1
input_x_1 = Input(shape=(backcast_length))
input_x_2 = Input(shape=(backcast_length))

d1 = Dense(unit,activation = 'relu')(input_x_1)
d2 = Dense(unit,activation = 'relu')(d1)
d3 = Dense(unit,activation = 'relu')(d2)
d4 = Dense(unit,activation = 'relu')(d3)

theta_b_1 = Dense(theta_dim1, activation='linear', name='theta_b')(d4)
theta_f_1 = Dense(theta_dim2, activation='linear',  name='theta_f')(d4)

backcast_1 = Dense(backcast_length , activation='linear', use_bias=False,name='backcast')(theta_b_1)
forecast_1 = Dense(forecast_length, activation='linear', use_bias=False,name='forecast')(theta_f_1)

subtract_1 = Subtract()([input_x_2 , backcast_1])

block = Model(inputs=[input_x_1,input_x_2],outputs = [subtract_1 ,forecast_1])
optimizer = Adam(learning_rate=0.001)
block.compile(optimizer=optimizer,loss=['mean_squared_error','mean_squared_error'])
block.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_1 (InputLayer)           [(None, 10)]         0           []                               
                                                                                                      
     dense (Dense)                  (None, 128)          1408        ['input_1[0][0]']                
                                                                                                      
     dense_1 (Dense)                (None, 128)          16512       ['dense[0][0]']                  
                                                                                                      
     dense_2 (Dense)                (None, 128)          16512       ['dense_1[0][0]']                
                                                                                                      
     dense_3 (Dense)                (None, 128)          16512       ['dense_2[0][0]']                
                                                                                                      
     theta_b (Dense)                (None, 1)            129         ['dense_3[0][0]']                
                                                                                                      
     input_2 (InputLayer)           [(None, 10)]         0           []                               
                                                                                                      
     backcast (Dense)               (None, 10)           10          ['theta_b[0][0]']                
                                                                                                      
     theta_f (Dense)                (None, 1)            129         ['dense_3[0][0]']                
                                                                                                      
     subtract (Subtract)            (None, 10)           0           ['input_2[0][0]',                
                                                                      'backcast[0][0]']               
                                                                                                      
     forecast (Dense)               (None, 1)            1           ['theta_f[0][0]']                
                                                                                                      
    ==================================================================================================
    Total params: 51,213
    Trainable params: 51,213
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
K.clear_session()
early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)
epochs = 50
batch_size = 1

block.fit([train_x,train_x],[train_x,train_y], epochs=epochs, batch_size=1,callbacks=[early_stop])
          
```

    Epoch 1/50
    742/742 [==============================] - 1s 794us/step - loss: 0.0078 - subtract_loss: 1.4449e-05 - forecast_loss: 0.0078
    Epoch 2/50
    742/742 [==============================] - 1s 796us/step - loss: 0.0043 - subtract_loss: 1.0551e-06 - forecast_loss: 0.0043
    Epoch 3/50
    742/742 [==============================] - 1s 799us/step - loss: 0.0038 - subtract_loss: 1.2249e-06 - forecast_loss: 0.0038
    Epoch 4/50
    742/742 [==============================] - 1s 812us/step - loss: 0.0038 - subtract_loss: 2.7787e-07 - forecast_loss: 0.0038
    Epoch 5/50
    742/742 [==============================] - 1s 799us/step - loss: 0.0038 - subtract_loss: 3.0886e-07 - forecast_loss: 0.0038
    Epoch 6/50
    742/742 [==============================] - 1s 801us/step - loss: 0.0036 - subtract_loss: 2.4266e-07 - forecast_loss: 0.0036
    Epoch 7/50
    742/742 [==============================] - 1s 798us/step - loss: 0.0037 - subtract_loss: 5.2790e-08 - forecast_loss: 0.0037
    Epoch 8/50
    742/742 [==============================] - 1s 797us/step - loss: 0.0037 - subtract_loss: 2.5302e-08 - forecast_loss: 0.0037
    Epoch 9/50
    742/742 [==============================] - 1s 798us/step - loss: 0.0034 - subtract_loss: 1.0877e-08 - forecast_loss: 0.0034
    Epoch 10/50
    742/742 [==============================] - 1s 798us/step - loss: 0.0035 - subtract_loss: 1.5300e-08 - forecast_loss: 0.0035
    Epoch 11/50
    742/742 [==============================] - 1s 804us/step - loss: 0.0038 - subtract_loss: 1.6409e-10 - forecast_loss: 0.0038
    Epoch 12/50
    742/742 [==============================] - 1s 799us/step - loss: 0.0035 - subtract_loss: 9.4212e-14 - forecast_loss: 0.0035
    Epoch 12: early stopping
    




    <keras.callbacks.History at 0x294a2e46380>




```python
pred_ = block.predict([test_x,test_x])
pred_[0].shape, pred_[1].shape
```

    8/8 [==============================] - 0s 855us/step
    




    ((248, 10), (248, 1))




```python
plt.plot(test_x.flatten(),label='x')
plt.plot(pred_[0].flatten(), label='backcast', linewidth = 0.5)
plt.legend() 
plt.show()
```


    
![png](output_53_0.png)
    



```python
plt.plot(test_y,label='y')
plt.plot(pred_[1].flatten(),label='forecast_y')

plt.legend() 
plt.show()
```


    
![png](output_54_0.png)
    



```python
from sklearn.metrics import mean_squared_error

mse1 = mean_squared_error(test_y,pred_[1].flatten())

print('{:.10f}'.format(mse1))
```

    0.0072574279
    

---

## block stacking 구현

- block 2개를 연결

![image.png](attachment:image.png)


```python
# time step 
backcast_length = 10
# horizen
forecast_length = 1
feature1 = 1
# block setting
unit = 128
theta_dim1,theta_dim2= 1,1

K.clear_session()
#############################################################################################
#### blcok 1
input_x_1 = Input(shape=(backcast_length))
input_x_2 = Input(shape=(backcast_length))

d1 = Dense(unit,activation = 'relu')(input_x_1)
d2 = Dense(unit,activation = 'relu')(d1)
d3 = Dense(unit,activation = 'relu')(d2)
d4 = Dense(unit,activation = 'relu')(d3)

theta_b_1 = Dense(theta_dim1, activation='linear', name='theta_b')(d4)
theta_f_1 = Dense(theta_dim2, activation='linear',  name='theta_f')(d4)

backcast_1 = Dense(backcast_length , activation='linear', use_bias=False,name='backcast')(theta_b_1)
forecast_1 = Dense(forecast_length, activation='linear', use_bias=False,name='forecast')(theta_f_1)

subtract_1 = Subtract()([input_x_2 , backcast_1])
#############################################################################################
#### block 2
d1_2 = Dense(unit,activation = 'relu')(subtract_1 )
d2_2 = Dense(unit,activation = 'relu')(d1_2)
d3_2 = Dense(unit,activation = 'relu')(d2_2)
d4_2 = Dense(unit,activation = 'relu')(d3_2)

theta_b_2 = Dense(theta_dim1, activation='linear', name='theta_b2')(d4_2)
theta_f_2 = Dense(theta_dim2, activation='linear',  name='theta_f2')(d4_2)

backcast_2 = Dense(backcast_length , activation='linear', use_bias=False,name='backcast2')(theta_b_2)
forecast_2 = Dense(forecast_length, activation='linear', use_bias=False,name='forecast2')(theta_f_2)

subtract_2 = Subtract()([subtract_1 , backcast_2]) # 다음 블락에 연결되어짐

output = Add()([forecast_1,forecast_2])
###############################################################################################
#### model
stack = Model(inputs=[input_x_1,input_x_2],outputs = [subtract_2 ,output])
optimizer = Adam(learning_rate=0.001)
stack.compile(optimizer=optimizer,loss=['mean_squared_error','mean_squared_error'])
stack.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_1 (InputLayer)           [(None, 10)]         0           []                               
                                                                                                      
     dense (Dense)                  (None, 128)          1408        ['input_1[0][0]']                
                                                                                                      
     dense_1 (Dense)                (None, 128)          16512       ['dense[0][0]']                  
                                                                                                      
     dense_2 (Dense)                (None, 128)          16512       ['dense_1[0][0]']                
                                                                                                      
     dense_3 (Dense)                (None, 128)          16512       ['dense_2[0][0]']                
                                                                                                      
     theta_b (Dense)                (None, 1)            129         ['dense_3[0][0]']                
                                                                                                      
     input_2 (InputLayer)           [(None, 10)]         0           []                               
                                                                                                      
     backcast (Dense)               (None, 10)           10          ['theta_b[0][0]']                
                                                                                                      
     subtract (Subtract)            (None, 10)           0           ['input_2[0][0]',                
                                                                      'backcast[0][0]']               
                                                                                                      
     dense_4 (Dense)                (None, 128)          1408        ['subtract[0][0]']               
                                                                                                      
     dense_5 (Dense)                (None, 128)          16512       ['dense_4[0][0]']                
                                                                                                      
     dense_6 (Dense)                (None, 128)          16512       ['dense_5[0][0]']                
                                                                                                      
     dense_7 (Dense)                (None, 128)          16512       ['dense_6[0][0]']                
                                                                                                      
     theta_b2 (Dense)               (None, 1)            129         ['dense_7[0][0]']                
                                                                                                      
     theta_f (Dense)                (None, 1)            129         ['dense_3[0][0]']                
                                                                                                      
     theta_f2 (Dense)               (None, 1)            129         ['dense_7[0][0]']                
                                                                                                      
     backcast2 (Dense)              (None, 10)           10          ['theta_b2[0][0]']               
                                                                                                      
     forecast (Dense)               (None, 1)            1           ['theta_f[0][0]']                
                                                                                                      
     forecast2 (Dense)              (None, 1)            1           ['theta_f2[0][0]']               
                                                                                                      
     subtract_1 (Subtract)          (None, 10)           0           ['subtract[0][0]',               
                                                                      'backcast2[0][0]']              
                                                                                                      
     add (Add)                      (None, 1)            0           ['forecast[0][0]',               
                                                                      'forecast2[0][0]']              
                                                                                                      
    ==================================================================================================
    Total params: 102,426
    Trainable params: 102,426
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
K.clear_session()
early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1)
epochs = 50
batch_size = 1

stack.fit([train_x,train_x],[train_x,train_y], epochs=epochs, batch_size=1,callbacks=[early_stop])
          
```

    Epoch 1/50
    742/742 [==============================] - 1s 1ms/step - loss: 0.0067 - subtract_1_loss: 5.7966e-05 - add_loss: 0.0066
    Epoch 2/50
    742/742 [==============================] - 1s 993us/step - loss: 0.0045 - subtract_1_loss: 8.8713e-06 - add_loss: 0.0045
    Epoch 3/50
    742/742 [==============================] - 1s 997us/step - loss: 0.0041 - subtract_1_loss: 3.2973e-06 - add_loss: 0.0041
    Epoch 4/50
    742/742 [==============================] - 1s 997us/step - loss: 0.0039 - subtract_1_loss: 2.8451e-06 - add_loss: 0.0039
    Epoch 5/50
    742/742 [==============================] - 1s 997us/step - loss: 0.0038 - subtract_1_loss: 1.9567e-06 - add_loss: 0.0038
    Epoch 6/50
    742/742 [==============================] - 1s 992us/step - loss: 0.0037 - subtract_1_loss: 1.6847e-06 - add_loss: 0.0037
    Epoch 7/50
    742/742 [==============================] - 1s 992us/step - loss: 0.0036 - subtract_1_loss: 1.6103e-06 - add_loss: 0.0036
    Epoch 8/50
    742/742 [==============================] - 1s 991us/step - loss: 0.0038 - subtract_1_loss: 2.7884e-06 - add_loss: 0.0038
    Epoch 9/50
    742/742 [==============================] - 1s 993us/step - loss: 0.0039 - subtract_1_loss: 1.9291e-06 - add_loss: 0.0039
    Epoch 10/50
    742/742 [==============================] - 1s 992us/step - loss: 0.0036 - subtract_1_loss: 8.2038e-07 - add_loss: 0.0036
    Epoch 11/50
    742/742 [==============================] - 1s 991us/step - loss: 0.0035 - subtract_1_loss: 8.8098e-07 - add_loss: 0.0035
    Epoch 12/50
    742/742 [==============================] - 1s 993us/step - loss: 0.0036 - subtract_1_loss: 4.3042e-07 - add_loss: 0.0036
    Epoch 13/50
    742/742 [==============================] - 1s 995us/step - loss: 0.0037 - subtract_1_loss: 1.1768e-06 - add_loss: 0.0037
    Epoch 14/50
    742/742 [==============================] - 1s 1ms/step - loss: 0.0034 - subtract_1_loss: 7.3131e-07 - add_loss: 0.0034
    Epoch 15/50
    742/742 [==============================] - 1s 1ms/step - loss: 0.0033 - subtract_1_loss: 1.0159e-06 - add_loss: 0.0033
    Epoch 16/50
    742/742 [==============================] - 1s 996us/step - loss: 0.0034 - subtract_1_loss: 1.1043e-06 - add_loss: 0.0034
    Epoch 17/50
    742/742 [==============================] - 1s 997us/step - loss: 0.0035 - subtract_1_loss: 6.4103e-07 - add_loss: 0.0035
    Epoch 18/50
    742/742 [==============================] - 1s 1ms/step - loss: 0.0034 - subtract_1_loss: 3.0144e-07 - add_loss: 0.0034
    Epoch 18: early stopping
    




    <keras.callbacks.History at 0x2948e817640>




```python
pred = stack.predict([test_x,test_x])
```

    8/8 [==============================] - 0s 998us/step
    


```python
pred[0].shape, pred[1].shape
```




    ((248, 10), (248, 1))



## `-` subtract / x


```python
plt.plot(test_x.flatten(),label='x')
plt.plot(pred[0].flatten(), label='backcast', linewidth = 0.5)
plt.legend() 
plt.show()
```


    
![png](output_64_0.png)
    


## `-` forecast / y


```python
plt.plot(test_y,label='y')
plt.plot(pred[1].flatten(),label='forecast_y')

plt.legend() 
plt.show()
```


    
![png](output_66_0.png)
    



```python
from sklearn.metrics import mean_squared_error

mse1 = mean_squared_error(test_y,pred[1].flatten())

print('{:.10f}'.format(mse1))
```

    0.0034077407
    

---
