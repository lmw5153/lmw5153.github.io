# 단변량 시계열 예측 n-beats model

- 코드출처 : https://github.com/philipperemy/n-beats/tree/master
- 관련논문 : https://arxiv.org/pdf/1905.10437.pdf?trk=public_post_comment-text

- 주식데이터의 시작가로 종가를 예측하는 모형


```python
from nbeats_keras.model import NBeatsNet as NBeatsKeras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch
from keras.optimizers import RMSprop, Adam
import yfinance as yf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import time
```

# 예제 데이터


```python
start_date = '2022-12-31'
end_date ='2024-12-31'

ticker = yf.Ticker('005930.KS')
 
df1= ticker.history(
               interval='1h',
               start=start_date,
               end=end_date,
               actions=True,
               auto_adjust=True)

X = df1.iloc[:,:4].loc[:'2023-11-30 14:00:00+09:00',:]
Y = df1.iloc[:,:4].loc['2023-12-01 10:00:00+09:00':,:]

X, Y = np.log(X),np.log(Y)
```


```python
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
```

### train set 


```python
train= X[['Close']]
train1= X[['Open']]
win= WINdow(train,5)
win.window()
win1= WINdow(train1,5)
win1.window()
X_train = win1.feature
y_train = win.y_label

from sklearn.preprocessing import MinMaxScaler, Normalizer

minmax = MinMaxScaler()
norm = Normalizer()
X_train_scale= norm.fit_transform(X_train)
y_train_scale= norm.fit_transform(y_train.reshape(-1, 1) )
```

    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    

### test set


```python
test= Y[['Close']]
test1= Y[['Open']]
wint= WINdow(test,5)
wint.window()
wint1= WINdow(test1,5)
wint1.window()
X_test = wint1.feature
y_test = wint.y_label


X_test_scale= norm.fit_transform(X_test)
y_test_scale= norm.fit_transform(y_test.reshape(-1, 1) )
```

    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    C:\Users\Public\Documents\ESTsoft\CreatorTemp\ipykernel_16968\1267832976.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self.df['shift_{}'.format(i)] = self.df.iloc[:,0].shift(i)
    

---

#### model 생성

- nb_blocks_per_stack : stack 안 블락의 갯수
- 각 블락 추출한 theta의 갯수 : (4,4,4)
- input_data_shape = (length, tiemstep, 1)
- 파라미터 셋팅 : loss의 변화가 안정적으로 감소하는 기준으로 셋팅

`-` model
![image](https://github.com/lmw5153/lmw5153.github.io/assets/154956154/f2dcf494-94ad-4421-88a2-c4631704f632)


`-` source code
![image](https://github.com/lmw5153/lmw5153.github.io/assets/154956154/737ef2aa-315f-4906-abf1-dd136de0553b)

![image](https://github.com/lmw5153/lmw5153.github.io/assets/154956154/806efe23-45b5-4f1f-a921-903f4ae72fc1)



```python
time_steps, input_dim, output_dim = 5, 1, 1
K.clear_session()
model= NBeatsKeras( backcast_length=time_steps, forecast_length=output_dim,
            stack_types=(NBeatsKeras.GENERIC_BLOCK,NBeatsKeras.TREND_BLOCK, NBeatsKeras.SEASONALITY_BLOCK),
            nb_blocks_per_stack=5, thetas_dim=(4,4,4),share_weights_in_stack=True,
            hidden_layer_units=128)
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='mse', optimizer=optimizer )
model.summary()
```

    Model: "forecast"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_variable (InputLayer)    [(None, 5, 1)]       0           []                               
                                                                                                      
     lambda (Lambda)                (None, 5)            0           ['input_variable[0][0]']         
                                                                                                      
     0/0/generic/d1 (Dense)         (None, 128)          768         ['lambda[0][0]',                 
                                                                      'subtract[0][0]',               
                                                                      'subtract_1[0][0]',             
                                                                      'subtract_2[0][0]',             
                                                                      'subtract_3[0][0]']             
                                                                                                      
     0/0/generic/d2 (Dense)         (None, 128)          16512       ['0/0/generic/d1[0][0]',         
                                                                      '0/0/generic/d1[1][0]',         
                                                                      '0/0/generic/d1[2][0]',         
                                                                      '0/0/generic/d1[3][0]',         
                                                                      '0/0/generic/d1[4][0]']         
                                                                                                      
     0/0/generic/d3 (Dense)         (None, 128)          16512       ['0/0/generic/d2[0][0]',         
                                                                      '0/0/generic/d2[1][0]',         
                                                                      '0/0/generic/d2[2][0]',         
                                                                      '0/0/generic/d2[3][0]',         
                                                                      '0/0/generic/d2[4][0]']         
                                                                                                      
     0/0/generic/d4 (Dense)         (None, 128)          16512       ['0/0/generic/d3[0][0]',         
                                                                      '0/0/generic/d3[1][0]',         
                                                                      '0/0/generic/d3[2][0]',         
                                                                      '0/0/generic/d3[3][0]',         
                                                                      '0/0/generic/d3[4][0]']         
                                                                                                      
     0/0/generic/theta_b (Dense)    (None, 4)            512         ['0/0/generic/d4[0][0]',         
                                                                      '0/0/generic/d4[1][0]',         
                                                                      '0/0/generic/d4[2][0]',         
                                                                      '0/0/generic/d4[3][0]',         
                                                                      '0/0/generic/d4[4][0]']         
                                                                                                      
     0/0/generic/backcast (Dense)   (None, 5)            25          ['0/0/generic/theta_b[0][0]',    
                                                                      '0/0/generic/theta_b[1][0]',    
                                                                      '0/0/generic/theta_b[2][0]',    
                                                                      '0/0/generic/theta_b[3][0]',    
                                                                      '0/0/generic/theta_b[4][0]']    
                                                                                                      
     subtract (Subtract)            (None, 5)            0           ['lambda[0][0]',                 
                                                                      '0/0/generic/backcast[0][0]']   
                                                                                                      
     subtract_1 (Subtract)          (None, 5)            0           ['subtract[0][0]',               
                                                                      '0/0/generic/backcast[1][0]']   
                                                                                                      
     subtract_2 (Subtract)          (None, 5)            0           ['subtract_1[0][0]',             
                                                                      '0/0/generic/backcast[2][0]']   
                                                                                                      
     subtract_3 (Subtract)          (None, 5)            0           ['subtract_2[0][0]',             
                                                                      '0/0/generic/backcast[3][0]']   
                                                                                                      
     subtract_4 (Subtract)          (None, 5)            0           ['subtract_3[0][0]',             
                                                                      '0/0/generic/backcast[4][0]']   
                                                                                                      
     1/0/trend/d1 (Dense)           (None, 128)          768         ['subtract_4[0][0]',             
                                                                      'subtract_5[0][0]',             
                                                                      'subtract_6[0][0]',             
                                                                      'subtract_7[0][0]',             
                                                                      'subtract_8[0][0]']             
                                                                                                      
     1/0/trend/d2 (Dense)           (None, 128)          16512       ['1/0/trend/d1[0][0]',           
                                                                      '1/0/trend/d1[1][0]',           
                                                                      '1/0/trend/d1[2][0]',           
                                                                      '1/0/trend/d1[3][0]',           
                                                                      '1/0/trend/d1[4][0]']           
                                                                                                      
     1/0/trend/d3 (Dense)           (None, 128)          16512       ['1/0/trend/d2[0][0]',           
                                                                      '1/0/trend/d2[1][0]',           
                                                                      '1/0/trend/d2[2][0]',           
                                                                      '1/0/trend/d2[3][0]',           
                                                                      '1/0/trend/d2[4][0]']           
                                                                                                      
     1/0/trend/d4 (Dense)           (None, 128)          16512       ['1/0/trend/d3[0][0]',           
                                                                      '1/0/trend/d3[1][0]',           
                                                                      '1/0/trend/d3[2][0]',           
                                                                      '1/0/trend/d3[3][0]',           
                                                                      '1/0/trend/d3[4][0]']           
                                                                                                      
     1/0/trend/theta_f_b (Dense)    (None, 4)            512         ['1/0/trend/d4[0][0]',           
                                                                      '1/0/trend/d4[0][0]',           
                                                                      '1/0/trend/d4[1][0]',           
                                                                      '1/0/trend/d4[1][0]',           
                                                                      '1/0/trend/d4[2][0]',           
                                                                      '1/0/trend/d4[2][0]',           
                                                                      '1/0/trend/d4[3][0]',           
                                                                      '1/0/trend/d4[3][0]',           
                                                                      '1/0/trend/d4[4][0]',           
                                                                      '1/0/trend/d4[4][0]']           
                                                                                                      
     lambda_1 (Lambda)              (None, 5)            0           ['1/0/trend/theta_f_b[1][0]']    
                                                                                                      
     subtract_5 (Subtract)          (None, 5)            0           ['subtract_4[0][0]',             
                                                                      'lambda_1[0][0]']               
                                                                                                      
     lambda_3 (Lambda)              (None, 5)            0           ['1/0/trend/theta_f_b[3][0]']    
                                                                                                      
     subtract_6 (Subtract)          (None, 5)            0           ['subtract_5[0][0]',             
                                                                      'lambda_3[0][0]']               
                                                                                                      
     lambda_5 (Lambda)              (None, 5)            0           ['1/0/trend/theta_f_b[5][0]']    
                                                                                                      
     subtract_7 (Subtract)          (None, 5)            0           ['subtract_6[0][0]',             
                                                                      'lambda_5[0][0]']               
                                                                                                      
     lambda_7 (Lambda)              (None, 5)            0           ['1/0/trend/theta_f_b[7][0]']    
                                                                                                      
     subtract_8 (Subtract)          (None, 5)            0           ['subtract_7[0][0]',             
                                                                      'lambda_7[0][0]']               
                                                                                                      
     lambda_9 (Lambda)              (None, 5)            0           ['1/0/trend/theta_f_b[9][0]']    
                                                                                                      
     subtract_9 (Subtract)          (None, 5)            0           ['subtract_8[0][0]',             
                                                                      'lambda_9[0][0]']               
                                                                                                      
     2/0/seasonality/d1 (Dense)     (None, 128)          768         ['subtract_9[0][0]',             
                                                                      'subtract_10[0][0]',            
                                                                      'subtract_11[0][0]',            
                                                                      'subtract_12[0][0]',            
                                                                      'subtract_13[0][0]']            
                                                                                                      
     2/0/seasonality/d2 (Dense)     (None, 128)          16512       ['2/0/seasonality/d1[0][0]',     
                                                                      '2/0/seasonality/d1[1][0]',     
                                                                      '2/0/seasonality/d1[2][0]',     
                                                                      '2/0/seasonality/d1[3][0]',     
                                                                      '2/0/seasonality/d1[4][0]']     
                                                                                                      
     2/0/seasonality/d3 (Dense)     (None, 128)          16512       ['2/0/seasonality/d2[0][0]',     
                                                                      '2/0/seasonality/d2[1][0]',     
                                                                      '2/0/seasonality/d2[2][0]',     
                                                                      '2/0/seasonality/d2[3][0]',     
                                                                      '2/0/seasonality/d2[4][0]']     
                                                                                                      
     2/0/seasonality/d4 (Dense)     (None, 128)          16512       ['2/0/seasonality/d3[0][0]',     
                                                                      '2/0/seasonality/d3[1][0]',     
                                                                      '2/0/seasonality/d3[2][0]',     
                                                                      '2/0/seasonality/d3[3][0]',     
                                                                      '2/0/seasonality/d3[4][0]']     
                                                                                                      
     2/0/seasonality/theta_b (Dense  (None, 1)           128         ['2/0/seasonality/d4[0][0]',     
     )                                                                '2/0/seasonality/d4[1][0]',     
                                                                      '2/0/seasonality/d4[2][0]',     
                                                                      '2/0/seasonality/d4[3][0]']     
                                                                                                      
     lambda_11 (Lambda)             (None, 5)            0           ['2/0/seasonality/theta_b[0][0]']
                                                                                                      
     subtract_10 (Subtract)         (None, 5)            0           ['subtract_9[0][0]',             
                                                                      'lambda_11[0][0]']              
                                                                                                      
     lambda_13 (Lambda)             (None, 5)            0           ['2/0/seasonality/theta_b[1][0]']
                                                                                                      
     subtract_11 (Subtract)         (None, 5)            0           ['subtract_10[0][0]',            
                                                                      'lambda_13[0][0]']              
                                                                                                      
     0/0/generic/theta_f (Dense)    (None, 4)            512         ['0/0/generic/d4[0][0]',         
                                                                      '0/0/generic/d4[1][0]',         
                                                                      '0/0/generic/d4[2][0]',         
                                                                      '0/0/generic/d4[3][0]',         
                                                                      '0/0/generic/d4[4][0]']         
                                                                                                      
     lambda_15 (Lambda)             (None, 5)            0           ['2/0/seasonality/theta_b[2][0]']
                                                                                                      
     0/0/generic/forecast (Dense)   (None, 1)            5           ['0/0/generic/theta_f[0][0]',    
                                                                      '0/0/generic/theta_f[1][0]',    
                                                                      '0/0/generic/theta_f[2][0]',    
                                                                      '0/0/generic/theta_f[3][0]',    
                                                                      '0/0/generic/theta_f[4][0]']    
                                                                                                      
     subtract_12 (Subtract)         (None, 5)            0           ['subtract_11[0][0]',            
                                                                      'lambda_15[0][0]']              
                                                                                                      
     stack_0-GenericBlock_0_Dim_0 (  (None, 1)           0           ['0/0/generic/forecast[0][0]']   
     Lambda)                                                                                          
                                                                                                      
     stack_0-GenericBlock_1_Dim_0 (  (None, 1)           0           ['0/0/generic/forecast[1][0]']   
     Lambda)                                                                                          
                                                                                                      
     add (Add)                      (None, 1)            0           ['stack_0-GenericBlock_0_Dim_0[0]
                                                                     [0]',                            
                                                                      'stack_0-GenericBlock_1_Dim_0[0]
                                                                     [0]']                            
                                                                                                      
     stack_0-GenericBlock_2_Dim_0 (  (None, 1)           0           ['0/0/generic/forecast[2][0]']   
     Lambda)                                                                                          
                                                                                                      
     add_1 (Add)                    (None, 1)            0           ['add[0][0]',                    
                                                                      'stack_0-GenericBlock_2_Dim_0[0]
                                                                     [0]']                            
                                                                                                      
     stack_0-GenericBlock_3_Dim_0 (  (None, 1)           0           ['0/0/generic/forecast[3][0]']   
     Lambda)                                                                                          
                                                                                                      
     add_2 (Add)                    (None, 1)            0           ['add_1[0][0]',                  
                                                                      'stack_0-GenericBlock_3_Dim_0[0]
                                                                     [0]']                            
                                                                                                      
     stack_0-GenericBlock_4_Dim_0 (  (None, 1)           0           ['0/0/generic/forecast[4][0]']   
     Lambda)                                                                                          
                                                                                                      
     lambda_2 (Lambda)              (None, 1)            0           ['1/0/trend/theta_f_b[0][0]']    
                                                                                                      
     add_3 (Add)                    (None, 1)            0           ['add_2[0][0]',                  
                                                                      'stack_0-GenericBlock_4_Dim_0[0]
                                                                     [0]']                            
                                                                                                      
     stack_1-TrendBlock_0_Dim_0 (La  (None, 1)           0           ['lambda_2[0][0]']               
     mbda)                                                                                            
                                                                                                      
     lambda_4 (Lambda)              (None, 1)            0           ['1/0/trend/theta_f_b[2][0]']    
                                                                                                      
     add_4 (Add)                    (None, 1)            0           ['add_3[0][0]',                  
                                                                      'stack_1-TrendBlock_0_Dim_0[0][0
                                                                     ]']                              
                                                                                                      
     stack_1-TrendBlock_1_Dim_0 (La  (None, 1)           0           ['lambda_4[0][0]']               
     mbda)                                                                                            
                                                                                                      
     lambda_6 (Lambda)              (None, 1)            0           ['1/0/trend/theta_f_b[4][0]']    
                                                                                                      
     lambda_17 (Lambda)             (None, 5)            0           ['2/0/seasonality/theta_b[3][0]']
                                                                                                      
     add_5 (Add)                    (None, 1)            0           ['add_4[0][0]',                  
                                                                      'stack_1-TrendBlock_1_Dim_0[0][0
                                                                     ]']                              
                                                                                                      
     stack_1-TrendBlock_2_Dim_0 (La  (None, 1)           0           ['lambda_6[0][0]']               
     mbda)                                                                                            
                                                                                                      
     lambda_8 (Lambda)              (None, 1)            0           ['1/0/trend/theta_f_b[6][0]']    
                                                                                                      
     subtract_13 (Subtract)         (None, 5)            0           ['subtract_12[0][0]',            
                                                                      'lambda_17[0][0]']              
                                                                                                      
     add_6 (Add)                    (None, 1)            0           ['add_5[0][0]',                  
                                                                      'stack_1-TrendBlock_2_Dim_0[0][0
                                                                     ]']                              
                                                                                                      
     stack_1-TrendBlock_3_Dim_0 (La  (None, 1)           0           ['lambda_8[0][0]']               
     mbda)                                                                                            
                                                                                                      
     lambda_10 (Lambda)             (None, 1)            0           ['1/0/trend/theta_f_b[8][0]']    
                                                                                                      
     2/0/seasonality/theta_f (Dense  (None, 1)           128         ['2/0/seasonality/d4[0][0]',     
     )                                                                '2/0/seasonality/d4[1][0]',     
                                                                      '2/0/seasonality/d4[2][0]',     
                                                                      '2/0/seasonality/d4[3][0]',     
                                                                      '2/0/seasonality/d4[4][0]']     
                                                                                                      
     add_7 (Add)                    (None, 1)            0           ['add_6[0][0]',                  
                                                                      'stack_1-TrendBlock_3_Dim_0[0][0
                                                                     ]']                              
                                                                                                      
     stack_1-TrendBlock_4_Dim_0 (La  (None, 1)           0           ['lambda_10[0][0]']              
     mbda)                                                                                            
                                                                                                      
     lambda_12 (Lambda)             (None, 1)            0           ['2/0/seasonality/theta_f[0][0]']
                                                                                                      
     add_8 (Add)                    (None, 1)            0           ['add_7[0][0]',                  
                                                                      'stack_1-TrendBlock_4_Dim_0[0][0
                                                                     ]']                              
                                                                                                      
     stack_2-SeasonalityBlock_0_Dim  (None, 1)           0           ['lambda_12[0][0]']              
     _0 (Lambda)                                                                                      
                                                                                                      
     lambda_14 (Lambda)             (None, 1)            0           ['2/0/seasonality/theta_f[1][0]']
                                                                                                      
     add_9 (Add)                    (None, 1)            0           ['add_8[0][0]',                  
                                                                      'stack_2-SeasonalityBlock_0_Dim_
                                                                     0[0][0]']                        
                                                                                                      
     stack_2-SeasonalityBlock_1_Dim  (None, 1)           0           ['lambda_14[0][0]']              
     _0 (Lambda)                                                                                      
                                                                                                      
     lambda_16 (Lambda)             (None, 1)            0           ['2/0/seasonality/theta_f[2][0]']
                                                                                                      
     add_10 (Add)                   (None, 1)            0           ['add_9[0][0]',                  
                                                                      'stack_2-SeasonalityBlock_1_Dim_
                                                                     0[0][0]']                        
                                                                                                      
     stack_2-SeasonalityBlock_2_Dim  (None, 1)           0           ['lambda_16[0][0]']              
     _0 (Lambda)                                                                                      
                                                                                                      
     lambda_18 (Lambda)             (None, 1)            0           ['2/0/seasonality/theta_f[3][0]']
                                                                                                      
     add_11 (Add)                   (None, 1)            0           ['add_10[0][0]',                 
                                                                      'stack_2-SeasonalityBlock_2_Dim_
                                                                     0[0][0]']                        
                                                                                                      
     stack_2-SeasonalityBlock_3_Dim  (None, 1)           0           ['lambda_18[0][0]']              
     _0 (Lambda)                                                                                      
                                                                                                      
     lambda_20 (Lambda)             (None, 1)            0           ['2/0/seasonality/theta_f[4][0]']
                                                                                                      
     add_12 (Add)                   (None, 1)            0           ['add_11[0][0]',                 
                                                                      'stack_2-SeasonalityBlock_3_Dim_
                                                                     0[0][0]']                        
                                                                                                      
     stack_2-SeasonalityBlock_4_Dim  (None, 1)           0           ['lambda_20[0][0]']              
     _0 (Lambda)                                                                                      
                                                                                                      
     add_13 (Add)                   (None, 1)            0           ['add_12[0][0]',                 
                                                                      'stack_2-SeasonalityBlock_4_Dim_
                                                                     0[0][0]']                        
                                                                                                      
     reshape (Reshape)              (None, 1, 1)         0           ['add_13[0][0]']                 
                                                                                                      
    ==================================================================================================
    Total params: 152,734
    Trainable params: 152,734
    Non-trainable params: 0
    __________________________________________________________________________________________________
    

---

### model fitting


```python
early_stop = EarlyStopping(monitor='loss', patience=3, verbose=1, restore_best_weights=True)
#checkpoint = ModelCheckpoint('best_model.h9', monitor='loss', save_best_only=True, mode='max', verbose=1)
#model.save('best_model.h9')
history = model.fit(X_train_scale,y_train_scale ,
                    validation_data=(X_test_scale, y_test_scale),
                    epochs=50, batch_size=2,callbacks=[early_stop])
```

    Epoch 1/50
    675/675 [==============================] - 7s 7ms/step - loss: 0.0146 - val_loss: 1.2700e-09
    Epoch 2/50
    675/675 [==============================] - 4s 6ms/step - loss: 1.8586e-09 - val_loss: 1.2831e-09
    Epoch 3/50
    675/675 [==============================] - 4s 6ms/step - loss: 2.1762e-09 - val_loss: 1.7332e-09
    Epoch 4/50
    675/675 [==============================] - 4s 6ms/step - loss: 2.9603e-09 - val_loss: 1.2333e-09
    Epoch 5/50
    668/675 [============================>.] - ETA: 0s - loss: 3.3439e-09Restoring model weights from the end of the best epoch: 2.
    675/675 [==============================] - 4s 6ms/step - loss: 3.3339e-09 - val_loss: 2.6526e-09
    Epoch 5: early stopping
    

---

### loss


```python
# 훈련과 검증 손실 저장
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

![output_17_0](https://github.com/lmw5153/lmw5153.github.io/assets/154956154/61aad840-86e1-4487-a553-c130a49cf000)

    

    


---

### test


```python
pred =model.predict(X_train_scale)
pred1 = model.predict(X_test_scale)
```

    43/43 [==============================] - 1s 3ms/step
    10/10 [==============================] - 0s 3ms/step
    


```python
from sklearn.metrics import mean_squared_error
mse1= mean_squared_error(np.exp(y_train_scale.flatten()),np.exp(pred.flatten()))
mse2= mean_squared_error(np.exp(y_test_scale.flatten()),np.exp(pred1.flatten()))
print('train MSE{:.10f} '.format(mse1))
print('test MSE{:.10f} '.format(mse2))
```

    train MSE0.0000000104 
    test MSE0.0000000095 
    


```python
plt.plot(np.exp(y_test_scale.flatten()),label = 'y_observed')
plt.plot(np.exp(pred1.flatten()),label = 'y_pred')
plt.legend()
plt.show()
```


![output_22_0](https://github.com/lmw5153/lmw5153.github.io/assets/154956154/930b03bc-bd21-4e74-8267-0c704ed4d75c)


    

