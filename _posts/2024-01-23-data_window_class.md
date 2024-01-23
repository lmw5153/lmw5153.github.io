
# data window class 



- 단변량 데이터인 경우 모델에 적합하기 위해서 차원을 넓혀야 하는 경우가 있음
- 따라서 data를 input 차원에 맞추기 위해 window처럼 사각형의 틀을 만들어야한다.

---


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

---

`-` example data

- 1차원의 단변량 데이터인 경우
- timestep에 따라 window size를 만듦
- numpy 변수에 pandas 데이터프레임으로 가공


```python
x = np.arange(1,20)

df = pd.DataFrame({'y':x})
```

`-` window function



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
test = WINdow(df,5)
test.window().head()
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
      <th>shift_5</th>
      <th>shift_4</th>
      <th>shift_3</th>
      <th>shift_2</th>
      <th>shift_1</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>6</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7</td>
    </tr>
    <tr>
      <td>7</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>8</td>
    </tr>
    <tr>
      <td>8</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>9</td>
    </tr>
    <tr>
      <td>9</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



- y란 단일 값에 대해 tiemstep에 따른 sequence를 생성

- 생성된 window에 마지막 열이 y_label

`-` feature


```python
test.feature
```




    array([[ 1.,  2.,  3.,  4.,  5.],
           [ 2.,  3.,  4.,  5.,  6.],
           [ 3.,  4.,  5.,  6.,  7.],
           [ 4.,  5.,  6.,  7.,  8.],
           [ 5.,  6.,  7.,  8.,  9.],
           [ 6.,  7.,  8.,  9., 10.],
           [ 7.,  8.,  9., 10., 11.],
           [ 8.,  9., 10., 11., 12.],
           [ 9., 10., 11., 12., 13.],
           [10., 11., 12., 13., 14.],
           [11., 12., 13., 14., 15.],
           [12., 13., 14., 15., 16.],
           [13., 14., 15., 16., 17.],
           [14., 15., 16., 17., 18.]])



`-` y_label


```python
test.y_label
```




    array([ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])


