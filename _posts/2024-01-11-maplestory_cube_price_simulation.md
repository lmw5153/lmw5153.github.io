```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

# 1. 윗잠 레전등업 큐브 시뮬레이션

- 1.4%로 레전드리 등급업 가정
- 총 108번 독립시행을 100 번 반복
- 108번 동안 등급업 옵션이 뜨지 않는 경우 천장을 채운 것으로 해석함 


```python
p = 0.014
n = np.arange(1,100)
lst = []
for i in n:
    cube = np.random.choice(['등급업','유지'],108,p=[p,1-p])
    lst.append(cube)
```


```python
lst2= []
for i in n-1:
    df = pd.DataFrame({'cube':lst[i]}).reset_index()
    time = df.query("cube == '등급업'")
    lst2.append(time.index)
```


```python
lst3 = []

for i in n-1:
    if sum(lst[i] == '등급업') == 0:
        lst3.append(108)
    else :
        df = pd.DataFrame({'n':np.arange(1,len(lst[0])+1),'cube':lst[i]})
        t = df.query("cube=='등급업'").iloc[0,0]
        lst3.append(t)

```

# 2. 무통, 메소 큐브 가격


```python
        
# 무통메소, 큐브가격, 큐브메소가격        
mutongprice = 2780, 5000
cubeprice = 2200
cubemesoprice = [4590] # 200제 기준
# 단위 원
mepo=np.array(lst3)*2200
meso=np.array(lst3)*(cubemesoprice[0]/10000 *mutongprice[0])
meso_max=np.array(lst3)*(cubemesoprice[0]/10000 *mutongprice[1])
```


```python
df_price= pd.DataFrame({'mesocube':meso,'mepocube':mepo,'meso_max':meso_max})
df_price.describe()
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
      <th>mesocube</th>
      <th>mepocube</th>
      <th>meso_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>99.000000</td>
      <td>99.000000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>69613.980000</td>
      <td>120022.222222</td>
      <td>125205.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>49238.576821</td>
      <td>84892.767360</td>
      <td>88558.591406</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1276.020000</td>
      <td>2200.000000</td>
      <td>2295.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>24882.390000</td>
      <td>42900.000000</td>
      <td>44752.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>66353.040000</td>
      <td>114400.000000</td>
      <td>119340.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>118031.850000</td>
      <td>203500.000000</td>
      <td>212287.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>137810.160000</td>
      <td>237600.000000</td>
      <td>247860.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.hist(df_price);
```



    
![output_8_0](https://github.com/lmw5153/lmw5153.github.io/assets/154956154/964f163f-8eae-463a-81d6-66805446808c)


- 무통 가격이 5000원이 되어야 기존 블랙큐브 2200원일 때 레전등급업에 필요한 현금이 든다.
- 향후 무통가격이 얼만큼 치솟을지는 모르겠지만 당분간은 레전드리 접근이 훨씬 쉬울 것으로 보임
