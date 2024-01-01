# 딥러닝 과제

- 202350895
- 이민우


```python

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

# dataset 


```python
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path
```




    'C:\\Users\\minu\\.keras\\datasets\\auto-mpg.data'




```python
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.shape

```




    (398, 8)




```python
dataset = dataset.dropna()
origin = dataset.pop('Origin')
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
```


```python
dataset.shape
```




    (392, 9)




```python
y = dataset.pop("MPG")
```

---

# data 정규화


```python
col = dataset.columns[:]
```


```python
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
scaler_dataset = scaler.fit_transform(dataset)


dataset_ = pd.DataFrame(scaler_dataset,columns = col)
```

# train/test set 분리


```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dataset_, y,train_size=0.8 ,  random_state=1)
```

# model


```python
dataset_.shape
```




    (392, 8)



`-` relu_model


```python
def relu_model():
  model = keras.Sequential([

    layers.Dense(64, activation='relu', input_shape=[len(x_train.keys())]),
    layers.Dense(64, activation='relu'),  # 'linear' instead of 'relu'
    layers.Dense(64, activation='relu'),
    #layers.Dense(64, activation='relu'),
    layers.Dense(1) ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
```


```python
model1 = relu_model()
model1.summary()

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500
m1 = model1.fit(
  x_train, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
```

    Model: "sequential_21"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_66 (Dense)            (None, 64)                576       
                                                                     
     dense_67 (Dense)            (None, 64)                4160      
                                                                     
     dense_68 (Dense)            (None, 64)                4160      
                                                                     
     dense_69 (Dense)            (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 8961 (35.00 KB)
    Trainable params: 8961 (35.00 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................

`-` linear_model


```python
def linear_model():
  model = keras.Sequential([

    layers.Dense(64, activation='linear', input_shape=[len(x_train.keys())]),
    layers.Dense(64, activation='linear'),  # 'linear' instead of 'relu'
    layers.Dense(64, activation='linear'),
    #layers.Dense(64, activation='relu'),
    layers.Dense(1) ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
```


```python
model2 = linear_model()
model2.summary()

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500
m2 = model2.fit(
  x_train, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
```

    Model: "sequential_22"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_70 (Dense)            (None, 64)                576       
                                                                     
     dense_71 (Dense)            (None, 64)                4160      
                                                                     
     dense_72 (Dense)            (None, 64)                4160      
                                                                     
     dense_73 (Dense)            (None, 1)                 65        
                                                                     
    =================================================================
    Total params: 8961 (35.00 KB)
    Trainable params: 8961 (35.00 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................

# plot


```python
def plot_history(history):
  hist = pd.DataFrame(m1.history)
  hist['epoch'] = m1.epoch
  
  hist2 = pd.DataFrame(m2.history)
  hist2['epoch'] = m2.epoch
    
    
  plt.figure(figsize=(8,12))

  plt.subplot(4,1,1)
  plt.xlabel('relu_Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.subplot(4,1,2)
  plt.xlabel('relu_Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()

  plt.subplot(4,1,3)
  plt.xlabel('linear_Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist2['epoch'], hist2['mae'],
           label='Train Error')
  plt.plot(hist2['epoch'], hist2['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.subplot(4,1,4)
  plt.xlabel('linear_Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist2['epoch'], hist2['mse'],
           label='Train Error')
  plt.plot(hist2['epoch'], hist2['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  
  plt.show()

plot_history(history)
```


    
![output_23_0](https://github.com/lmw5153/lmw5153.github.io/assets/154956154/d96f10e3-957a-4e49-8747-4c4ded3c13c7)

    



```python
test_predictions = model1.predict(x_test).flatten()
test_predictions2 = model2.predict(x_test).flatten()

y1 = np.array(test_predictions)
#x1 = np.array(x_test["Weight"])
y2 = np.array(test_predictions2)
#x2 = np.array(x_test["Weight"])

print('relu_MSE', np.mean(y1-y_test)**2)
print('linear_MSE', np.mean(y2-y_test)**2)
```

    3/3 [==============================] - 0s 2ms/step
    3/3 [==============================] - 0s 2ms/step
    relu_MSE 0.04080340496695432
    linear_MSE 0.08269818007656253
    
