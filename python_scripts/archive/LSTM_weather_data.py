#!venv/bin/python3

'''
    Author: Massimo Clementi
    Date:   2021-04-07
    
    Predict next-day rain by training classification models on the target variable RainTomorrow.
    This dataset contains about 10 years of daily weather observations from many locations across Australia.
    RainTomorrow is the target variable to predict. It means -- did it rain the next day, Yes or No?
    This column is Yes if the rain for that day was 1mm or more.
    
    ! THE CODE DOES NOT WORK AT THE MOMENT

'''


# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import urllib.request
from datetime import datetime
import os

# %% Load and prepare dataset

# Download dataset here:
#   https://www.kaggle.com/jsphyg/weather-dataset-rattle-package/download

data_path = '../datasets/weatherAUS.csv'
assert os.path.exists(data_path)
df = pd.read_csv(data_path)

df = df[[
    'Date', 'Location','MinTemp', 'MaxTemp',
    'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm',
    'Temp9am', 'Temp3pm',
    'RainToday', 'RainTomorrow',
    'Rainfall'
]]

# Rough normalization
df.iloc[:,2:] = (df.iloc[:,2:] - df.iloc[:,2:].max()) / df.iloc[:,2:].std()

# Define dictionaries
cities = df.Location.unique()
cities_to_num = {v:k for k,v in enumerate(cities)}
YesNo_to_num = {
    'No': 0,
    'Yes': 1
}

df.Location = df.Location.map(cities_to_num)
df.RainToday = df.RainToday.map(YesNo_to_num)
df.RainTomorrow = df.RainTomorrow.map(YesNo_to_num)

data = []
for city in range(len(cities)):
    to_add = df[df.Location == city].to_numpy()[:,1:]
    # Do not add cities with low number of training examples
    if to_add.shape[0] >= 3000:
        data.append(to_add[:3000,:])
data = np.array(data).astype(np.float)

# Define train and test sets
x_train = data[:,:1998,:]
y_train = data[:,1999,:]
x_test = data[:,2000:2998,:]
y_test = data[:,2999,:]



# %% Define NN
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(
        units=12,   # output dim is full input
        activation='linear',
        recurrent_activation='sigmoid',
        input_shape=(3000,12)
    )
])

#print(model.output_shape)
model.summary()


# %% Compile, train and evaluate model
loss_fn = tf.keras.losses.CategoricalCrossentropy()
model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy']    
)

''' Early stopping to prevent overfitting '''
def get_callbacks():
  return [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=150)
  ]

''' Train the model '''
print("Training started at", datetime.now())
history = model.fit(x_train,
          y_train,
          epochs=10,
          callbacks=get_callbacks()
)
print("Training ended at", datetime.now())

model.save_weights(   #save trained weights
    filepath='../saved_models/3_LSTM/LSTM',
    overwrite=True,
)

''' Visualize performances over the epochs '''
ax = sns.lineplot(
    x=history.epoch,
    y=history.history['accuracy'],
    color='green',
    label='accuracy'
)
ax2 = ax.twinx()
sns.lineplot(
    x=history.epoch,
    y=history.history['loss'],
    label='loss',
    color='red',
    ax=ax2
)

model.evaluate(x_test, y_test, verbose=2)
# %%
