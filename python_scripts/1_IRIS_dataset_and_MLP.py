#!venv/bin/python3

'''
    Author: Massimo Clementi
    Date:   2021-04-04
    
    Display the 4D IRIS dataset and train a simple MLP on it 

'''

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import urllib.request

# %% Download, show and prepare the Iris dataset
iris_dataset_url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
data = pd.read_csv(
    urllib.request.urlretrieve(iris_dataset_url)[0]
)

_ = sns.pairplot(data, hue="species", height= 2.5)

data_norm = pd.DataFrame(data)

''' Normalize features '''
for feature in data.columns[:-1]:
    data_norm[feature] = (data[feature] - data[feature].mean()) /  data[feature].std()

''' Map each string label to one-hot encode '''
labels2num = {
    'setosa': np.array([[1, 0, 0]]),
    'versicolor': np.array([[0, 1, 0]]),
    'virginica': np.array([[0, 0, 1]])
}
data_norm.species = data.species.map(labels2num)

print(data_norm.mean())
print(data_norm.std())

''' Create shuffled train and test sets '''
n_samples, dim_samples = data_norm.shape
data_norm = data_norm.sample(frac=1)    #shuffle
x_train = data_norm.head(int(n_samples * 2/3))[data_norm.columns[:-1]].to_numpy()
y_train = data_norm.head(int(n_samples * 2/3))['species'].to_numpy()
y_train = np.squeeze(np.stack(y_train)) # reshape from list of ndarray to ndarray
x_test = data_norm.tail(int(n_samples * 1/3))[data_norm.columns[:-1]].to_numpy()
y_test = data_norm.tail(int(n_samples * 1/3))['species'].to_numpy()
y_test = np.squeeze(np.stack(y_test))   # reshape from list of ndarray to ndarray

print('x_train:',x_train.shape,'y_train:', y_train.shape)
print('x_test:', x_test.shape, 'y_test:', y_test.shape)
print('x_train min:',x_train.min(),'x_train max:', x_train.max())


# %% Define model

''' Define NN'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1,4)),
    tf.keras.layers.Dense(23, activation='tanh'),
    tf.keras.layers.Dense(7, activation='tanh'),
    tf.keras.layers.Dense(3, activation='tanh'),
    tf.keras.layers.Softmax()
])

print('output_shape=', model.output_shape)


# %% Train NN

''' Define loss function and optimizer'''
loss_fn = tf.keras.losses.CategoricalCrossentropy()
model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy']    
)

''' Train the model '''
model.fit(x_train, y_train, epochs=800)

''' Evaluate model '''
model.evaluate(x_test, y_test, verbose=2)


# %% Investigating the trained model

import itertools

model_samplings = []
sampling_range = np.arange(
    x_train.min(),
    x_train.max(),
    0.5)

for x1, x2, x3, x4 in itertools.product(
    sampling_range, sampling_range, sampling_range, sampling_range):
    prediction = model(np.array([x1, x2, x3, x4]).reshape([1,4])).numpy()[0]
    prediction = np.argmax(prediction)
    model_samplings.append(
        [x1, x2, x3, x4, prediction]
    )

model_samplings = pd.DataFrame(
    data=model_samplings,
    columns=['x1','x2','x3','x4','prediction'])

_ = sns.pairplot(model_samplings,
                 hue="prediction",
                 height= 2.5,
                 palette={0: 'blue', 1: 'orange', 2: 'green'})

# %%
