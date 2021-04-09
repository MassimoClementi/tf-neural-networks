#!venv/bin/python3

'''
    Author: Massimo Clementi
    Date:   2021-04-03
    
    Basics of Tensorflow and MLP bare training on a generated dataset

'''

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %% Generate and show dataset

def add_MVN(dataset, mean, cov, n_samples, label):
    samples = np.random.multivariate_normal(
        mean=mean,
        cov=cov,
        size=n_samples
    )

    ''' Add samples to training set '''
    for i, sample in enumerate(samples):
        dataset.append(
            np.concatenate((sample, np.array([label])), axis=0)
        )



dataset = []

''' Generate class w1 '''
add_MVN(dataset,
        mean=[0.25,0.3],
        cov=np.array([[0.01, 0], [0,0.02]]),
        n_samples=150,
        label=0
)
add_MVN(dataset,
        mean=[0.45,0.7],
        cov=np.array([[0.007, 0], [0,0.01]]),
        n_samples=150,
        label=0
)
add_MVN(dataset,
        mean=[0.7,0.2],
        cov=np.array([[0.015, 0], [0,0.007]]),
        n_samples=150,
        label=0
)


''' Generate class w2 '''
add_MVN(dataset,
        mean=[0.75,0.6],
        cov=np.array([[0.01, -0.005], [-0.005,0.02]]),
        n_samples=150,
        label=1
)
add_MVN(dataset,
        mean=[0.15,0.6],
        cov=np.array([[0.005, 0.001], [0.001,0.007]]),
        n_samples=150,
        label=1
)
add_MVN(dataset,
        mean=[0.3,0.15],
        cov=np.array([[0.004, 0.003], [0.003,0.006]]),
        n_samples=75,
        label=1
)

''' Convert dataset into DataFrame '''
dataset = pd.DataFrame(
    data=dataset,
    columns=['x1', 'x2', 'label']
)

''' Show dataset '''
ax = plt.subplot()
sns.scatterplot(
    x='x1',
    y='x2',
    hue='label',
    palette=sns.color_palette('Set2',2),
    data=dataset)
_ = ax.set_xlim([0,1])
_ = ax.set_ylim([0,1])


# %% Train a simple MLP to separate the data

import tensorflow as tf

''' Create shuffled train and test sets '''
n_samples, dim_samples = dataset.shape
dataset_shuffled = dataset.sample(frac=1)
x_train = dataset_shuffled.head(int(n_samples * 2/3))[['x1','x2']].to_numpy()
y_train = dataset_shuffled.head(int(n_samples * 2/3))['label'].to_numpy()
x_test = dataset_shuffled.tail(int(n_samples * 1/3))[['x1','x2']].to_numpy()
y_test = dataset_shuffled.tail(int(n_samples * 1/3))['label'].to_numpy()


''' Define model '''
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1,2)),
    tf.keras.layers.Dense(31, activation= 'tanh'),
    tf.keras.layers.Dense(11, activation= 'tanh'),
    tf.keras.layers.Dense(1, activation= 'tanh')
])


#model(x_train[-1].reshape([1,2])).numpy()


''' Define loss function '''
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

#loss_fn(
#    y_train[-1],
#    model(x_train[-1].reshape([1,2]))
#)

model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy']
)

''' Train the model '''
model.fit(x_train, y_train, epochs=800)


# %% Evaluate and display the trained model

''' Evaluate the model '''
model.evaluate(x_test, y_test, verbose=2)

model_samplings = []
sampling_range = np.arange(0, 1, 0.01)
for y in np.flip(sampling_range):
    for x in sampling_range:
        prediction = model(np.array([x, y]).reshape([1,2])).numpy()[0,0]
        model_samplings.append(
            prediction
        )


ax = plt.subplot()
_ = sns.heatmap(
    data=np.array(model_samplings).reshape(
        [len(sampling_range), len(sampling_range)]
        ),
    cmap='vlag',
    vmin=0,
    vmax=1
)





# %%
