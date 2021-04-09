#!venv/bin/python3

'''
    Author: Massimo Clementi
    Date:   2021-04-07
    
    Preprocess properly the MNIST dataset to have correct shape and one-hot labels
    Define LeNet neural network with Conv2D and MaxPool2D layers
    Show architecture, define regularization callbacks and save weights
    Collect and show training history
    Show some examples of test set images and their predictions

'''


# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import urllib.request
from datetime import datetime

# %% Prepare MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

''' Add single channel dimension for complying with Conv2D '''
x_train = x_train.reshape(
    x_train.shape[0],
    x_train.shape[1],
    x_train.shape[2],
    1
)
x_test = x_test.reshape(
    x_test.shape[0],
    x_test.shape[1],
    x_test.shape[2],
    1
)

''' Convert labels from integer to one-hot vector '''
y_train = tf.one_hot(
    indices=y_train,
    depth=10
)
y_test = tf.one_hot(
    indices=y_test,
    depth=10
)

# %% Define NN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        filters=20,
        kernel_size=5,
        activation='relu',
        data_format='channels_last',    #batch_size + (img_size, img_size, channels)
        input_shape=(28,28,1),
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=(2,2)
    ),
    tf.keras.layers.Conv2D(
        filters=50,
        kernel_size=5,
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=(2,2)
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        units=500,
        activation='tanh'
    ),
    tf.keras.layers.Dense(
        units=10,
        activation='tanh'
    ),
    tf.keras.layers.Softmax()
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
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200)
  ]

''' Train the model '''
print("Training started at", datetime.now())
history = model.fit(x_train,
          y_train,
          epochs=15,
          callbacks=get_callbacks()
)
print("Training ended at", datetime.now())

model.save_weights(   #save trained weights
    filepath='../saved_models/2_MNIST_dataset_and_CNNs/CNN',
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

# %% See results over a random subset of test images

''' Restore weights if necessary '''
model.load_weights('../saved_models/2_MNIST_dataset_and_CNNs/CNN')

test_idxs = np.random.randint(low=0, high=x_test.shape[0]-1, size=5)

for idx in test_idxs:
    x_sample = np.array([x_test[idx]])  # create single dimension for batch
    pred = model(x_sample).numpy()
    num_hat = np.argmax(pred[0])           # prediction of what number
    confidence = pred[0,num_hat]          # confidence of the prediction
    plt.imshow(x_test[idx],cmap='gray')
    plt.title(str(num_hat)+' with confidence '+str(confidence))
    #print(pred[0])
    plt.show()

# %%
