'''
    Author: Massimo Clementi
    Date:   2021-04-12
    
    Very useful links:
    https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
    https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e
    
'''

# %% Imports

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time


# %% Load CIFAR10 dataset

'''
    This dataset contains RGB images of size 32x32 and for each of those
    images provides the ground truth label, between 10 different classes
'''
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


# %% Process train and test pipelines separately

''' Training set '''
x_train =  x_train / 255  # from [0,255] pixel values to [0,1]
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)


''' Test set '''
x_test = x_test / 255
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)



batch_size = 256

''' Convert training dataset into Tensorflow Dataset object '''
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)


''' Do the same with test dataset '''
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size=1024)
test_dataset = test_dataset.batch(batch_size, drop_remainder=True)




# Show shapes
for batch in train_dataset.take(1):
    print('train x batch shape:',batch[0].shape)
    print('train y batch shape:', batch[1].shape)


# Pick a batch and show first 25 images
for (images, labels) in train_dataset.take(1):
    plt.figure()
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(images[i])
        #plt.title(class_names[np.argmax(labels[i])])



# %% Define data augmentation model
def data_augmentation_model():
    '''
        Definition of data augmentation layers that have the task of improving
        generalization capabilities and helping the training to avoid over-fitting.
        However be careful to not be too harsh with the augmentation and consequently
        slow down too much the training and/or underfit the data
        
    '''
    
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(
        0.1,
        input_shape=(32,32,3)
    ))
    model.add(tf.keras.layers.experimental.preprocessing.RandomTranslation((-0.1,0.1),(-0.1,0.1)))
    model.add(tf.keras.layers.experimental.preprocessing.RandomZoom((-0.1,0)))

    return model

''' Show some examples '''
for i in range(8):
  da_model = data_augmentation_model()
  augmented_img = np.squeeze(da_model(np.array([x_train[i]])))
  plt.figure()
  plt.imshow(augmented_img)




# %% Define model

def CNN_Model(show_summary=False):
    
    '''
        Sequential API
            look also for functional and sub-classing for more control over NN
            https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e
        
        Model inspired by this StackExchange thread:
        https://stats.stackexchange.com/questions/272607/cifar-10-cant-get-above-60-accuracy-keras-with-tensorflow-backend
        
    '''
    
    model = tf.keras.models.Sequential()

    model.add(data_augmentation_model())    # custom data augmentation layers
    
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))


    model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=3))
    model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))


    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Softmax())
    
    
    # Print model summary
    if show_summary:
        model.build()
        print(model.summary())
    
    return model

        
model = CNN_Model(show_summary=True)



# %% Training of the model

epochs = 125

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Prepare metrics
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

# Quick arrays to save variables to plot
train_acc = []
test_acc = []
train_loss = []
test_loss = []

''' Iterate over epochs '''
for epoch in range(epochs):
    t = time.time()
    print('Start of epoch',epoch)
    
    ''' Iterate over training batches '''
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        
        ''' Auto differentiation '''
        with tf.GradientTape() as tape:
            
            ''' Evaluate model on training batch'''
            logits = model(x_batch_train, training=True)
            
            ''' Compute train loss '''
            train_loss_value = loss_fn(y_batch_train, logits)
        
        ''' Compute gradients '''
        gradients = tape.gradient(train_loss_value, model.trainable_weights)
        
        ''' Update weights '''
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        ''' Update train metrics '''
        train_acc_metric.update_state(y_batch_train, logits)
        
         # Log every 50 batches.
        if step % 50 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(train_loss_value))
            )
            #print("Seen so far: %s samples" % ((step + 1) * batch_size))
            
    
    ''' Iterate over test batches to see performances '''
    for step, (x_batch_test, y_batch_test) in enumerate(test_dataset):

        ''' Evaluate model on training batch'''
        test_logits = model(x_batch_test, training=False)
            
        ''' Compute test loss '''
        test_loss_value = loss_fn(y_batch_test, test_logits)
        
        ''' Update test metrics '''
        test_acc_metric.update_state(y_batch_test, test_logits)
        
    
    print('time:', time.time()-t)
    print('loss:', train_loss_value.numpy())
    print('acc:', train_acc_metric.result().numpy())
    print('test loss:', test_loss_value.numpy())
    print('test acc:', test_acc_metric.result().numpy())
    
    train_acc.append(train_acc_metric.result().numpy())
    test_acc.append(test_acc_metric.result().numpy())
    train_loss.append(train_loss_value.numpy())
    test_loss.append(test_loss_value.numpy())
    

    ''' Save weights every 20 epochs '''
    if epoch % 25 == 0:
      file_str = '../saved_models/3_CIFAR_and_tf_data_module/epoch_{e}_weights'.format(
          e=str(epoch).zfill(3)
      )
      model.save_weights(file_str)
      print('Checkpoint saved as',file_str)

    print('---')


    ''' Reset metrics at the end of each epoch '''
    train_acc_metric.reset_states()
    test_acc_metric.reset_states()

file_str = '../saved_models/3_CIFAR_and_tf_data_module/epoch_FINAL_weights'
model.save_weights(file_str)
print('Final weights saved as',file_str)
print('END of training!')



# Show training stats
fig = plt.figure(figsize=(7,7))
ax1 = plt.subplot(2, 1, 1)
sns.lineplot(
    x=range(len(train_acc)),
    y=train_acc,
    color='red',
    label='accuracy over training set',
    ax = ax1
)
sns.lineplot(
    x=range(len(test_acc)),
    y=test_acc,
    color='blue',
    label='accuracy over test set',
    ax=ax1
)

ax2 = plt.subplot(2, 1, 2)
sns.lineplot(
    x=range(len(train_loss)),
    y=train_loss,
    color='red',
    label='loss over training set',
    ax=ax2
)
sns.lineplot(
    x=range(len(test_loss)),
    y=test_loss,
    color='blue',
    label='loss over test set',
    ax=ax2
)


# %% Test pretrained model

# Create model and load weights
model = CNN_Model()
#!pwd   #<- check working folder if problems in loading weights
print('Loading weights...')
model.load_weights('../saved_models/3_CIFAR_and_tf_data_module/epoch_FINAL_weights')

loss_fn = tf.keras.losses.CategoricalCrossentropy()

pretrained_test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

print('Testing model...')
for step, (x_batch_test, y_batch_test) in enumerate(test_dataset):

    ''' Evaluate model on training batch'''
    test_logits = model(x_batch_test, training=False)
            
    ''' Compute test loss '''
    test_loss_value = loss_fn(y_batch_test, test_logits)
        
    ''' Update test metrics '''
    pretrained_test_acc_metric.update_state(y_batch_test, test_logits)
    
print('Accuracy over test set:', pretrained_test_acc_metric.result().numpy())


# %%
