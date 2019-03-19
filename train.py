# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:57:58 2019

@author: quantum
"""

from keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import os

import tensorflow as tf
#defining the model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=5, padding="same",  input_shape=(100, 100, 1), activation="relu"))
   # model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(64, kernel_size=5, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(128, kernel_size=5, padding="same", activation="relu"))
    model.add(Conv2D(128, kernel_size=5, padding="same", activation="relu"))
    model.add(Conv2D(128, kernel_size=5, padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=1024, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(units=10, activation="softmax"))
    return model



train_dir =  "C://Users//quantum//Desktop//ml//Sign-Language-Recognizer//Sign-Language-Recognizer-master//Dataset"

val_dir =  "C://Users//quantum//Desktop//ml//Sign-Language-Recognizer//Sign-Language-Recognizer-master//e//val"

from keras.preprocessing.image import ImageDataGenerator

batch_size = 16

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, fill_mode='reflect', width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(100,100),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical'
        )

validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(100,100),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical'
        )



model = define_model()
optimizer = tf.keras.optimizers.Adam(lr=0.00001)
model.compile(optimizer= optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
#lr is optional we can avoid it actually
history = model.fit_generator(train_generator, 
                              steps_per_epoch=
                              batch_size, epochs=400, verbose=1,
                              validation_data=validation_generator,
                              validation_steps=  batch_size)


#import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


model.summary()
                      
model.save("modelh.h5")
