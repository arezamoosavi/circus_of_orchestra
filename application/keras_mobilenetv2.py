from __future__ import print_function, division
from builtins import range, input

# Note: you may need to update your version of future
# sudo pip install -U future

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model

from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

IMAGE_SIZE = [160, 160]
BATCH_SIZE = 16
EPOCHS = 32

train_path = "~/train"
valid_path = "~/val"

image_files = glob(train_path + "/*/*.jp*g")
valid_image_files = glob(valid_path + "/*/*.jp*g")

folders = glob(train_path + "/*")

gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input,
)

train_dataset = gen.flow_from_directory(
    batch_size=BATCH_SIZE, directory=train_path, target_size=IMAGE_SIZE
)

validation_dataset = gen.flow_from_directory(
    batch_size=BATCH_SIZE, directory=valid_path, target_size=IMAGE_SIZE
)

mnetv2 = MobileNetV2(
    input_shape=IMAGE_SIZE + [3], include_top=False, weights="imagenet",
)

# don't train existing weights
for layer in mnetv2.layers:
    layer.trainable = False

# our layers - you can add more if you want
x = Flatten()(mnetv2.output)

# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation="softmax")(x)

# create a model object
model = Model(inputs=mnetv2.input, outputs=prediction)

# tell the model what cost and optimization method to use
model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

r = model.fit_generator(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    steps_per_epoch=len(image_files) // BATCH_SIZE,
    validation_steps=len(valid_image_files) // BATCH_SIZE,
)

model.save("mobilenet_v2.h5")
