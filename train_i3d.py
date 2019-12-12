import numpy as np
import pandas as pd
import os
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling3D
from models.i3d import Inception_Inflated3d as I3D
from math import ceil
from constants import TRAIN_NODULES_PATH, SCAN_CUBES_PATH
from data_handler import get_train_nodules_generators

BATCH_SIZE = 32
IMAGE_SIZE = 80
seed = 1

training_generator, validation_generator, _, _, df, classes = get_train_nodules_generators(nrows=100, seed=seed, shuffle=True, batch_size=BATCH_SIZE, path_prefix=SCAN_CUBES_PATH)

# Load the I3D model
base_model = I3D(weights='rgb_imagenet_and_kinetics', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

for layer in base_model.layers:
    layer.trainable = False

print ("Base Model summary")
print(base_model.summary())

# Add classification layers
x = base_model.output
x = GlobalAveragePooling3D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print ("Final Model summary")
model.summary()


# Fit model
model.fit_generator(
    generator=training_generator,
    steps_per_epoch=ceil(0.75 * (df.size / BATCH_SIZE)),

    validation_data=validation_generator,
    validation_steps=ceil(0.25 * (df.size / BATCH_SIZE)),

    epochs=1,
    verbose=1
)

model.save('weights/i3d.h5')