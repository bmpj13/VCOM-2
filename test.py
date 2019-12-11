import numpy as np
import pandas as pd
import os
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling3D
from sklearn.model_selection import train_test_split
from models.i3d import Inception_Inflated3d as I3D
from data_generator import DataGenerator
from math import ceil
from constants import TRAIN_NODULES_PATH, SCAN_CUBES_PATH

BATCH_SIZE = 32
IMAGE_SIZE = 80

seed = 1

# Read CSV file
df = pd.read_csv(TRAIN_NODULES_PATH, nrows=100, error_bad_lines=True)
df['filename'] = df.apply(lambda row: 'LNDb-{:04}_finding{}_rad{}.npy'.format(int(row['LNDbID']), int(row['FindingID']), int(row['RadID'])), axis=1)
df = df.sample(frac=1).reset_index(drop=True)
classes = sorted(pd.unique(df['Text']))

X_train, X_valid, y_train, y_valid = train_test_split(df['filename'], df['Text'], random_state=seed, shuffle=True, stratify=df['Text'])

training_generator = DataGenerator(SCAN_CUBES_PATH + X_train, y_train, BATCH_SIZE, classes)
validation_generator = DataGenerator(SCAN_CUBES_PATH + X_train, y_train, BATCH_SIZE, classes)

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