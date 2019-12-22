import os
import numpy as np
import pandas as pd
import argparse
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling3D, Dropout
from models.i3d import Inception_Inflated3d as I3D
from math import ceil
from constants import TRAIN_NODULES_PATH, SCAN_CUBES_PATH
from data_handler import getTrainNodules, splitData, getDataGenerators, getFoldNodules

BATCH_SIZE = 32
IMAGE_SIZE = 80
DROPOUT_PROB = 0.2
NUM_FOLDS = 4

def run(method, nrows, epochs):
    # df, classes = getTrainNodules(TRAIN_NODULES_PATH, nrows = nrows)
    # train, valid = splitData(df, shuffle=True)
    # training_generator, validation_generator = getDataGenerators(train, valid, classes, method=method, batch_size=BATCH_SIZE)

    _, classes = getTrainNodules(TRAIN_NODULES_PATH, nrows=None)

    # Load the I3D model
    base_model = I3D(weights='rgb_imagenet_and_kinetics', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), dropout_prob=DROPOUT_PROB)

    for layer in base_model.layers:
        layer.trainable = False

    print ("Base Model summary")
    print(base_model.summary())

    # Add classification layers
    x = base_model.output
    x = GlobalAveragePooling3D()(x)
    x = Dropout(DROPOUT_PROB)(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print ("Final Model summary")
    model.summary()

    print()
    print()
    for fold in range(0, NUM_FOLDS):
        train, valid = getFoldNodules(nrows=nrows, fold=fold, shuffle=True)
        training_generator, validation_generator = getDataGenerators(train, valid, classes, method=method, batch_size=BATCH_SIZE)

        # Fit model
        print('Fold', fold)
        model.fit_generator(
            generator=training_generator,
            steps_per_epoch=ceil(train[0].size / BATCH_SIZE),

            validation_data=validation_generator,
            validation_steps=ceil(valid[0].size / BATCH_SIZE),

            epochs=epochs,
            verbose=1
        )
        print()

    model.save('weights/{}_i3d.h5'.format(method))


parser = argparse.ArgumentParser(description="I3D Neural Network for the LNDb challenge C")
parser.add_argument('--method', help="How data is handled", choices=('scan_cubes', 'masked_scan_cubes'), default='scan_cubes', type=str)
parser.add_argument('--nrows', help="Number of rows loaded", default=None, type=int)
parser.add_argument('--epochs', help="Number of training epochs", default=1, type=int)
args = parser.parse_args()

run(args.method, args.nrows, args.epochs)