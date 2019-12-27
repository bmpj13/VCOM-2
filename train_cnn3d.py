import os
import numpy as np
import pandas as pd
import argparse
import keras
from keras.callbacks import CSVLogger
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling3D, Dropout
from models.cnn3d import model as CNN3D
from keras.optimizers import SGD
from math import ceil
from constants import TRAIN_NODULES_PATH, SCAN_CUBES_PATH
from data_handler import getTrainNodules, splitData, getDataGenerators, getFoldNodules

BATCH_SIZE = 32
IMAGE_SIZE = 80
DROPOUT_PROB = 0.3
NUM_FOLDS = 4

def run(method, nrows, epochs):
    NAME = '{}_cnn3d'.format(method)
    _, classes = getTrainNodules(TRAIN_NODULES_PATH, nrows=None)
    
    model = CNN3D(input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), dropout_prob=DROPOUT_PROB)
    model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

    print ("Final Model summary")
    model.summary()

    callbacks = [
        CSVLogger('results/' + NAME + '.csv', append=False, separator=';')
    ]

    print()
    model.save_weights('weights/' + NAME + '_initial.h5')
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
            callbacks=callbacks,
            shuffle=True,
            verbose=1
        )
        model.load_weights('weights/' + NAME + '_initial.h5')
        print()

    model.save('weights/' + NAME + '.h5')


parser = argparse.ArgumentParser(description="I3D Neural Network for the LNDb challenge C")
parser.add_argument('--method', help="How data is handled", choices=('scan_cubes', 'masked_scan_cubes'), default='scan_cubes', type=str)
parser.add_argument('--nrows', help="Number of rows loaded", default=None, type=int)
parser.add_argument('--epochs', help="Number of training epochs", default=1, type=int)
args = parser.parse_args()

run(args.method, args.nrows, args.epochs)