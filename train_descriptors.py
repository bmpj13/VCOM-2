import numpy as np
import pandas as pd
import argparse
import keras
from keras.callbacks import CSVLogger
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling1D
from data_handler import getTrainNodules, splitData, getDataGenerators, getFoldNodules
from constants import TRAIN_NODULES_PATH
from math import ceil

BATCH_SIZE = 32
IMAGE_SIZE = 80
DROPOUT_PROB = 0.2
NUM_FOLDS = 4
INPUT_SHAPE = (None, 32)

def run(method, nrows, epochs):
    _, classes = getTrainNodules(TRAIN_NODULES_PATH, nrows = nrows)

    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=INPUT_SHAPE))
    model.add(Dropout(DROPOUT_PROB))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(DROPOUT_PROB))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(len(classes), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    callbacks = [
        # keras.callbacks.EarlyStopping(
        #     # Stop training when `val_loss` is no longer improving
        #     monitor='val_loss',
        #     # "no longer improving" being further defined as "for at least 20 epochs"
        #     patience=20,
        #     verbose=1),
        CSVLogger('results/descriptors.csv', append=False, separator=';')
    ]

    print()
    model.save_weights('weights/descriptors_initial.h5')
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
            verbose=1,
        )
        model.load_weights('weights/descriptors_initial.h5')
        print()

    model.save('weights/descriptors.h5')


parser = argparse.ArgumentParser(description="I3D Neural Network for the LNDb challenge C")
parser.add_argument('--method', help="How data is handled", choices=('descriptors'), default='descriptors', type=str)
parser.add_argument('--nrows', help="Number of rows loaded", default=None, type=int)
parser.add_argument('--epochs', help="Number of training epochs", default=1, type=int)
args = parser.parse_args()

run(args.method, args.nrows, args.epochs)