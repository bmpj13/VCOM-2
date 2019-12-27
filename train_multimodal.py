import numpy as np
import pandas as pd
import os
import argparse
import keras
from keras.callbacks import CSVLogger
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling3D, Dropout, concatenate
from models.i3d import Inception_Inflated3d as I3D
from math import ceil
from constants import TRAIN_NODULES_PATH, SCAN_CUBES_PATH
from data_handler import getTrainNodules, splitData, getDataGenerators, getFoldNodules

BATCH_SIZE = 32
IMAGE_SIZE = 80
DROPOUT_PROB = 0.2
NUM_FOLDS = 4

def run(method, nrows, epochs):
    _, classes = getTrainNodules(TRAIN_NODULES_PATH, nrows = nrows)

    # Load the I3D model (for scan modality)
    base_model1 = I3D(weights='rgb_imagenet_and_kinetics', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), dropout_prob=DROPOUT_PROB)

    for layer in base_model1.layers:
        layer.name = 'scan_' + layer.name
        layer.trainable = False

    x1 = base_model1.output
    x1 = GlobalAveragePooling3D()(x1)
    x1 = Dropout(DROPOUT_PROB)(x1)

    # Load the I3D model (for mask modality)
    base_model2 = I3D(weights='rgb_imagenet_and_kinetics', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), dropout_prob=DROPOUT_PROB)

    for layer in base_model2.layers:
        layer.name = 'mask_' + layer.name
        # layer.trainable = False

    x2 = base_model2.output
    x2 = GlobalAveragePooling3D()(x2)
    x2 = Dropout(DROPOUT_PROB)(x2)

    # Merge subnetworks
    x = concatenate([x1, x2])
    x = Dense(2048, activation='relu')(x)
    predictions = Dense(len(classes), activation='softmax')(x)

    model = Model(inputs=[base_model1.input, base_model2.input], outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    callbacks = [
        # keras.callbacks.EarlyStopping(
        #     # Stop training when `val_loss` is no longer improving
        #     monitor='val_loss',
        #     # "no longer improving" being further defined as "for at least 20 epochs"
        #     patience=20,
        #     verbose=1),
        CSVLogger('results/multimodal.csv', append=True, separator=';')
    ]

    print()
    model.save_weights('weights/multimodal_initial.h5')
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
        model.load_weights('weights/multimodal_initial.h5')
        print()

    model.save('weights/multimodal.h5')


parser = argparse.ArgumentParser(description="I3D Neural Network for the LNDb challenge C")
parser.add_argument('--method', help="How data is handled", choices=('multimodal'), default='multimodal', type=str)
parser.add_argument('--nrows', help="Number of rows loaded", default=None, type=int)
parser.add_argument('--epochs', help="Number of training epochs", default=1, type=int)
args = parser.parse_args()

run(args.method, args.nrows, args.epochs)