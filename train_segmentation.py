import os
import numpy as np
import pandas as pd
import argparse
import keras
from keras.models import Model
from models.unet3d import get_unet
from math import ceil
from data_handler import getDataGenerators, getFoldNodules
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import matplotlib.pyplot as plt

BATCH_SIZE = 16
NUM_FOLDS = 4
 
 
def plotTrainingHistory(fold, history):
    
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
   
    plt.savefig("imgs/dice_coeff"+ str(fold) +".png")


def run(epochs):

    # Define callbacks
    model_checkpoint = ModelCheckpoint('weights/3D-Unet.h5', verbose=1, monitor='val_loss', save_best_only=True)
    csv_logger = CSVLogger('results/3D-Unet.txt', separator=',', append=False)

    callbacks_list = [model_checkpoint, csv_logger]

    # Load the 3D-UNet model
    model = get_unet()

    # Train model
    for fold in range(0, NUM_FOLDS):

        train, valid = getFoldNodules(fold=fold, shuffle=True)
        training_generator, validation_generator = getDataGenerators(train, valid, None, method='segmentation', batch_size=BATCH_SIZE)

        # Fit model
        print('Fold', fold)
        history = model.fit_generator(
            generator=training_generator,
            steps_per_epoch=ceil(train[0].size / BATCH_SIZE),
            validation_data=validation_generator,
            validation_steps=ceil(valid[0].size / BATCH_SIZE),
            epochs=epochs,
            verbose=1,
            callbacks=callbacks_list
        )

        print('-'*30)
        # plotTrainingHistory(fold, history)
    
    
parser = argparse.ArgumentParser(description="3D-UNet for the LNDb challenge B")
parser.add_argument('--epochs', help="Number of training epochs", default=20, type=int)
args = parser.parse_args()

run(args.epochs)
