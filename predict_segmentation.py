import os
import numpy as np
import pandas as pd
import argparse
import keras
from models.unet3d import get_unet
import matplotlib.pyplot as plt
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras import backend as K

# file to predict
file = 'LNDb-0001_finding1_rad3.npy'



def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))

def weighted_dice_coefficient_loss(y_true, y_pred):
    return 1 - weighted_dice_coefficient(y_true, y_pred)




MASK_CUBES_PATH = 'lndb/mask_cubes/'
SCAN_CUBES_PATH = 'lndb/scan_cubes/' 
WEIGHTS_PATH = 'weights/3D-Unet.h5'
 
def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error



# get model
 
model = load_old_model('model.h5')
 
# original 
org = np.load(SCAN_CUBES_PATH+file)

# convert to grayscale
org = org.astype(np.float64)
org = (org - org.min()) * (255.0 / (org.max() - org.min()))  
org = org.astype(np.float32)
org /= 255.

# input
img = org.copy()
img.shape =  (1,) + img.shape 
img = img[None, :]

img_mask_test = model.predict(img, batch_size=1, verbose=1)
img_mask_test = np.squeeze(img_mask_test)
img_mask_test = img_mask_test.astype(np.float64)
    

 

# original mask
img_mask = np.load(MASK_CUBES_PATH+file)
    
 
# plot 
fig, axs = plt.subplots(3,3)
axs[0,0].imshow(org[int(org.shape[0]/2),:,:])
axs[0,1].imshow(img_mask[int(img_mask.shape[0]/2),:,:])
axs[0,2].imshow(img_mask_test[int(img_mask_test.shape[2]/2),:,:])
 
axs[1,0].imshow(org[:,int(org.shape[0]/2),:])
axs[1,1].imshow(img_mask[:,int(img_mask.shape[0]/2),:])
axs[1,2].imshow(img_mask_test[:,int(img_mask_test.shape[0]/2),:])

axs[2,0].imshow(org[:,:,int(org.shape[0]/2)])
axs[2,1].imshow(img_mask[:,:,int(img_mask.shape[0]/2)])
axs[2,2].imshow(img_mask_test[:,:,int(img_mask_test.shape[0]/2)])
 

plt.show()
 


