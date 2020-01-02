import os
import numpy as np
import pandas as pd
import argparse
import keras
from models.unet3d import get_unet
import matplotlib.pyplot as plt
 

# file to predict
file = 'LNDb-0001_finding1_rad3.npy'


MASK_CUBES_PATH = 'lndb/mask_cubes/'
SCAN_CUBES_PATH = 'lndb/scan_cubes/' 
WEIGHTS_PATH = 'weights/3D-Unet.h5'
 

# get model
model = get_unet()
model.load_weights(WEIGHTS_PATH)
 
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
    

# output
img_mask_test[img_mask_test>=0.5] = 1
img_mask_test[img_mask_test<0.5] = 0
 

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
 


