import numpy as np
import os
import SimpleITK as sitk
from constants import MASK_CUBES_PATH, SCAN_CUBES_PATH
from lndb.scripts.utils import readCsv

# for filename in os.listdir(MASK_CUBES_PATH):
#     print(filename)


#Extract and display cube for example nodule
lnd = 1
finding = 1
rad = 1

# Read scan
scan_cube = np.load(SCAN_CUBES_PATH + 'LNDb-{:04}_finding{}_rad{}.npy'.format(lnd,finding,rad))
print('Scan Cube')
print(scan_cube)
print(scan_cube.shape)
print()

# Read segmentation mask
mask_cube = np.load(MASK_CUBES_PATH + 'LNDb-{:04}_finding{}_rad{}.npy'.format(lnd,finding,rad))
print('Mask Cube')
print(mask_cube)
print(mask_cube.shape)
print()


# Remove nodule's surrounding by its mask
# scan_cube[mask_cube == 0] = scan_cube.min()

# Transform scan_cube
scan_cube = scan_cube.astype(np.float64)
scan_cube = (scan_cube - scan_cube.min()) * (255.0 / (scan_cube.max() - scan_cube.min()))
scan_cube = scan_cube.astype(np.uint8)
print(scan_cube)
print(scan_cube.shape)

scan_cube.shape = scan_cube.shape + (1,)
scan_cube = np.concatenate((scan_cube, scan_cube, scan_cube), axis=3)
print(scan_cube)
print(scan_cube.shape)


# Display nodule scan/mask slice
from matplotlib import pyplot as plt
fig, axs = plt.subplots(2,3)
axs[0,0].imshow(scan_cube[int(scan_cube.shape[0]/2),:,:])
axs[1,0].imshow(mask_cube[int(mask_cube.shape[0]/2),:,:])
axs[0,1].imshow(scan_cube[:,int(scan_cube.shape[1]/2),:])
axs[1,1].imshow(mask_cube[:,int(mask_cube.shape[1]/2),:])
axs[0,2].imshow(scan_cube[:,:,int(scan_cube.shape[2]/2)])
axs[1,2].imshow(mask_cube[:,:,int(mask_cube.shape[2]/2)])    
plt.show()
