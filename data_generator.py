import numpy as np
import pandas as pd
import keras
from constants import SCAN_CUBES_PATH, MASK_CUBES_PATH, DESCRIPTORS_PATH
from abc import ABC

class DataGenerator(ABC, keras.utils.Sequence):
    def convertRGB(self, img):
        # convert to grayscale
        img = img.astype(np.float64)
        img = (img - img.min()) * (255.0 / (img.max() - img.min()))
        img = img.astype(np.uint8)

        # convert to rgb
        img.shape = img.shape + (1,)
        img = np.concatenate((img, img, img), axis=3)

        return img

    def getScanRGB(self, file_path, use_mask = False):
        scan_cube = np.load(SCAN_CUBES_PATH + file_path)

        if use_mask:
            mask_cube = np.load(MASK_CUBES_PATH + file_path)
            scan_cube[mask_cube == 0] = scan_cube.min() # we convert to min because it will turn into 0 in the RGB conversion

        scan_cube = self.convertRGB(scan_cube)

        return scan_cube

    def getMaskRGB(self, file_path):
        mask_cube = np.load(MASK_CUBES_PATH + file_path)
        mask_cube = self.convertRGB(mask_cube)

        return mask_cube

    def getDescriptors(self, file_path):
        descriptors = np.load(DESCRIPTORS_PATH + file_path, allow_pickle=True)

        return descriptors

    def computeOneHotLabels(self, batch_labels, classes):
        batch_y = np.zeros((len(batch_labels), len(classes)), np.int64)
        for i, _ in enumerate(zip(batch_labels, batch_y)):
            label = batch_labels.iloc[i]
            batch_y[i, label] = 1

        return batch_y

    def normalize(self, file_path, path):

        img = np.load(path + file_path)
        
        img = img.astype(np.float64)
        img = (img - img.min()) * (255.0 / (img.max() - img.min()))
        img = img.astype(np.float32)
        img /= 255.
        img.shape =  (1,) + img.shape
        
        return img

class ScanDataGenerator(DataGenerator) :
    def __init__(self, file_paths, labels, batch_size, classes, use_mask = False) :
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.classes = classes
        self.use_mask = use_mask

    def __len__(self) :
        return (np.ceil(len(self.file_paths) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_files = self.file_paths[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x = np.array([ self.getScanRGB(file_path, self.use_mask) for file_path in batch_files ])

        batch_labels = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.computeOneHotLabels(batch_labels, self.classes)

        return batch_x, batch_y

class MultimodalDataGenerator(DataGenerator) :
    def __init__(self, file_paths, labels, batch_size, classes) :
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.classes = classes

    def __len__(self) :
        return (np.ceil(len(self.file_paths) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_files = self.file_paths[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x1 = np.array([ self.getScanRGB(file_path) for file_path in batch_files ])
        batch_x2 = np.array([ self.getMaskRGB(file_path) for file_path in batch_files ])

        batch_labels = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.computeOneHotLabels(batch_labels, self.classes)

        return [batch_x1, batch_x2], batch_y

class DescriptorsDataGenerator(DataGenerator) :
    def __init__(self, file_paths, labels, batch_size, classes) :
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.classes = classes

    def __len__(self) :
        return (np.ceil(len(self.file_paths) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_files = self.file_paths[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_des = np.array([ self.getDescriptors(file_path) for file_path in batch_files ])

        # adding padding to descriptors (that have variable size)
        maxlen = max(len(r) for r in batch_des)
        batch_x = np.zeros((len(batch_files), maxlen, 32))
        for enu, row in enumerate(batch_des):
            batch_x[enu, :len(row), :] += row

        batch_labels = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.computeOneHotLabels(batch_labels, self.classes)
        
        return batch_x, batch_y

class SegmentationDataGenerator(DataGenerator) :
    
    def __init__(self, file_paths, batch_size) :
        self.file_paths = file_paths
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.file_paths) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        
        batch_files = self.file_paths[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x = np.array([ self.normalize(file_path, SCAN_CUBES_PATH) for file_path in batch_files ])
        batch_y = np.array([ self.normalize(file_path, MASK_CUBES_PATH) for file_path in batch_files ])

        return batch_x, batch_y   
             