import numpy as np
import pandas as pd
import keras

class DataGenerator(keras.utils.Sequence) :
  
    def __init__(self, file_paths, labels, batch_size, classes) :
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.classes = classes


    def __len__(self) :
        return (np.ceil(len(self.file_paths) / float(self.batch_size))).astype(np.int)


    def __getitem__(self, idx) :
        batch_files = self.file_paths[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x = np.array([ self.getScanRGB(file_path) for file_path in batch_files ])

        batch_labels = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = np.zeros((len(batch_labels), len(self.classes)), np.int64)
        for i, _ in enumerate(zip(batch_labels, batch_y)):
            label = batch_labels.iloc[i]
            batch_y[i, label] = 1

        return batch_x, batch_y

    def getScanRGB(self, file_path):
        scan_cube = np.load(file_path)

        # convert to grayscale
        scan_cube = scan_cube.astype(np.float64)
        scan_cube = (scan_cube - scan_cube.min()) * (255.0 / (scan_cube.max() - scan_cube.min()))
        scan_cube = scan_cube.astype(np.uint8)

        # convert to rgb
        scan_cube.shape = scan_cube.shape + (1,)
        scan_cube = np.concatenate((scan_cube, scan_cube, scan_cube), axis=3)

        return scan_cube

