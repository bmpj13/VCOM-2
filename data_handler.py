import sys
import pandas as pd
from constants import TRAIN_NODULES_PATH, SCAN_CUBES_PATH, FOLDS_PATH
from sklearn.model_selection import train_test_split
from data_generator import ScanDataGenerator, MultimodalDataGenerator, DescriptorsDataGenerator

def getFoldNodules(path = FOLDS_PATH, nrows = None, fold = 0):
    filename = '{}fold{}_Nodules.csv'.format(path, fold)

    df_train = getTrainNodules(filename, nrows)
    print(df_train)


def getTrainNodules(path = TRAIN_NODULES_PATH, nrows = None):
    df = pd.read_csv(path, nrows=nrows, error_bad_lines=True)

    file_str = lambda lndbid, findid, radid: 'LNDb-{:04}_finding{}_rad{}.npy'.format(int(lndbid), int(findid), int(radid))
    df['filename'] = df.apply(lambda row: file_str(row['LNDbID'], row['FindingID'], row['RadID']), axis=1)
    df['Text'].astype(int)

    # Group texture values as in the specification
    isNonNodule = df['Text'] == 0
    isGGO = (df['Text'] == 1) | (df['Text'] == 2)
    isPartSolid = df['Text'] == 3
    isSolid = (df['Text'] == 4) | (df['Text'] == 5)

    df.loc[isNonNodule, 'Text'] = 0
    df.loc[isGGO, 'Text'] = 1
    df.loc[isPartSolid, 'Text'] = 2
    df.loc[isSolid, 'Text'] = 3

    classes = pd.unique(df['Text'])

    return df, classes

def splitData(df, seed = None, shuffle = False):
    X_train, X_valid, y_train, y_valid = train_test_split(
        df['filename'],
        df['Text'],
        random_state=seed,
        shuffle=shuffle,
        stratify=df['Text']
    )

    return (X_train, y_train), (X_valid, y_valid)

def getDataGenerators(train, valid, classes, method = 'scan_cubes', batch_size = 32):
    (X_train, y_train) = train
    (X_valid, y_valid) = valid

    if method == 'scan_cubes':
        training_generator = ScanDataGenerator(X_train, y_train, batch_size, classes)
        validation_generator = ScanDataGenerator(X_valid, y_valid, batch_size, classes)
    elif method == 'masked_scan_cubes':
        training_generator = ScanDataGenerator(X_train, y_train, batch_size, classes, use_mask=True)
        validation_generator = ScanDataGenerator(X_valid, y_valid, batch_size, classes, use_mask=True)
    elif method == 'multimodal':
        training_generator = MultimodalDataGenerator(X_train, y_train, batch_size, classes)
        validation_generator = MultimodalDataGenerator(X_valid, y_valid, batch_size, classes)
    elif method == 'descriptors':
        training_generator = DescriptorsDataGenerator(X_train, y_train, batch_size, classes)
        validation_generator = DescriptorsDataGenerator(X_valid, y_valid, batch_size, classes)
    else:
        sys.exit('Method not available')

    return training_generator, validation_generator



if __name__ == '__main__':
    df, classes = getTrainNodules()
    print(df)
    print(classes)