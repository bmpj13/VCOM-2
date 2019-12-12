import pandas as pd
from constants import TRAIN_NODULES_PATH
from sklearn.model_selection import train_test_split
from data_generator import DataGenerator

def get_train_nodules(path = TRAIN_NODULES_PATH, nrows = None):
    df = pd.read_csv(path, nrows=nrows, error_bad_lines=True)
    
    file_str = lambda lndbid, findid, radid: 'LNDb-{:04}_finding{}_rad{}.npy'.format(int(lndbid), int(findid), int(radid))
    df['filename'] = df.apply(lambda row: file_str(row['LNDbID'], row['FindingID'], row['RadID']), axis=1)
    
    classes = sorted(pd.unique(df['Text']))

    return df, classes

def get_train_nodules_split(path = TRAIN_NODULES_PATH, nrows = None, seed = None, shuffle = False):
    df, classes = get_train_nodules(path, nrows)

    X_train, X_valid, y_train, y_valid = train_test_split(
        df['filename'],
        df['Text'],
        random_state=seed,
        shuffle=shuffle,
        stratify=df['Text']
    )

    return (X_train, y_train), (X_valid, y_valid), df, classes

def get_train_nodules_generators(path = TRAIN_NODULES_PATH, nrows = None, seed = None, shuffle = False, batch_size = 32, path_prefix = ''):
    (X_train, y_train), (X_valid, y_valid), df, classes = get_train_nodules_split(path, nrows, seed, shuffle)

    training_generator = DataGenerator(path_prefix + X_train, y_train, batch_size, classes)
    validation_generator = DataGenerator(path_prefix + X_valid, y_valid, batch_size, classes)

    return training_generator, validation_generator, (X_train, y_train), (X_valid, y_valid), df, classes