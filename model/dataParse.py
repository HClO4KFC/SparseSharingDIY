import pandas as pd
import numpy as np

from errReport import CustomError

def get_dataset(dataset, ratio = None):
    print('loading', dataset)

    if dataset == 'mimic27':
        x = np.array(pd.read_csv('../gain_collection/collected_gain/mimic27/mimic_x.csv', header=None, sep=' '))
        y = np.array(pd.read_csv('../gain_collection/collected_gain/mimic27/mimic_y_valid.csv', header=None, sep=' '))
        testx = np.array(pd.read_csv('../gain_collection/collected_gain/mimic27/mimic_x.csv', header=None, sep=' '))
        testy = np.array(
            pd.read_csv('../gain_collection/collected_gain/mimic27/mimic_y_test.csv', header=None, sep=' '))
    elif dataset == 'taskonomy':
        x = np.array(pd.read_csv('../gain_collection/collected_gain/taskonomy/taskonomy_x.csv', header=None))
        y = np.array(pd.read_csv('../gain_collection/collected_gain/taskonomy/taskonomy.csv', sep=' ', header=None))
        testx = np.array(pd.read_csv('../gain_collection/collected_gain/taskonomy/taskonomy_x.csv', header=None))
        testy = np.array(pd.read_csv('../gain_collection/collected_gain/taskonomy/taskonomy.csv', sep=' ', header=None))
    else :
        raise CustomError('Cannot find dataset:', dataset)
    return x, y, testx, testy