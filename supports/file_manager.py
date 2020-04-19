"""
support functions
"""

import pandas as pd
import numpy as np
import os

def get_datasets_list():
    return os.listdir('datasets')

def get_csv_list(data_dir):
    return os.listdir('datasets/'+data_dir)

def read_data(data_dir, csv_file):
    path = 'datasets/' + data_dir + '/' + csv_file
    df = pd.read_csv(path, sep=",")
    df = df.reindex(
        np.random.permutation(df.index))
    print("Overview:")
    print(df.shape)
    print(df.head(),'\n')
    print('Basic statistical values:\n', df.describe())
    return df

def write_data(dataframe, file_name):
    dataframe.to_csv("output/"+file_name+".csv", index = False)

if __name__ == "__main__":
    # example
    print(get_datasets_list())
    print(get_csv_list('data-gempa'))

    read_data('data-gempa', '2009.csv')