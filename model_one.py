import tensorflow as tf
import numpy as np
import pandas as pd


def generate_dataset(file_name):
    raw_input = pd.read_csv(file_name)
    X = raw_input[list(raw_input)[1:]]
    Y = pd.get_dummies(raw_input['label'])
    return train_valid_test(X), train_valid_test(Y)


def train_valid_test(input_data, train_portion=0.95):
    num_sample = input_data.shape[0]
    train_num = int(num_sample * train_portion)
    valid_num = int((num_sample - train_num)/2)
    return input_data[:train_num], input_data[train_num:valid_num], input_data[valid_num:]


(train_X, valid_X, test_X), (train_Y, valid_Y, test_Y) = generate_dataset('train.csv')

