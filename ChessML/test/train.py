import glob
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def main():
    path_to_csv = "C:\\Users\\vinhp\\Desktop\\ChessML\\archive\\data.csv"

    li = []

    for filename in files_fischer:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    train = pd.concat(li, axis=0, ignore_index=True)
    print(train)
    train = shuffle(train)

    print(train.shape)
    print(train.head())

    features = list(train.iloc[:, 0:192].columns)
    X = train[features]
    y = train["good_move"]

    categorical_columns = list(X.iloc[:, 0:63].columns)
    numerical_columns = list(X.iloc[:, 64:192].columns)
    feature_columns = []

    for feature_name in categorical_columns:
        vocabulary = X[feature_name].unique()
        feature_columns.append(
            tf.feature_column.categorical_column_with_vocabulary_list(
                feature_name, vocabulary
            )
        )

    for feature_name in numerical_columns:
        feature_columns.append(
            tf.feature_column.numeric_column(feature_name, dtype=tf.float32)
        )

    batches_X, batches_y = split_into_batches(train)


def split_into_batches(df, batch_size=100000):
    nb_rows = len(df.index)
    intervals = []

    for i in range(0, nb_rows + 1, batch_size):
        intervals.append(i)

    if intervals[-1] != nb_rows:
        intervals.append(nb_rows)

    batches_X = []
    batches_y = []

    for i in range(0, len(intervals) - 1):
        batches_X.append(train.iloc[intervals[i] : intervals[i + 1], :][features])
        batches_y.append(train.iloc[intervals[i] : intervals[i + 1], :]["good_move"])

    return batches_X, batches_y


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds

    return input_function


def split_into_batches(df, batch_size=100000):
    nb_rows = len(df.index)
    intervals = []

    for i in range(0, nb_rows + 1, batch_size):
        intervals.append(i)

    if intervals[-1] != nb_rows:
        intervals.append(nb_rows)

    batches_X = []
    batches_y = []

    for i in range(0, len(intervals) - 1):
        batches_X.append(train.iloc[intervals[i] : intervals[i + 1], :][features])
        batches_y.append(train.iloc[intervals[i] : intervals[i + 1], :]["good_move"])

    return batches_X, batches_y
