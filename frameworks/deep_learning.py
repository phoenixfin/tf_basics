# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from supports.grapher import plot_the_loss_curve

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

class SequentialLayer(object):
    def __init__(self, df):
        self.model = None
        self.train_df, self.test_df = train_test_split(df, test_size=0.2)        
        self.features = []
        self.target = None

    def build(self, features, target_var, learning_rate):
        self.features = self.train_df[features]
        self.target = self.train_df[target_var]
        
        self.model = tf.keras.models.Sequential()

        # Add one linear layer to the model to yield a simple linear regressor.
        self.model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

        # Compile the model topography into code that TensorFlow can efficiently
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                    loss="mean_squared_error",
                    metrics=[tf.keras.metrics.RootMeanSquaredError()])

    def train(self, epochs, batch_size=None, validation_split=0.1):
        """Feed a dataset into the model in order to train it."""

        history = self.model.fit(
            x = self.features,
            y = self.target,
            batch_size = batch_size,
            epochs = epochs,
            validation_split = validation_split
        )

        # The list of epochs is stored separately from the 
        # rest of history.
        epochs = history.epoch

        # Isolate the root mean squared error for each epoch.
        hist = pd.DataFrame(history.history)
        rmse = hist["root_mean_squared_error"]

        return epochs, rmse, history.history   

    def get_model_parameter(self, feat):
        feat = feat.replace(' ', '_')
        pos = 'linear/linear_model/'
        weight = self.model.get_variable_value(pos + feat + '/weights')[0]
        bias = self.model.get_variable_value(pos + 'bias_weights')
        return weight, bias
    
    def run_model(self, batch_size, epochs_num, validation_split):
        epochs, rmse, history = self.train(
            epochs_num, batch_size, validation_split
        )

        plot_the_loss_curve(epochs, history["root_mean_squared_error"], 
                            history["val_root_mean_squared_error"])
