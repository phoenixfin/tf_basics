# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf

from supports.graph_manager import plot_the_loss_curve
from frameworks._general_model import GeneralModel

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

class SequentialLayer(GeneralModel):
    def __init__(self, df):
        super().__init__(df)

    def build(self, features, target_var, learning_rate):
        super()._build(features, target_var)
                
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
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

        epochs = history.epoch

        hist = pd.DataFrame(history.history)
        rmse = hist["root_mean_squared_error"]

        return epochs, rmse, history.history   

    def get_model_parameter(self, feat):
        feat = feat.replace(' ', '_')
        pos = 'linear/linear_model/'
        weight = self.model.get_variable_value(pos + feat + '/weights')[0]
        bias = self.model.get_variable_value(pos + 'bias_weights')
        return weight, bias
    
    def run_model(self, validation_split, batch_size, epochs):
        epochs, rmse, history = self.train(
            epochs, batch_size, validation_split
        )

        plot_the_loss_curve(epochs, history["root_mean_squared_error"], 
                            history["val_root_mean_squared_error"])
        return history["root_mean_squared_error"]