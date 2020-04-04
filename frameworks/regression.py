import math
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics

from frameworks._general_model import GeneralModel

tf.keras.backend.set_floatx('float64')
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

class LinearRegressor(GeneralModel):
    def __init__(self, df):
        super().__init__(df)
        self.train_df = df
    
    def build(self, features, target_var, learning_rate):
        super()._build(features, target_var)
        optimizer = tf.keras.optimizers.SGD(
            lr=learning_rate, 
            clipvalue = 5.0
        )
        
        feature_columns = []
        for f in features:
            feature_columns.append(tf.feature_column.numeric_column(f))
            
        self.model = tf.estimator.LinearRegressor(
            feature_columns = feature_columns,
            optimizer = optimizer
        )

    def input_fn(self, batch_size=1, shuffle=True, num_epochs=None):
        # Convert pandas data into a dict of np arrays.
        features = {key:np.array(value) for key,value in dict(self.features).items()}                                           

        # Construct a dataset, and configure batching/repeating.
        ds = tf.python.data.Dataset.from_tensor_slices((features, self.target)) 
        ds = ds.batch(batch_size).repeat(num_epochs)

        # Shuffle the data, if specified.
        if shuffle:
            ds = ds.shuffle(buffer_size=10000)

        # Return the next batch of data.
        features, labels = ds.make_one_shot_iterator().get_next()
        
        return features, labels

    def analyze_result(self, pred):
        print('\n======= RESULT ANALYSIS =======\n')
        
        # Print Mean Squared Error and Root Mean Squared Error.
        mean_squared_error = metrics.mean_squared_error(pred, self.target)
        root_mean_squared_error = math.sqrt(mean_squared_error)
        print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
        print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

        min_house_value = self.target.min()
        max_house_value = self.target.max()
        min_max_difference = max_house_value - min_house_value

        print("Min. Target Value: %0.3f" % min_house_value)
        print("Max. Target Value: %0.3f" % max_house_value)
        print("Difference between Min. and Max.: %0.3f" % min_max_difference)
        print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

        calibration_data = pd.DataFrame()
        calibration_data["predictions"] = pd.Series(pred)
        calibration_data["targets"] = pd.Series(self.target)
        print(calibration_data.describe())
        print('\n=============================\n')        

    def train(self, steps, *fn):
        training_input_fn, prediction_input_fn = fn
        self.model.train(input_fn = training_input_fn, steps = steps)

        pred = self.model.predict(input_fn = prediction_input_fn)
        return np.array([item['predictions'][0] for item in pred])


    def run_model(self, steps, batch_size, epochs):
        tr_fn = lambda:self.input_fn(batch_size=batch_size)
        pr_fn = lambda:self.input_fn(num_epochs=epochs, shuffle=False)

        predictions = self.train(steps, tr_fn, pr_fn)
        self.analyze_result(predictions)
        return predictions
