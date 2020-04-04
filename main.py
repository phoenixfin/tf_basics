from supports import reader, grapher
from frameworks import regression

def main_regression(df, features, target, learning_rate = 0.01, 
                    steps_num = 100, batch_size = 50, 
                    epochs_num = 1, plot = False):

    regressor = regression.LinearRegressor(df)
    regressor.build(
        features = features, 
        target_var = target, 
        learning_rate = 0.001
    )
    regressor.run_model(
        steps = 500,
        batch_size = 200, 
        epochs = 1,
    )

    if plot:
        grapher.plot_regression(dataset, features[0], target, regressor.model)
    
if __name__ == "__main__":
    dataset = reader.read_data('world-happiness', '2019.csv')

    # Example
    main_regression(
        df = dataset, 
        features = ['GDP per capita'],
        target = 'Score',
        learning_rate = 0.001, 
        steps_num = 500, 
        batch_size = 200, 
        epochs_num = 1, 
        plot = True
    )