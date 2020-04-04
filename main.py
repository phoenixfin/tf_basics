from supports import reader, grapher
from frameworks import regression, deep_learning

def main(df, features, target, tool, learning_rate = 0.01, steps_num = 100, 
         batch_size = 50, epochs_num = 1, validation_split=0.2, plot = False):

    if tool == 'Linear Regressor':
        model = regression.LinearRegressor(df)
        custom_var = steps_num
    elif tool == 'Sequential Layer':
        model = deep_learning.SequentialLayer(df)
        custom_var = validation_split
        
    model.build(
        features = features, 
        target_var = target, 
        learning_rate = learning_rate
    )
    
    model.run_model(
        custom_var,
        batch_size = batch_size, 
        epochs = epochs_num,
    )

    if plot:
        grapher.plot_regression(df, features[0], target, model.model)
    
if __name__ == "__main__":
    dataset = reader.read_data('world-happiness', '2019.csv')

    # Example
    main(
        df = dataset, 
        features = ['GDP per capita'],
        target = 'Perceptions of corruption',
        tool = 'Linear Regressor',
        learning_rate = 0.01, 
        steps_num = 500, 
        batch_size = 100, 
        epochs_num = 1, 
        plot = True
    )

    # main(
    #     df = dataset, 
    #     features = ['GDP per capita'],
    #     target = 'Perceptions of corruption',
    #     tool = 'Sequential Layer',
    #     learning_rate = 0.001, 
    #     batch_size = 2, 
    #     epochs_num = 50,
    #     validation_split = 0.3,
    #     plot = True
    # )
    
    