import time
import numpy as np

from supports import file_manager, data_manager, graph_manager
from frameworks import regression, deep_learning


def main(df, features, target, tool, learning_rate = 0.01, steps_num = 100, 
         batch_size = 50, epochs_num = 1, validation_split=0.2, plot = False):
    
    start = time.time()
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
    
    error = model.run_model(
        custom_var,
        batch_size = batch_size, 
        epochs = epochs_num,
    )

    if plot:
        graph_manager.plot_regression(df, features[0], target, model.model)
        
    elapsed = round(time.time()-start, 3)
    print("Elapsed time:", elapsed, "seconds")    
    return error, elapsed
    
def iterate_main(df, var, iteration, tool):
    errors = []
    times = []
        
    arglist = {
        'df': df,
        'features': ['GDP per capita'],
        'target': 'Score',
        'learning_rate': 0.01,
        'tool': tool,
        'steps_num': 500,
        'validation_split': 0.3,
        'plot': False
    }
    
    if tool == 'Linear Regressor':
        arglist['batch_size'] = 100
        arglist['epochs_num'] = 1
    elif tool == 'Sequential Layer':
        arglist['batch_size'] = 2
        arglist['epochs_num'] = 50
                
    # Example
    for i in iteration:
        arglist[var] = i
        err, dur = main(**arglist)
        errors.append(err)
        times.append(dur)        

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(var)
    ax1.set_ylabel('RMSE', color=color)
    ax1.plot(iteration, errors, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Processing Time (s)', color=color)  
    ax2.plot(iteration, times, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.show()
    
def world_happiness():
    default_range = {
        'learning_rate': np.arange(0.0001,0.01,0.0005),
        'steps_num': np.arange(50,1000,50),
        'batch_size': np.arange(2, 100, 5)
    }
    
    from matplotlib import pyplot as plt
    dataset1 = file_manager.read_data('world-happiness', '2019.csv')
    dataset2 = file_manager.read_data('world-happiness', '2018.csv')
    dataset = dataset1 + dataset2
        
    var = 'batch_size'
    # iterate_main(dataset, var, default_range[var], 'Linear Regressor')
    # print(dataset.columns)
    
    features = ['GDP per capita', 'Social support', 
                'Healthy life expectancy', 'Freedom to make life choices', 
                'Generosity', 'Perceptions of corruption']
        
    # for feat in features:
    main(
        df = dataset,
        features = ['Perceptions of corruption'],
        target = 'Score',
        tool = 'Linear Regressor',
        learning_rate = 0.01, 
        steps_num = 500,
        batch_size = 100, 
        epochs_num = 1,
        validation_split = 0.3,
        plot = True
    )

if __name__ == "__main__":
    # world_happiness()
    dataset = file_manager.read_data('world-happiness', '2018.csv')
    norm = data_manager.DataNormalizer(dataset)

    """ Part 1: Missing data imputation """    
    for feature in dataset.columns[2:]:
        norm.change_feature(feature)
        norm.data_filler('median', fillzero = True)
        
    """ Part 2: Grouping countries """    
    dataset2 = file_manager.read_data('world-countries-and-continents-details','countries and continents.csv')
    continent_dict = {}
    continents = dataset2['Continent'].unique()
    for continent in continents:
        countries = dataset2[dataset2['Continent']==continent]['name'].tolist()
        continent_dict[continent] = countries
    norm.change_feature('Country or region')
    norm.bucketing(continent_dict, continents, 'Continent')
    print(norm.df)

    """ Part 3: One Hot Encoding Continents """
    norm.change_feature('Continent')
    norm.one_hot_encoding(remove_original = False)
    print(norm.df)
    
    """ Part 4: Binning GDP """
    norm.change_feature("GDP per capita")
    GDP_Labels = ['Low Income', 'Lower-Middle Income', 
                  'Upper-middle Income', 'High Income']
    norm.binning('value', bins=4, labels=GDP_Labels)
    print(norm.df)
    
    """ Part 5: One Hot Encoding GDP Bins """
    norm.change_feature("GDP per capita bin")
    norm.one_hot_encoding(remove_original = False)
    print(norm.df)
    
    """ Part 6: Range Scaling Score """
    norm.change_feature("Score")
    norm.normalize("range scaling")
    print(norm.df)
    
    """ Part 7: Clipping Generosity  """
    norm.change_feature("Generosity")
    norm.normalize("clipping", 0, 0.39)
    print(norm.df)
    
    """ Part 8: Log-Scale Perception of Corruption """
    norm.change_feature("Perceptions of corruption")
    norm.normalize("log scaling")
    
    file_manager.write_data(norm.df, "2018_scaled")
    print(norm.df.describe())
    # for feature in dataset.columns[2:]:
    #     print(feature)
    #     norm.change_feature(feature)
    #     print(norm.get_outlier('standard deviation',3))
    #     print()     

    # # norm.normalize('Z value scaling')
    # print(norm.binning('quantile'))
    # # print(norm.df['Score'])
    # print(norm.df['Score'].hist())
