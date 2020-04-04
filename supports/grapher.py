"""
Plot functions
"""

from matplotlib import pyplot as plt

def plot_regression(df, x, y, model):
    feat = x.replace(' ', '_')
    pos = 'linear/linear_model/'
    weight = model.get_variable_value(pos + feat + '/weights')[0]
    bias = model.get_variable_value(pos + 'bias_weights')

    sample = df.sample(n=500, replace = True)
    x_0 = sample[x].min()
    x_1 = sample[x].max()

    y_0 = weight * x_0 + bias 
    y_1 = weight * x_1 + bias

    plt.plot([x_0, x_1], [y_0, y_1], c='r')
    plt.ylabel(y)
    plt.xlabel(x)
    plt.scatter(sample[x], sample[y])
    plt.show()