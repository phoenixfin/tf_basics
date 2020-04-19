"""
Plot functions
"""

from matplotlib import pyplot as plt

def plot_regression(df, x, y, model):
    if type(model).__name__ == "LinearRegressorV2":
        feat = x.replace(' ', '_')
        pos = 'linear/linear_model/'
        weight = model.get_variable_value(pos + feat + '/weights')[0]
        bias = model.get_variable_value(pos + 'bias_weights')
    else:
        weight = model.get_weights()[0][0]
        bias = model.get_weights()[1][0]
        
    print(weight, bias)
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
    
#@title Define the plotting function
def plot_the_loss_curve(epochs, mae_training, mae_validation):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
    plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
    plt.legend()
    
    # We're not going to plot the first epoch, since the loss on the first epoch
    # is often substantially greater than the loss for other epochs.
    merged_mae_lists = mae_training[1:] + mae_validation[1:]
    highest_loss = max(merged_mae_lists)
    lowest_loss = min(merged_mae_lists)
    delta = highest_loss - lowest_loss
    print(delta)

    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)
    
    plt.ylim([bottom_of_y_axis, top_of_y_axis])
    plt.show()  

