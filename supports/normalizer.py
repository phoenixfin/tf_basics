import numpy as np

class DataNormalizer(object):
    def __init__(self, dataframe):
        self.df = dataframe

    def get_outlier(self, feature, method, factor):
        if method == 'standard deviation':
            upper_lim = self.df[feature].mean() + self.df[feature].std() * factor
            lower_lim = self.df[feature].mean() - self.df[feature].std() * factor
        elif method == 'quantile':
            upper_lim = self.df[feature].quantile(1-factor)
            lower_lim = self.df[feature].quantile(factor)
        outliers = data[(self.df[feature] < upper_lim) & (self.df[feature] > lower_lim)]
        return upper_lim, lower_lim, outliers

    def remove_nan(self, feature, method):
        data = self.df[feature]
        fill_value = {
            'zero' : 0,
            'median' : data.median(),
            'mean' : data.mean(),
            'mode' : data.mode()
        }
        self.df[feature] = data.fillna(fill_value[method])

    def normalize(self, feature, method, *args):
        data = self.df[feature]
        method_name = method.replace(' ','_').lower()
        result = getattr(self, '_'+method_name)(data, *args)
        self.df[feature] = result
        
    def _range_scaling(self, data, lower=0, upper=1):
        dmax = np.max(data)
        dmin = np.min(data)
        old_range = dmax-dmin
        scaled_range = upper-lower
        return lower + ((data-dmin)/old_range)*scaled_range        

    def _clipping(self, data, lower=None, upper=None):
        return data.clip(lower, upper)
        
    def _log_scaling(self, data, base = np.e):
        return np.log(data+1)/np.log(base)
    
    def _zvalue_scaling(self, data):
        return (data - data.mean())/(data.std()+0.000001)
    
    
    