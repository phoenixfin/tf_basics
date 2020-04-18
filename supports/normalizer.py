import numpy as np

class DataNormalizer(object):
    def __init__(self, dataframe):
        self.df = dataframe

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
        return np.log(data)/np.log(base)
    
    def _zvalue_scaling(self, data):
        return (data - data.mean())/(data.std()+0.000001)
    
    
    