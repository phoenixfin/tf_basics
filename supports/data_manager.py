import pandas as pd
import numpy as np

class DataNormalizer(object):
    def __init__(self, dataframe, feature=None):
        self.old_df = dataframe
        self.ori_df = dataframe
        self.df = dataframe
        if feature == None: feature = self.df.columns[0]
        self.feature = feature
        self.transform_list = []

    @property
    def data(self):
        return self.df[self.feature]

    @data.setter
    def data(self, value):
        self.df[self.feature] = value

    def _general_transform(self, method_name):
        self.transform_list.append(method_name+'_'+self.feature)
        self.old_df = self.df

    def change_feature(self, new_feature):
        self.feature = new_feature

    def get_mean_std(self):
        return self.data.mean(), self.data.std()
    
    def get_range(self):
        dmax = float(self.data.max())
        dmin = float(self.data.min())
        return dmax, dmin, dmax-dmin

    def undo_previous_transformation(self):
        self.df = self.old_df
        self.transform_list.pop()

    def restore_dataframe(self):
        self.df = self.ori_df

    def binning(self, method, bins=10, labels=[]):
        self._general_transform('binning')
        if method == 'quantile':
            ds = 1./bins
            grids = [self.data.quantile(ds*i) for i in range(bins+1)]
        elif method == 'value':
            dmax, dmin, drange = self.get_range()
            ds = drange/bins
            grids = np.arange(dmin, dmax+ds, ds)
        if not labels: labels = [method+'_bin_'+str(j) for j in range(bins)]
        self.df[self.feature+' bin'] = pd.cut(self.data, bins=grids, labels=labels)

    def bucketing(self, vocab, labels, group=None):
        self._general_transform('bucketing')
        condition = [[k in vocab[label] for k in self.data] for label in labels]
        if group is None:
            group = self.feature + ' grouping'
        self.df[group] = np.select(condition, labels, default='Other')
    
    def one_hot_encoding(self, remove_original=True):
        self._general_transform('one_hot_encoding')        
        categorized = pd.get_dummies(self.data)
        if remove_original: self.df.pop(self.feature)
        self.df = pd.concat((self.df, categorized), axis=1)        

    def get_outlier(self, method, factor):
        if method == 'standard deviation':
            mean, std = self.get_mean_std()
            upper_lim = mean + std * factor
            lower_lim = mean - std * factor
        elif method == 'quantile':
            upper_lim = self.data.quantile(1-factor)
            lower_lim = self.data.quantile(factor)
        outliers = self.data[(self.data > upper_lim) | (self.data < lower_lim)]
        return upper_lim, lower_lim, outliers

    def data_filler(self, method, fillzero = False):
        self._general_transform('data_filling')        
        fill_value = {
            'zero' : 0,
            'median' : self.data.median(),
            'mean' : self.data.mean(),
            'mode' : self.data.mode()
        }
        self.data = self.data.fillna(fill_value[method])
        if fillzero:
            self.data = self.data.replace(0, fill_value[method])

    def normalize(self, method, *args):        
        method_name = method.replace(' ','_').lower()
        self._general_transform(method_name)              
        self.data = getattr(self, '_'+method_name)(*args)
        
    def _range_scaling(self, lower=0, upper=1):
        dmax, dmin, drange = self.get_range()
        scaled_range = upper-lower
        return lower + ((self.data-dmin)/drange)*scaled_range        

    def _clipping(self, lower=None, upper=None):
        return self.data.clip(lower, upper)
        
    def _log_scaling(self, base = np.e):
        return np.log(self.data+1)/np.log(base)
    
    def _z_value_scaling(self):
        mean, std = self.get_mean_std()
        return (self.data - mean)/(std+0.000001)
    
    