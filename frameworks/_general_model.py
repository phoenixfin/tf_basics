class GeneralModel(object):
    def __init__(self, df):
        self.model = None
        self.train_df = None
        self.features = []
        self.target = None
        
    def _build(self, features, target_var):
        self.features = self.train_df[features]
        self.target = self.train_df[target_var]
            