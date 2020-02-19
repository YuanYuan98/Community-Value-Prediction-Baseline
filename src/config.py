class config:
    def __init__(self):
        self.params = {'max_depth':7,
        'num_leaves':31,
        'objective':'regression',
        'metric':'mae',
        'learning_rate':0.1,
        'max_bin':60,
        'lambda_l1': 0,    
        'lambda_l2': 0.5,
        'bagging_fraction': 0.9,   
        'feature_fraction': 0.8,
        'min_data_in_leaf': 1}

        self.max_round = 300
        self.cv_folds = 10
        self.early_stop_round = 30
        self.categorical_feature = None
        self.feature_size = 21
        self.embedding_size = 64
        self.deepwalk_embedding_size = 64
        self.output_size = 1

Config = config()