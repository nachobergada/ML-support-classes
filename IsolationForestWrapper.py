import pandas as pd
import numpy as np

class IsolationForestWrapper():    
    '''
    IsolationForestWrapper
    This class is a wrapper for IsolationForest. We override the predict method to return fraud/non-fraud instead of outlier/non-outlier
    https://scikit-learn.org/stable/developers/develop.html
    '''
    
    model = None
    n_estimators = None
    max_samples = None
    max_features = None
    contamination = None
    bootstrap = None
    verbose = None
    random_state = None
    
    def __init__(self, n_estimators = 100, max_samples = 1.0, max_features = 1.0, contamination = 'auto', bootstrap = False, verbose = 0, random_state = RANDOM_STATE):
        
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination
        self.bootstrap = bootstrap
        self.verbose = verbose
        self.random_state = random_state
        
        self.model = IsolationForest(
            n_estimators = n_estimators, 
            max_samples = max_samples, 
            max_features = max_features, 
            contamination = contamination, 
            bootstrap = bootstrap, 
            verbose = verbose, 
            random_state = random_state, 
        )
    
    def fit(self, X, y=None):
        self.model.fit(X)
    
    def predict(self, X):
        # IsolationForest predict method returns whether a value is an outlier (-1) or not (1)
        # We have to "transform" our prediction based on this. If -1, we mean Fraud, and by 1 we mean NoFraud
        y_pred = pd.Series(self.model.predict(X)).apply(lambda x: 1 if x==-1 else 0).to_numpy()
        return y_pred # the output is an array of values
    
    def predict_proba(self, X):
        return None
    
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {
            "n_estimators" : self.n_estimators, 
            "max_samples" : self.max_samples, 
            "max_features" : self.max_features, 
            "contamination" : self.contamination, 
            "bootstrap" : self.bootstrap, 
            "verbose" : self.verbose, 
            "random_state" : self.random_state, 
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
