import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.stats.stattools import durbin_watson

import statsmodels.api as sm

class LinearRegressionValidator():
    '''
    LinearRegressionValidator
    Class that validate a linear regression model
    https://towardsdatascience.com/how-do-you-check-the-quality-of-your-regression-model-in-python-fa61759ff685
    
    parameters:
    y: target (ground truth)
    y_pred: predicted target
    X: predictors
    lrmodel: linear regression model
    '''
    y = None # target
    y_pred = None # predicted target
    X = None # predictors
    lrmodel = None # regression model
    err = None # errors
    
    def __init__(self, y, y_pred, X, lrmodel):
        self.y = y
        self.y_pred = y_pred
        self.lrmodel = lrmodel
        self.X = X
    
    def validate(self, all=False):
        '''
        validate
        Method that validates the main assumptions of linear regressions
        Thanks to https://jeffmacaluso.github.io/post/LinearRegressionAssumptions/
        '''
        
        p_value_thresh = 0.05 # (5% double sided)
        
        self.err = self.y_pred - self.y
        
        display(HTML("<h1>Linear Regression Validation Report</h1>"))
        
        display(HTML("<h3>Metrics</h3>"))
        display(HTML(
            f"<b>R2:</b> {r2_score(self.y, self.y_pred)} <br> \
            <b>RMSE:</b> {mean_squared_error(self.y, self.y_pred, squared=False)}"
        ))
        
        display(HTML("<h3>Linear Assumption</h3>"))
        # Predicted vs Real
        fig = plt.figure(figsize=(4, 3))
        df_ = pd.DataFrame(self.y.rename("y"), index=self.y.index).join(pd.DataFrame(self.y_pred.rename("y_pred"), index=self.y_pred.index))
        sns.scatterplot(x='y', y='y_pred', data=df_)
        # Plotting the diagonal line
        min_ = max(self.y.min(),self.y_pred.min())
        max_ = max(self.y.max(),self.y_pred.max())
        plt.plot((min_, max_), (min_, max_), color='darkorange', linestyle='--')
        plt.title('Real vs. Predicted')
        plt.show()
        
        display(HTML("<h3>Normal Assumption of errors - Anderson-Darling</h3>"))
        # Using the Anderson-Darling test for normal distribution
        p_value = normal_ad(self.err)[1]
        if p_value <= p_value_thresh: display(HTML(f'<span style="color:red;font-weight:bold">Normal assumption not satisfied (p_value: {p_value})</span> --> confidence intervals will likely be affected. Try to perform nonliear transformations on variables. Info on QQPlot: <a href="https://seankross.com/2016/02/29/A-Q-Q-Plot-Dissection-Kit.html">here</a>.'))
        
        # Normal error distribution
        fig, axes = plt.subplots(1, 2, figsize=(8,3))
        plt.tight_layout(pad=0.3)
        sns.distplot(
            self.y, 
            fit = norm, 
            ax = axes[0], 
        )
        axes[0].set_title(f"Error normality (skw: {round(self.err.skew(),2)})")
        
        #qqplot
        sm.qqplot(self.err, line ='q', ax=axes[1])
        axes[1].set_title(f"qqplot")
        
        plt.show()
        
        display(HTML("<h3>Assumption of non-autocorrelation</h3>"))
        # Assumes that there is no autocorrelation in the residuals. If there is
        # autocorrelation, then there is a pattern that is not explained due to
        # the current value being dependent on the previous value.
        # This may be resolved by adding a lag variable of either the dependent
        # variable or some of the predictors.
        # Durbin-Watson Test
        # Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data
        # 0 to 2< is positive autocorrelation
        # >2 to 4 is negative autocorrelation
        durbinWatson = durbin_watson(self.err)
        if durbinWatson < 1.5:
            display(HTML('<span style="color:red;font-weight:bold">Signs of positive autocorrelation</span>'))
        elif durbinWatson > 2.5:
            display(HTML('<span style="color:red;font-weight:bold">Signs of negative autocorrelation</span>'))
        else:
            display(HTML('No signs of autocorrelation'))
        
        display(HTML("<h3>Assumption of Random error vs predictors - Homoscedasticity</h3>"))
        # If heteroscedasticity: Variables with high ranges may be the cause
        # For the dependent variable, use rates or per capita ratios instead of raw variables. That may change the project.
        # 
        
        fig = plt.figure(figsize=(4, 3))
        sns.scatterplot(x=self.y_pred, y=self.err)
        plt.title('Error vs. Predicted')
        plt.show()
        if all:
            # Random errors vs predictors
            num_pred = len(self.X.columns)
            fig_columns = 3
            fig, axes = plt.subplots(math.ceil(num_pred/fig_columns), fig_columns, figsize=(4*fig_columns, math.ceil(num_pred/fig_columns)*3))
            #plt.tight_layout(pad=1)
            row = 0
            col = 0
            for i in range(num_pred):
                axes[row, col].axhline(y=0, linewidth=4, color='r')
                axes[row, col].set_title(f"Err vs {self.X.columns[i]}")
                sns.scatterplot(
                    x = self.X.iloc[:,i], 
                    y = self.err, 
                    ax = axes[row, col]
                )
                if ((i+1)%fig_columns == 0): col = 0 
                else: col+=1
                if col == 0: row += 1
            plt.show()
            
        # summarize feature importance
        df_imp_ = pd.DataFrame(self.lrmodel.coef_, index=self.X.columns, columns=["importance"])
        df_imp_["importance_abs"] = df_imp_["importance"].abs()
        df_imp_.sort_values(by="importance_abs", ascending=False, inplace=True)
        df_imp_.drop("importance_abs", axis=1, inplace=True)
        s_imp_ = df_imp_["importance"]
        # plot feature importance
        fig = plt.figure(figsize=(15,4))
        plt.xticks(rotation=90)
        sns.barplot(x=s_imp_.index, y=s_imp_.values)
        plt.show()
