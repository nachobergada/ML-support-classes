import pandas as pd
import seaborn as sns

class GridSearchCVResults():
    
    '''
    GridSearchCVResults
    Class that shows GridSearchCV results in a more friendly way
    
    parameters:
    GridSearchCV: GridSearchCV trained model
    greater_is_better: False if the score is better the bigger (e.g R2) or better the smaller (eg RMSE)
    '''
    
    d_res = None # cv_results dictionary
    df_res = None # dataframe with the tabulated results
    variable_params = None # list with the params that vary
    static_params = None # list with the params that do not vary
    
    def __init__(self, oGridSearchCV, greater_is_better = True):
        
        self.d_res = oGridSearchCV.cv_results_
        self.df_res = pd.DataFrame(self.d_res["params"])
        
        for key in self.d_res:
            if (str(key).find("split"))!=-1: #it is a split results key
                self.df_res[key] = pd.Series(self.d_res[key])
        self.df_res.insert(0, "rank_test_score", pd.Series(self.d_res["rank_test_score"]))
        self.df_res.insert(1, "mean_test_score", pd.Series(self.d_res["mean_test_score"]))
        self.df_res.insert(2, "std_test_score", pd.Series(self.d_res["std_test_score"]))
        self.df_res.insert(3, "mean_train_score", pd.Series(self.d_res["mean_train_score"]))
        self.df_res.insert(4, "std_train_score", pd.Series(self.d_res["std_train_score"]))
        self.df_res.insert(5, "%mean_diff", round(100*(pd.Series(self.d_res["mean_test_score"]) - pd.Series(self.d_res["mean_train_score"]))/pd.Series(self.d_res["std_train_score"]),2))
        
        # and now, we leave only the columns that have different information
        for column in df_results.columns:
            if self.df_res[column].nunique()==1: self.df_res.drop(column, axis=1, inplace=True)
        self.df_res.sort_values(by="rank_test_score", inplace=True)
        
        self.show()
    
    def show(self):
        '''
        show
        Function that plots the different score values for each ranked combination
        '''
        split_cols = [x for x in self.df_res.columns if x.find("split")!=-1] # only get split columns
        melt_ = self.df_res.melt(id_vars=["rank_test_score"], value_vars=split_cols)
        melt_["variable"] = melt_["variable"].apply(lambda x: "test" if x.find("_test_")!=-1 else "train")
        if not greater_is_better: melt_["value"] = abs(melt_["value"])
        fig = plt.figure(figsize=(10,5))
        sns.lineplot(data=melt_, x="rank_test_score", y="value", hue="variable")
        plt.xticks(melt_["rank_test_score"])
        plt.grid()
        plt.show()
        
    def get_params(self, rank=1):
        '''
        get_params
        Function that prints the parameters of the selected ranked option
        
        params:
        rank: number of rank to display
        '''
        display(HTML(
                f'<b>Showing parameters for rank {rank}</b><br> \
                {self.d_res["params"][rank]}<br> \
                Mean test score: {self.df_res.loc[self.df_res["rank_test_score"]==(rank),"mean_test_score"].values[0]}<br> \
                Mean train score: {self.df_res.loc[self.df_res["rank_test_score"]==(rank),"mean_train_score"].values[0]}<br>'
            )
        )
    
    def show_results(self):
        '''
        show_results
        Method that returns all the data of cv_results_ in a tabular format
        '''
        display(self.df_res)
