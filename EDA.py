import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, probplot

from statsmodels.stats.outliers_influence import variance_inflation_factor

class EDA:
    
    df = np.nan
    df_desc = np.nan
    target = np.nan
    analysis = pd.DataFrame(columns=["column","key","value"]) # dataframe with all the actions to be taken for each column
    potential_features = {} # dictionary with potential features that we may have to create
    collinear = pd.DataFrame() # a dataframe that will include potential multicollinear columns to be droped.
    
    def __init__(self, df, target, category_sensibility=10, forced_number=[], forced_binary=[], forced_category=[], to_excel=False):
        '''
        init EDA
        Class that allow a quick analysis of the dataset variables.
        Parameters:
        df: dataset to analyze
        target: name of the target column
        category_sensibility: number of unique values below which a numerical variable will be analyzed as a category
        forced_category: list of columns that we want to be treated as category even though they may not look like
        forced_binary: list of columns that we want to be treated as binary
        '''

        # variables locales
        MAX_UNIQUE_VALUES_PRINTED = 15 # Maximum number of values to be ploted. For large categories.

        # Copy the original info to the class
        self.df = df.copy()
        self.target = target
        
        # We will create a descriptive dataset that will help us do the analysis
        df_desc = df.describe(include="all").T

        # Create a number of vars that will describe each column --> WARNING: it is necessary to sort them later in the code. If not, they will not show.
        df_desc["type"] = np.nan
        df_desc["type_isforced"] = np.nan
        df_desc["count"] = np.nan
        df_desc["nulls"] = np.nan
        df_desc["nulls_perc"] = np.nan
        df_desc["unique"] = np.nan
        _s = pd.Series(np.nan, dtype='object')
        df_desc["unique_values"] = _s
        _s = pd.Series(np.nan, dtype='object')
        df_desc["outliers"] = _s
        df_desc["has_outliers"] = np.nan
        df_desc["mode"] = np.nan
        df_desc["freq"] = np.nan
        df_desc["freq_perc"] = np.nan
        df_desc["skew"] = np.nan


        # Loop throught the dataset columns
        for column in df_desc.index:        
            # common descriptors
            df_desc.loc[column,"dtype"] = df[column].dtype
            df_desc.loc[column,"count"] = df[column].count()
            df_desc.loc[column,"nulls"] = df[df[column].isna()][column].size
            df_desc.loc[column,"nulls_perc"] = round(df_desc.loc[column,"nulls"]/df[column].size,2)
            df_desc.loc[column,"unique"] = df[column].nunique()

            if df_desc.loc[column,"unique"] <= MAX_UNIQUE_VALUES_PRINTED: 
                df_desc.at[column,"unique_values"] = list(df[column].unique())
            else:
                df_desc.at[column,"unique_values"] = []

            # Depeding on the DATATYPE
            # FIRST, we make sure this is not a FORCED NUMBER
            if (column in forced_number):
                # Type of data
                df_desc.loc[column,"type"] = "number"
                df_desc.loc[column,"type_isforced"] = 1

                # Check for outliers
                bounds, df_desc.loc[column,"has_outliers"] = self.outliers(column, type="IQR")
                df_desc.at[column,"outliers"] = bounds
                # skweness
                df_desc.loc[column,"skew"] = df[column].skew()
            # SECOND, we make sure this is not a forced FORCED CATEGORY
            elif (column in forced_category):
                # Tyep of data
                df_desc.loc[column,"type"] = "category"
                df_desc.loc[column,"type_isforced"] = 1

                if (df[column].dtype=="int") |  (df[column].dtype=="float"):
                    df_desc.loc[column,"min"] = df[column].min()
                    df_desc.loc[column,"max"] = df[column].max()

                # Modes and frequencies
                _s_modefreq = df[column].value_counts().sort_values(ascending=False)
                if (_s_modefreq.size < df[column].size): 
                    df_desc.loc[column,"mode"] = _s_modefreq.index[0]
                    df_desc.loc[column,"freq"] = _s_modefreq.iloc[0]
                    df_desc.loc[column,"freq_perc"] = round(_s_modefreq.iloc[0]/df_desc.loc[column,"count"],2)
            # THIRD, we make sure this is not a FORCED BINARY
            elif (column in forced_binary):
                # Type of data
                df_desc.loc[column,"type"] = "binary"
                df_desc.loc[column,"type_isforced"] = 1

            # FOURTH INT OR FLOAT (and not forced, in case)
            elif (((df[column].dtype == "int") | (df[column].dtype == "float")) & (column not in forced_category) & (column not in forced_binary)):

                # Is it numeric , binary or categorical
                if ((df_desc.loc[column,"nulls"]==0) & (df_desc.loc[column,"unique"]==2) & (0 in list(df_desc.loc[column,"unique_values"]))):
                    df_desc.loc[column,"type"] = "binary"
                    df_desc.loc[column,"type_isforced"] = -1
                else:
                    if ((df_desc.loc[column,"nulls"]>0) & (df_desc.loc[column,"unique"]==2) & (0 in list(df_desc.loc[column,"unique_values"]))):
                        df_desc.loc[column,"type"] = "binary"
                        df_desc.loc[column,"type_isforced"] = -1
                    else:
                        if ((df_desc.loc[column,"unique"]<=category_sensibility)):
                            df_desc.loc[column,"type"] = "category"
                            df_desc.loc[column,"type_isforced"] = -1
                        else:
                            df_desc.loc[column,"type"] = "number"
                            df_desc.loc[column,"type_isforced"] = 0
                            # Outliers
                            bounds, df_desc.loc[column,"has_outliers"] = self.outliers(column, type="IQR")
                            df_desc.at[column,"outliers"] = bounds
                            # skweness
                            df_desc.loc[column,"skew"] = df[column].skew()

                # Modes and frequencies only if category
                if ((df_desc.loc[column,"type"] == "category") | (df_desc.loc[column,"type"] == "binary")):
                    _s_modefreq = df[column].value_counts().sort_values(ascending=False)
                    # Si la serie de frecuencias es más pequeña que la serie de la columna, entonces tiene algo de sentido informarlo
                    if (_s_modefreq.size < df[column].size): 
                        df_desc.loc[column,"mode"] = _s_modefreq.index[0]
                        df_desc.loc[column,"freq"] = _s_modefreq.iloc[0]
                        df_desc.loc[column,"freq_perc"] = round(_s_modefreq.iloc[0]/df_desc.loc[column,"count"],2)
            # FIFTH, objects
            elif df[column].dtype == "object":
                # Type of data
                df_desc.loc[column,"type"] = "category"
                df_desc.loc[column,"type_isforced"] = 0

                # Modes and frequencies
                if (df_desc.loc[column,"type"] == "category"):
                    _s_modefreq = df[column].value_counts().sort_values(ascending=False)
                    if ((_s_modefreq.size < df[column].size) & (_s_modefreq.size>0)): # we include >0. If there are no unique values (all nulls) it would crash
                        df_desc.loc[column,"mode"] = _s_modefreq.index[0]
                        df_desc.loc[column,"freq"] = _s_modefreq.iloc[0]
                        df_desc.loc[column,"freq_perc"] = round(_s_modefreq.iloc[0]/df_desc.loc[column,"count"],2)
            elif df[column].dtype == "bool":
                # Type of data
                df_desc.loc[column,"type"] = "binary"
                df_desc.loc[column,"type_isforced"] = 1

                _s_modefreq = df[column].value_counts().sort_values(ascending=False)
                if ((_s_modefreq.size < df[column].size) & (_s_modefreq.size>0)): # # we include >0. If there are no unique values (all nulls) it would crash
                    df_desc.loc[column,"mode"] = _s_modefreq.index[0]
                    df_desc.loc[column,"freq"] = _s_modefreq.iloc[0]
                    df_desc.loc[column,"freq_perc"] = round(_s_modefreq.iloc[0]/df_desc.loc[column,"count"],2)
            else:
                print("******************************* columna", column, "es de tipo", df[column].dtype)

        # Sort the columns for better comprehension
        columns = [
            'dtype',
            'type',
            'type_isforced', 
            'count', 
            'nulls', 
            'nulls_perc', 
            'unique', 
            'unique_values',
            'outliers', 
            'has_outliers',
            'skew',
            'mean', 
            'std', 
            'min', 
            '25%', 
            '50%', 
            '75%', 
            'max', 
            'mode', 
            'freq', 
            'freq_perc'
        ]
        df_desc = df_desc[columns]

        # Excel generation if needed
        if to_excel: 
            df_desc.insert(loc=0,column="NOTES",value=np.nan)
            df_desc.insert(loc=1,column="DROP",value=np.nan)
            df_desc.insert(loc=2,column="ACTIONS",value=np.nan)
            df_desc.to_excel("df_desc.xlsx")
            df_desc.drop(["NOTES","DROP","ACTIONS"], axis=1, inplace=True)

        # Update self
        self.df_desc = df_desc.copy()
        del df_desc

    def outliers(self, column, type='IQR'):
        '''
        OUTLIERS
        Method that returns outliers thresholds
        Parámetros
        column: column name to analyze
        type: outliers calculation (IQR or stdev)
        
        Returns
        [lower_outbound, upper_outbound]: lower threshold, upper threshold
        True if outliers, False if no
        '''

        if 'IQR':
            quantile25 = self.df[column].quantile(0.25)
            quantile75 = self.df[column].quantile(0.75)
            IQR = quantile75-quantile25
            upper_bound = quantile75 + 1.5*IQR
            lower_bound = quantile25 - 1.5*IQR
        else: #std
            mean = self.df[column].mean()
            std = self.df[column].std()
            margin = 3 * std
            lower_bound = mean - margin
            upper_bound = mean + margin

        has_outliers = False
        if ((self.df[self.df[column]<lower_bound][column].count()>0) | (self.df[self.df[column]>upper_bound][column].count()>0)): 
            has_outliers = True
            bounds = [lower_bound, upper_bound]
        else:
            bounds = np.nan

        return bounds, has_outliers
    
    def univar(self, column, log=False, sort="weight", fillna=np.nan):
        '''
        uni
        Function to do a univariant analysis of the corresponding column of the dataset
        
        Parameters
        column: column to analyze
        log: for logistic y scale
        sort: weight (default) or relevance (target)
        fillna: value to replace nulls in categorical descriptions (if wanted)
    
        Returns
        Variable visualization
        '''
        MAX_CATEGORIES_TO_PLOT = 300
        MIN_DATA_TO_NODE = 400

        l_data = ["dtype","type","type_isforced","count","nulls","nulls_perc"]
        # We add the different descriptors depending on the type of data
        if (self.df_desc.loc[column,"type"]=="binary"):
            l_data.extend(["mean"])
        elif (self.df_desc.loc[column,"type"]=="category"):
            l_data.extend(["unique","min","max","mode","freq","freq_perc"])
        else: #number
            l_data.extend(["mean","std","min","25%","50%","75%","max","mode","freq","freq_perc"])
        display(HTML(pd.DataFrame(self.df_desc.loc[column,l_data]).T.to_html()))

        # crosstable when column and target are category/binary
        if ((self.df_desc.loc[column,"type"]!="number") & (self.df_desc.loc[self.target,"type"]!="number")):
            
            display(HTML("********** <b>Valores únicos vs. Target (not nulls)</b> *************"))
            _df = pd.crosstab(self.df[column], self.df[self.target], margins= True, margins_name='Count')
            _df.sort_values(by="Count", ascending=False, inplace=True)
            _df.insert(0,"%Weight",value=0)
            _df.insert(1,"%Positive",value=0)
            _df["%Weight"] = round((_df.iloc[:,1]+_df.iloc[:,2])/(self.df[column].value_counts().sum()),2)
            _df["%Positive"] = round(_df.iloc[:,2]/_df.iloc[:,4],2)
            _df = _df.reindex(columns=list(_df.columns[[-1]].append(_df.columns[:-1])))
            display(HTML(_df.to_html()))
        
        # if COLUMN is NUMBER
        if (self.df_desc.loc[column,"type"]=="number"):
            # Univariant analysis
            fig, axes = plt.subplots(3 if column != self.target else 2, 1, figsize=(10,7), sharex=True)
            fig.tight_layout(pad=3)
            plt.xticks(rotation=90)
            axes[0].set_title("{} distribution (skw: {})".format(column,round(self.df[column].skew(),2)))
            axes[0].grid(True)
            axes[0].margins(0.2)
            try:
                sns.distplot(
                    self.df[column], 
                    ax = axes[0], 
                    fit = norm, 
                )
            except: pass
            
            axes[1].set_title("{} distribution".format(column))
            axes[1].grid(True)
            axes[1].margins(0.2)
            sns.boxplot(
                x = self.df[column], 
                ax = axes[1],  
            )
            
            # Bivariant analysis, only if not target
            if column != self.target:
                # if TARGET is NUMBER
                if self.df_desc.loc[self.target,"type"]=="number":
                    axes[2].set_title("{} vs {}".format(column, self.target))
                    axes[2].grid(True)
                    axes[2].margins(0.2)
                    sns.regplot(
                        x = self.df[column],
                        y = self.df[self.target], 
                        ax = axes[2], 
                    )
            
                # if TARGET is BINARY or CATEGORY
                if self.df_desc.loc[self.target,"type"]!="number":

                    # Graficamos sólo con los datos que tienen suficiente volumen
                    _pivot = pd.pivot_table(self.df, index=[column], values=self.target, aggfunc=[np.mean, len])
                    _pivot.reset_index(inplace=True)
                    _pivot.columns = _pivot.columns.get_level_values(0)
                    _pivot = _pivot[_pivot["len"]>=MIN_DATA_TO_NODE] #Eliminamos los registros que no aportan volumen suficiente

                    plt.figure(figsize=(15,3))
                    plt.xticks(rotation=90)
                    plt.title("{} (order) vs % {} with more than {} records".format(column, self.target, MIN_DATA_TO_NODE))
                    _pivot.sort_values(by="mean", ascending=False, inplace=True)
                    sns.barplot(
                        x=_pivot[column],
                        y=_pivot["mean"],
                        order=_pivot[column]
                    )
                    plt.show()

                    plt.figure(figsize=(15,3))
                    plt.xticks(rotation=90)
                    plt.title("{} vs % {} (order) with more than {} records".format(column, self.target, MIN_DATA_TO_NODE))
                    _pivot.sort_values(by=column, ascending=False, inplace=True)
                    sns.barplot(
                        x=_pivot[column],
                        y=_pivot["mean"],
                        order=pd.Series(_pivot[column].unique()).sort_values().to_list()
                    )
                    plt.show()

                    plt.figure(figsize=(15,3))
                    plt.xticks(rotation=90)
                    plt.title("{} vs % {} with more than {} records (order)".format(column, self.target, MIN_DATA_TO_NODE))
                    _pivot.sort_values(by="len", ascending=False, inplace=True)
                    sns.barplot(
                        x=_pivot[column],
                        y=_pivot["mean"],
                        order=_pivot[column].to_list()
                    )
                    plt.show()
                
            plt.show()
            
            # if TARGET is NUMBER (this is part of the unvariant analysis)
            if self.df_desc.loc[self.target,"type"]=="number":
                # And we finally check the potential transformation to obtain a normal distribution
                self.column_potential_transformation(column)
                    
        # if CATEGORY or BINARY
        if (self.df_desc.loc[column,"type"]=="category") | (self.df_desc.loc[column,"type"]=="binary"):
            fig, axes = plt.subplots(2, 1, figsize=(10,5), sharex=True)
            fig.tight_layout(pad=3)
            plt.xticks(rotation=90)
            # Univariant analysis
            # If the number of categories is too high we show a warning
            num_categories = self.df[column].nunique()
            if (num_categories >= MAX_CATEGORIES_TO_PLOT):
                axes[0].text(0.5, 0.5, "<b>WARNING: TOO MANY CATEGORIES TO PLOT</b>", size=24, ha='center', va='center')
            else:
                s_ = self.df[column].fillna(fillna)
                s_ = s_.value_counts().sort_values(ascending=False) 
                axes[0].set_title("{} distribution".format(column))
                axes[0].grid(True)
                axes[0].margins(0.2)
                sns.barplot(
                    x = s_.index, 
                    y = s_, 
                    log = log,
                    order = s_.index, 
                    ax = axes[0], 
                )
                rects = axes[0].patches
                labels = [str(int(np.round(100*i/s_.sum(),0)))+"%" for i in s_]
                for rect, label in zip(rects, labels):
                    height = rect.get_height()
                    axes[0].text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
            
            # Bivariant analysis, only if not target
            if column != self.target:
                # vs NUMERIC TARGET
                if self.df_desc.loc[self.target,"type"]=="number":
                    axes[1].set_title("{} vs {} distribution".format(column, self.target))
                    axes[1].grid(True)
                    axes[1].margins(0.2)
                    sns.boxplot(
                        x = self.df[column].fillna(fillna), 
                        y = self.df[self.target], 
                        order = s_.index, 
                        ax = axes[1], 
                    )
                # vs BINARY or CATEGORY TARGET
                # we can use positive ratios, easier to plot and understand
                if self.df_desc.loc[self.target,"type"]!="number":
                    _df = pd.DataFrame(self.df.groupby(column)[self.target].mean()).reset_index()
                    columns = _df.columns[:-1].tolist()
                    columns.append("%" + self.target)
                    _df.columns = columns
                    _df.sort_values(by=column, inplace=True)

                    axes[1].set_title("{} (orden) vs % {} distribution".format(column,self.target))
                    axes[1].grid(True)
                    axes[1].margins(0.2)
                    sns.barplot(
                        x=_df[column],
                        y=_df["%"+self.target],
                        order=pd.Series(self.df[column].unique()).sort_values().to_list(), # Mismo orden que los anteriores
                        ax = axes[1]
                    )
                plt.show()
        
        # To finish, we print the next column :-)
        column_i = self.df.columns.tolist().index(column)
        if (column_i+1) == len(self.df.columns): print("--> no more columns")
        if (column_i+1) < len(self.df.columns): print("--> next column ({} of {}): {}".format(column_i+1+1, len(self.df.columns), self.df.columns[column_i+1]))

    def multivar(self):
        '''
        multi
        Function that plots the multivariant analysis of numerical columns in the dataset
        
        parameters
        none
        
        returns
        visualization of correlation matrix and vif values
        '''
        
        # Correlation
        corr_ = self.df.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(25, 25))

        # Generate a custom diverging colormap
        #sns.set_theme(style="white")
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr_, annot=True, mask=mask, cmap=cmap, vmax=1, center=0,
                  square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.show()
        
        #Variance Inflation Factor, only for numeric variables
        numeric_columns = self.df_desc[self.df_desc["type"]=="number"].index.to_list()
        numeric_columns.remove(self.target) # we do not want the target to be part of the VIF
        vif = pd.DataFrame()
        vif["variables"] = numeric_columns
        vif["VIF"] = [variance_inflation_factor(self.df[numeric_columns].dropna().values, i) for i in range(len(numeric_columns))]
        display(vif)
        
        # We store the columns that are potentially multicollinear
        self.collinear = vif[vif["VIF"]>=5.0]
        
        
    def column_potential_transformation(self, column):
        s = self.df[column].dropna() # we must get rid of nulls, as we will be performing some calculations
        
        has_nulls = len(s)<len(self.df[column])
        has_neg = s.min()<0
        has_zeros = len(s[s==0])>0
        
        plt.figure(figsize=(15,10))
        plt.tight_layout(pad=0.8)
        
        plt.subplot(3, 3, 1)
        try:
            s_sqr_ = s**0.5
            sns.distplot(s_sqr_, fit = norm)
        except: pass
        plt.title('Square Root (swk: ' + str(round(s_sqr_.skew(),2)) + ')')
        plt.subplot(3, 3, 4)
        res = probplot(s_sqr_, plot=plt)
        plt.title('')
        plt.subplot(3, 3, 7)
        sns.boxplot(s_sqr_)

        plt.subplot(3, 3, 2)
        try: 
            s_rec_ = 1/s
            sns.distplot(s_rec_, fit = norm)
        except: pass
        plt.title('Reciproque (skw: ' + str(round(s_rec_.skew(),2)) + ')')
        plt.subplot(3, 3, 5)
        res = probplot(s_rec_, plot=plt)
        plt.title('')
        plt.subplot(3, 3, 8)
        sns.boxplot(s_rec_)
        

        # For the log, let's check if data is positive
        # If negative, nothing can be done. If zero values, we can log(x+1)
        plt.subplot(3, 3, 3)
        s_log_ = np.nan
        if not has_zeros and not has_neg:
            s_log_ = np.log(s)
        elif has_zeros and not has_neg:
            s_log_ = np.log(s + 1)
        try:
            sns.distplot(s_log_, fit = norm)
        except: pass
        plt.title('Logaritmic (skw: ' + str(round(s_log_.skew(),2)) + ')')
        plt.subplot(3, 3, 6)
        res = probplot(s_log_, plot=plt)
        plt.title('')
        plt.subplot(3, 3, 9)
        sns.boxplot(s_log_)

        plt.show()
    
    def add_analysis(self, column, key, value=np.nan):
        '''
        add_analysis
        parameters:
        column: name of the dataset's column
        key: what is the analysis about
        value: the value of the kind
        '''
        self.analysis = self.analysis.append({'column':column, 'key':key, 'value':value}, ignore_index=True)
    
    def add_potential_feature(self, feature, description=""):
        '''
        add_potential_feature
        parameters:
        feature: name of the potential feature
        description: string with the description of this feature
        '''
        self.potential_features[feature] = {'description':description}
