import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML

class ProcessDataset:
    '''
    ProcessDataset
    Class that handles data processing. 
    
    Construct
    data: dataset with all the data
    cEDA: instance of the EDA class, with all the analysis run before
    submission: if a test dataset for submission purposes exists, it can be included so that independent variables are also transformed.
    '''
    
    data = np.nan # will include the processed data
    submission = np.nan # will include the processed submission dataset
    index_submission = [] # list with the indexes of the test dataset, when kaggle provides it. That allows us to process the X_test at the same time.
    origColumns = [] # original X columns to be processed
    analysis = np.nan # dataframe with the EDA analysis
    collinear = np.nan # dataframe with the EDA collinearity analysis
    html_report = "<h1>Dataset Processing Report</h1>"
    target = np.nan # target of the database
    
    def __init__(self, data, cEDA, submission = None):
        self.data = data.copy()
        self.collinear = cEDA.collinear
        self.analysis = cEDA.analysis
        self.analysis["processed"] = 0 # 0=not processed | -1=error | 1=success
        if submission is not None:
            self.index_submission = submission.index.to_list()
            self.data = pd.concat([self.data, submission])
        self.target = cEDA.target
        
        # store columns to loop through
        self.origColumns = self.data.columns
        
        # initialize html_report
        self.html_report += str(cEDA.analysis.to_html())
        
        display(HTML(f'<li>Act on collinearity.</li><li>You can proceed with the first column: {self.origColumns[0]}</li>'))
    
    def drop_collinear(self, vif_thresh=5):
        '''
        Function that drop collinear columns from both the X and the analysis dataframes
        
        parameters:
        none
        
        returns:
        none
        '''
        
        if len(self.collinear)>0:
            t_ = '<h3>--> COLLINEARITY Results</h3><br>'
            t_ += 'Original Data shape: ' + str(self.data.shape) + "<br>"
            
            # Drop from Data
            columns_to_drop = self.collinear[self.collinear["VIF"]>=vif_thresh]["variables"].to_list()
            self.data.drop(columns_to_drop, axis=1, inplace=True)
            
            # store columns to loop through
            self.origColumns = self.data.columns
            
            # Drop from analysis
            index_to_drop = self.analysis[self.analysis["column"].isin(columns_to_drop)].index
            self.analysis.drop(index_to_drop, axis=0, inplace=True)
            
            t_ += 'Columns with vif>=' + str(vif_thresh) + ' have been dropped: ' + str(columns_to_drop) + '.<br>'
            t_ += 'New X shape: ' + str(self.data.shape) + "<br>"
            
            display(HTML(t_))
        
            # Add to report
            self.update_report(t_)

    def show_var(self, column):
        '''
        show_var
        Function that gives the information of the column
        parameters:
        column: name of the column
        returns:
        analysis info visualization of the corresponding variable
        '''
        
        s_ = pd.Series(self.analysis[self.analysis["column"]==column]["value"].values, index=self.analysis[self.analysis["column"]==column]["key"].values)
        t_ = '<b>{}</b>'.format(column) + '<br>'
        if "description" in s_.index.to_list():
            t_ += '<u>Analysis</u>: {}'.format(s_.loc["description"]) + '<br>'
            s_.drop("description", inplace=True)
        if len(s_)>0:
            t_ += "<u>Actions</u>:"
            for i in range(len(s_)):
                t_ += '<li>' + str(s_.index[i]) + ": " + str(s_.iloc[i]) + "</li>"
        
        display(HTML(t_))
        
        iColumn = self.origColumns.to_list().index(column)
        if (iColumn+1) == len(self.origColumns): display(HTML('No more columns'))
        else: display(HTML(f'Next column: {self.origColumns[iColumn+1]}'))                
    
    def set_others(s, num_categories=0, min_elem_categories=0, categories=np.nan, others_name="Others", fill_nulls_value=np.nan):
        #----------------------------------------------------
        #----------------------------------------------------
        # CATEGORY_SET_OTHERS
        # Función que permite categorizar un número X de categorías para una serie
        # Parámetros
        # s: serie a categorizar
        # num_categories: número de categorías a incluir, de la más a la menos frecuente
        # others_name: nombre de la categoría Others
        # fill_nulls_value: Aprovechamos para rellenar los valores nulos con su propia categoría
        #
        # Retorna
        # 1. Serie categorizada o dataframe concatenado
        #----------------------------------------------------
        #----------------------------------------------------
        # Primero tratamos los nulls
        if (fill_nulls_value is not np.nan): s.fillna(fill_nulls_value, inplace=True)

        if (categories is not np.nan):
            topCategories = categories
        elif (num_categories > 0):
            topCategories = s.value_counts().head(num_categories)
            topCategories = topCategories.index.tolist()
        elif (min_elem_categories > 0):
             _s = s.value_counts()
             topCategories = _s[_s>=min_elem_categories].index.tolist()


        # Añadimos el valor nulo a las categorías
        if (fill_nulls_value is not np.nan): topCategories.append(fill_nulls_value)

        # Añadimos la categoría adicional de Others
        topCategories.append(others_name)
        s = pd.Categorical(s, categories=topCategories).fillna(others_name) ## Las que no encajan con la lista se asocian a Others

        return s

    # ENCODERS
    def to_ohe(self, column, n_1=True, new_name=None, repeat=False):
        '''
        to_ohe
        Function that transforms column to ohe and updates the X dataframe
        Parameters
        column: name of the variable to transform or list of columns for multiple ohe fusion.
        n_1: if True, drop one of the columns, as the value can be defined with the rest.
        new_name: if a new name for the column must be given
        repeat: if column is a list, it indicates that two or more values from the different columns cannot be repeated.
        
        Return
        Updated dataframe
        '''
        #first, we check if we have a column or a list of columns
        multiple = isinstance(column, list)
        if multiple: n_1=False # if joining multiple ohe's, we cannot do n-1 as the information would be incomplete
        
        if not multiple:
            # prefix initialization
            prefix = column

            # get_dummies (generate OHE)
            dumm_ = pd.get_dummies(self.data[column], prefix=prefix, prefix_sep="_")
            if n_1: dumm_.drop(dumm_.columns[-1], axis=1, inplace=True) # we get rid of the last column, specially in regression
            t_ = '<h3>--> ' + column + ' OHE Results (n_1: ' + str(n_1) + ')</h3><br>'
            t_ += 'OHE generated a total of ' + str(len(dumm_.columns)) + ' columns.<br>'
            t_ += 'Original Data shape: ' + str(self.data.shape) + "<br>"

            # Concat OHE columns
            self.data = pd.concat([self.data.drop(column, axis=1), dumm_], axis=1)
            t_+= f'New Data shape: {str(self.data.shape)} ({str(len(dumm_.columns))} OHE -1 dropped)<br>'
            display(HTML(t_))

            # Update analysis
            self.update_analysis(column, "ohe", 1)
        else:
            dumm_ = pd.DataFrame()
            # We generate a dummy for each column in the list
            for iColumn, nColumn in enumerate(column):
                coldumm_ = pd.get_dummies(self.data[nColumn], prefix=new_name, prefix_sep="_")
                # loop through the new dummied dataframe
                for nColumnDumm in coldumm_.columns:
                    if nColumnDumm in dumm_.columns: # if the col already exists, we must add the result
                        dumm_[nColumnDumm] = dumm_[nColumnDumm] + coldumm_[nColumnDumm]
                        if not repeat:
                            dumm_.iloc[dumm_[nColumnDumm]>0,dumm_.columns.to_list().index(nColumnDumm)]=1
                    else: # if the col does not exist, we create the column
                        dumm_[nColumnDumm] = coldumm_[nColumnDumm]
            
            t_ = f'<h3>--> {str(column)} (MULTIPLE) OHE Results (n_1: {str(n_1)}, repeat: {str(repeat)})</h3><br>'
            t_ += f'OHE generated a total of {str(len(dumm_.columns))} columns.<br>'
            t_ += f'Original Data shape: {str(self.data.shape)}<br>'
            
            # Concat OHE columns
            self.data = pd.concat([self.data.drop(column, axis=1), dumm_], axis=1)
            t_+= f'New Data shape: {str(self.data.shape)} ({str(len(dumm_.columns))} OHE -{len(column)} dropped)<br>'
            display(HTML(t_))

            # Update analysis
            # self.update_analysis(column, "ohe", 1) #TODO
        
        # Add to report
        self.update_report(t_)
    
    def to_oe(self, column, mapping):
        '''
        to_oe
        Function that transforms column to ordinal encoding and updates the X dataframe
        Parameters
        column: name of the variable to transform
        mapping: the value map (list of tupples)
        
        Return
        Updated dataframe
        '''
        
        t_ = '<h3>--> ' + column + ' OE Results</h3><br>'
        
        # map formating (we need list dictionaries)
        dict_map = {}
        for i in range(len(mapping)):
            dict_map[mapping[i][0]] = mapping[i][1]
        
        s_ = self.data[column].map(dict_map) # we store it in a temp variable to check if everything went well before proceeding
        if (len(s_[s_.isna()])>0): # that means that either the mapping was not correct, or that there are still null values
            t_ += 'WARNING: The transformed column would still have null values. Therefore, not able to encode the variable.<br>'
            # Update analysis
            self.update_analysis(column, "oe", -1)
        else:
            t_ += 'Original Data shape: ' + str(self.data.shape) + "<br>"
            dfprev_ = pd.DataFrame(self.data[column].value_counts()).reset_index().set_index(column)
            self.data[column] = s_
            t_ += 'New data shape: ' + str(self.data.shape) + '<br>'
            dfpost_ = pd.DataFrame(self.data[column].value_counts()).reset_index().set_index(column)
            df_ = dfprev_.join(dfpost_, lsuffix="_prev", rsuffix="_post")
            t_ += df_.to_html()
            # Update analysis
            self.update_analysis(column, "oe", 1)
        
        display(HTML(t_))
        
        # Add to report
        self.update_report(t_)
    
    def to_binary(self, column, mapping):
        '''
        to_binary
        Function that transforms column to binary encoding and updates the X dataframe
        Parameters
        column: name of the variable to transform
        mapping: the value map (list of tupples)
        
        Return
        Updated dataframe
        '''
        
        t_ = '<h3>--> ' + column + ' BINARY Results</h3><br>'
        
        # map formating (we need list dictionaries)
        dict_map = {}
        for i in range(len(mapping)):
            dict_map[mapping[i][0]] = mapping[i][1]
        
        s_ = self.data[column].map(dict_map) # we store it in a temp variable to check if everything went well before proceeding
        if (len(s_[s_.isna()])>0): # that means that either the mapping was not correct, or that there are still null values
            t_ += 'WARNING: The transformed column would still has null values. Therefore, not able to encode the variable.<br>'
            # Update analysis
            self.update_analysis(column, "binary", -1)
        else:
            t_ += 'Original Data shape: ' + str(self.data.shape) + "<br>"
            dfprev_ = pd.DataFrame(self.data[column].value_counts()).reset_index().set_index(column)
            self.data[column] = s_
            t_ += 'New X shape: ' + str(self.data.shape) + '<br>'
            dfpost_ = pd.DataFrame(self.data[column].value_counts()).reset_index().set_index(column)
            df_ = dfprev_.join(dfpost_, lsuffix="_prev", rsuffix="_post")
            t_ += df_.to_html()
            # Update analysis
            self.update_analysis(column, "binary", 1)
        
        display(HTML(t_))
        
        # Add to report
        self.update_report(t_)
    
    def to_pivot(self, column, value_column, new_name):
        '''
        to_pivot
        Function that implements a pivot as an ohe
        Parameters
        column: name of the column that will be used for ohe columns
        value_column: name of the column that will have the values
        new_name: if a new name for the column must be given
        
        Return
        Updated dataframe
        '''
        
        df_ = self.data[[column, value_column]]
        index_name = df_.index.name
        df_.reset_index(drop=False, inplace=True)
        df_columns = [index_name, column, value_column] #Id is the default name that reset_index gives when the index has no name
        pv_ = df_.pivot(index=[index_name], columns=[column], values=[value_column]).add_prefix(new_name + "_") # the pivot is like an ohe but with values instead of 1/0
        pv_.columns = pv_.columns.get_level_values(1)
        pv_.fillna(0, inplace=True)
        
        t_ = f'<h3>--> {column} PIVOT Results (column: {column}, value_column: {value_column})</h3><br>'
        t_ += f'PIVOT generated a total of {str(len(pv_.columns))} columns.<br>'
        t_ += 'Original Data shape: ' + str(self.data.shape) + "<br>"
        
        # join the dataframe with the pivot
        self.data = pd.concat([self.data.drop([column, value_column], axis=1), pv_], axis=1)
        
        t_+= f'New Data shape: {str(self.data.shape)} ({str(len(pv_.columns))} PIVOT -2 dropped)<br>'
        display(HTML(t_))

        # Update analysis
        self.update_analysis(column, "pivot", 1)
        
        # Add to report
        self.update_report(t_)
        
    def to_cut(self, column, bins, values, labels=None, right=False, include_lowest=True, drop_original=True):
      '''
      to_cut
      Transforms a numeric column into value bins depending on ranges of numbers.

      Parameters
      column: name of the column to transform
      bins: array with the limits (left and right) of the bins 
      values: array of values that will be assigned to each of the bins (this array has -1 dimensions compared to bins array)
      labels: array (same length as values) with the descriptive values of the bins. If None, values will be used
      right: False if we do not want to include the right value of each range
      include_lowest: True if we want to include the first number of the bins array in the range.
      drop_original: True if we want to drop the original column

      Returns
      updated dataset
      '''

      if labels is None: labels = values # if no description is given for the plot, then use values

      t_ = f'<h3>--> {column} CUT Results</h3><br>'

      s_ = self.data[column]
      s_ = pd.cut(
          s_,
          bins,
          right = right, 
          include_lowest = include_lowest, 
          labels = values
      )
      new_column_name = str(column) + str('' if drop_original else '_cut') # if we do not drop the original, we must differentiate both columns
      s_.rename(new_column_name, inplace=True) # This column will have a new name

      t_ += f'Original Data shape: {str(self.data.shape)}<br>'
      # join the dataframe with the pivot
      self.data = pd.concat([self.data, s_], axis=1)
      if drop_original: 
        self.data.drop(column, axis=1, inplace=True)
      t_ += f'New X shape: {str(self.data.shape)}<br>'

      display(HTML(t_))

      # We plot the results to make sure we have done it correctly
      fig, ax = plt.subplots(1, 1, figsize=(10,4))
      s_ = self.data[new_column_name].value_counts().sort_index()

      ax = sns.barplot(x = s_.index, y = s_, ax = ax)
      plt.xticks(np.arange(0,len(labels)), labels=labels, rotation=45)
      plt.grid(axis='y')
      plt.title(f"Number of {column} per bin - new column {column + '_cut'}")

      rects = ax.patches
      labels = [str(np.round(100*i/s_.sum(), decimals=2))+"%" for i in s_]
      for rect, label in zip(rects, labels):
          height = rect.get_height()
          ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

      plt.show()
        
      # Update analysis
      self.update_analysis(column, "cut", 1)
      # Add to report
      self.update_report(t_)

    # TRANSFORMATIONS
    def to_sqrt(self, column, drop_original=True):
        '''
        to_sqrt
        Function that transforms column to sqrt and updates X dataframe
        Parameters
        column: name of the variable to transform
        
        Return
        Updated dataframe
        '''
        has_neg = self.data[column].min() < 0
        new_column_name = str(column) + str('' if drop_original else '_sqrt') # if we do not drop the original, we must differentiate both columns
        
        t_ = '<h3>--> ' + column + ' SQRT Results</h3><br>'
        if has_neg:
            t_ += 'WARNING: The column has negative values. Therefore, not able to sqrt transform the variable.<br>'
            
            # Update analysis
            self.update_analysis(column, "sqrt", -1)
        else:
            t_ += 'Original Data shape: ' + str(self.data.shape) + ' - mean: ' + str(round(self.data[column].mean(),2)) + ' - skew: ' + str(round(self.data[column].skew(),2)) + '<br>'
            s_ = self.data[column]**0.5
            s_.rename(new_column_name, inplace=True)
            if drop_original: # we can overwrite the column
              self.data[column] = s_
            else:
              self.data = pd.concat([self.data, s_], axis=1) # we must concat the new column
            t_ += 'New Data shape: ' + str(self.data.shape) + ' - mean: ' + str(round(self.data[new_column_name].mean(),2)) + ' - skew: ' + str(round(self.data[new_column_name].skew(),2)) + '<br>'
            
            # Update analysis
            self.update_analysis(column, "sqrt", 1)
        
        display(HTML(t_))
        
        # Add to report
        self.update_report(t_)
    
    def to_inv(self, column, drop_original=True):
        '''
        to_inv
        Function that transforms column to 1/column and updates X dataframe
        Parameters
        column: name of the variable to transform
        
        Return
        Updated dataframe
        '''
        has_zeros = len(self.data[self.data[column]==0]) > 0
        new_column_name = str(column) + str('' if drop_original else '_inv') # if we do not drop the original, we must differentiate both columns

        t_ = '<h3>--> ' + column + ' INV Results</h3><br>'
        if has_zeros:
            t_ += 'WARNING: The column has negative values. Therefore, not able to inverse transform the variable.<br>'
            
            # Update analysis
            self.update_analysis(column, "inv", -1)
        else:
            t_ += 'Original Data shape: ' + str(self.data.shape) + ' - mean: ' + str(round(self.data[column].mean(),2)) + ' - skew: ' + str(round(self.data[column].skew(),2)) + '<br>'
            s_ = 1/self.data[column]
            s_.rename(new_column_name, inplace=True)
            if drop_original: # we can overwrite the column
              self.data[column] = s_
            else:
              self.data = pd.concat([self.data, s_], axis=1) # we must concat the new column
            t_ += 'New Data shape: ' + str(self.data.shape) + ' - mean: ' + str(round(self.data[new_column_name].mean(),2)) + ' - skew: ' + str(round(self.data[new_column_name].skew(),2)) + '<br>'
            
            # Update analysis
            self.update_analysis(column, "inv", 1)
        
        display(HTML(t_))
        
        # Add to report
        self.update_report(t_)
        
    def to_log(self, column, drop_original=True):
        '''
        to_log
        Function that transforms column to log and updates X dataframe
        Parameters
        column: name of the variable to transform
        
        Return
        Updated dataframe
        '''
        has_neg = self.data[column].min() < 0
        has_zeros = len(self.data[self.data[column]==0]) > 0
        new_column_name = str(column) + str('' if drop_original else '_log') # if we do not drop the original, we must differentiate both columns
        
        t_ = '<h3>--> ' + column + ' LOG Results</h3><br>'
        
        if has_neg:
            t_ += 'WARNING: The column has negative values. Therefore, not able to log transform the variable.<br>'
            
            # Update analysis
            self.update_analysis(column, "log", -1)
        else:
            t_ += 'Original Data shape: ' + str(self.data.shape) + ' - mean: ' + str(round(self.data[column].mean(),2)) + ' - skew: ' + str(round(self.data[column].skew(),2)) + '<br>'
            # Check if it has zeros to apply the right transformation
            if not has_zeros:
                s_ = np.log(self.data[column])
            elif has_zeros:
                s_ = np.log(self.data[column] + 1)
                t_ += '--> note: variable has zeros. Log(s+1) has been applied ot avoid infinity.<br>'
            
            # Manage the column name
            s_.rename(new_column_name, inplace=True)
            
            if drop_original: # we can overwrite the column
              self.data[column] = s_
            else:
              self.data = pd.concat([self.data, s_], axis=1) # we must concat the new column
            t_ += 'New Data shape: ' + str(self.data.shape) + ' - mean: ' + str(round(self.data[new_column_name].mean(),2)) + ' - skew: ' + str(round(self.data[new_column_name].skew(),2)) + '<br>'
            
            # Update analysis
            self.update_analysis(column, "log", 1)
        
        display(HTML(t_))
        
        # Add to report
        self.update_report(t_)
    
    def to_ratio(self, column, base, rounded=2, drop_original=True):
        '''
        to_ratio
        Function that obtains a ratio of the column values from a base
        Parameters
        column: name of the variable to transform
        base: column that is used as base for the ratio
        rounded = number of decimals
        
        Return
        Updated dataframe
        '''
        
        new_column_name = str(column) + str('' if drop_original else '_rat') # if we do not drop the original, we must differentiate both columns
        
        t_ = f'<h3>--> {column} RATIO Results</h3><br>'
        t_ += 'Original Data shape: ' + str(self.data.shape) + '<br>'
        
        # Generate new column
        s_ = round(self.data[column]/self.data[base],2)

        # Manage the column name
        s_.rename(new_column_name, inplace=True)
        
        # Manage drop original
        if drop_original: # we can overwrite the column
          self.data[column] = s_
        else:
          self.data = pd.concat([self.data, s_], axis=1) # we must concat the new column

        t_ += 'New Data shape: ' + str(self.data.shape) + '<br>'
        t_ += f'Column has been transformed to ratio base: {base}<br>'
        
        display(HTML(t_))
        
        # Update analysis
        self.update_analysis(column, "ratio", 1)
        
        # Add to report
        self.update_report(t_)
        
    # DATA DROP
    def drop_outliers(self, column, threshold):
        '''
        drop_outliers
        Function that drops outliers that go beyond a threshold from both X and y datasets
        Parameters
        column: name of the variable to transform
        threshold: to be determined as an outlier
        
        Return
        Updated dataframe
        
        IMPORTANT: outliers of the test set MUST NOT BE REMOVED.
        '''
        t_ = '<h3>--> ' + column + ' DROP OUTLIERS Results</h3><br>'
        t_ += 'Original Data shape: ' + str(self.data.shape) + ' - mean: ' + str(round(self.data[column].mean(),2)) + ' - skew: ' + str(round(self.data[column].skew(),2)) + '<br>'
        index_to_drop = eval("self.data[self.data[column]" + threshold + "].index.to_list()")
        index_to_drop = [e for e in index_to_drop if e not in self.index_submission]
        self.data.drop(index_to_drop, axis=0, inplace=True)
        t_ += 'New Data shape: ' + str(self.data.shape) + ' - mean: ' + str(round(self.data[column].mean(),2)) + ' - skew: ' + str(round(self.data[column].skew(),2)) + '<br>'
        display(HTML(t_))
        
        # Update analysis
        self.update_analysis(column, "outliers", 1)
        
        # Add to report
        self.update_report(t_)
    
    def drop(self, column):
        '''
        drop
        Function that drops a variable
        Parameters
        column: name of the variable to drop
        
        Return
        Updated dataframe
        '''
        t_ = '<h3>--> ' + column + ' DROP Results</h3><br>'
        t_ += 'Original Data shape: ' + str(self.data.shape) + '<br>'
        self.data = self.data.drop(column, axis=1)
        t_ += 'New Data shape: ' + str(self.data.shape) + '<br>'
        display(HTML(t_))
        
        # Update analysis
        self.update_analysis(column, "drop", 1)
        
        # Add to report
        self.update_report(t_)
    
    def drop_id(self, column, ids):
        '''
        drop_id
        Function that drops records
        Parameters
        column: just for reporting purposes
        ids: string or list of id label
        
        Return
        Updated dataframe
        '''
        t_ = '<h3>--> ' + column + ' DROP_ID Results</h3><br>'
        t_ += 'Original Data shape: ' + str(self.data.shape) + '<br>'
        self.data = self.data.drop(ids, axis=0)
        t_ += 'New Data shape: ' + str(self.data.shape) + '<br>'
        display(HTML(t_))
        
        # Update analysis
        self.update_analysis(column, "drop_id", 1)
        
        # Add to report
        self.update_report(t_)
    
    # VALUES CORRECTION
    def replace(self, column, mapping):
        '''
        replace
        Function that replaces some misspelled strings
        Parameters
        column: name of the variable to drop
        mapping: dictionary of pairs ("find":"replace")
        
        Return
        Updated dataframe
        '''
        t_ = f'<h3>--> {column} REPLACE Results</h3><br>'
        self.data[column].replace(mapping, inplace=True)
        t_ = f'The following changes have been successfully performed {str(mapping)}.'
        
        # Update analysis
        self.update_analysis(column, "replace", 1)
        
        # Add to report
        self.update_report(t_)        
        
    def fillna(self, column, value):
        '''
        fillna
        Function that fills the null values
        Parameters
        column: name of the variable to drop
        value: as parameter of pd.fillna
        
        Return
        Updated dataframe
        '''
        t_ = '<h3>--> ' + column + ' FILLNA Results</h3><br>'
        t_ += 'Original Data shape: ' + str(self.data.shape) + ' - num. nulls: ' + str(self.data[column].isna().sum()) + '<br>'
        self.data[column].fillna(value, inplace=True)
        t_ += 'New Data shape: ' + str(self.data.shape) + ' - num. nulls: ' + str(self.data[column].isna().sum()) + '<br>'
        display(HTML(t_))
        
        # Update analysis
        self.update_analysis(column, "fillna", 1)
        
        # Add to report
        self.update_report(t_)
        
    # FINAL CHECK
    def final_check(self):
      '''
      final_check
      Method that checks for overall consistency before proceeding to the model

      Parameters
      None

      Returns
      Display results
      '''
      ok = True
      # check if all columns have been processed
      non_processed = self.analysis.loc[self.analysis["processed"]!=1]["processed"]
      if (len(non_processed)>0):
        display(HTML('<span style="font-weight:bold;color:#FF0000">There are still non processed columns</span>'))
        display(non_processed)
        ok = False
      # check for nulls
      if (self.data.isna().sum().sum()>0):
        display(HTML('<span style="font-weight:bold;color:#FF0000">There are still null values</span>'))
        display(self.data.columns[self.data.isna().any()].to_list())
        ok = False
      
      if ok: display(HTML('<span style="font-weight:bold;">Everything is OK! Go and have fun with the model!</span>'))
    
    # ANALYSIS
    def update_analysis(self, column, action, result):
        
        self.analysis.loc[(self.analysis["column"]==column) & (self.analysis["key"]==action),"processed"] = result
    
    def update_report(self, t_):
        
        self.html_report += "<div style='margin-top: 15px'>" + t_ + "<div>"
        
    def show_report(self):
        
        display(HTML(self.html_report))
        
    # SPLIT STRATEGY
    def split_dev_test(self):
        '''
        split_dev_test
        Function that splits X and y into data and submission if the instance was propperly informed
        
        parameters:
        none
        
        return X, y, X_submission
        '''
        if (len(self.index_submission) > 0):
            X = self.data.loc[~self.data.index.isin(self.index_submission)].drop(self.target, axis=1)
            y = self.data.loc[X.index][self.target]
            X_submission = self.data.loc[self.data.index.isin(self.index_submission)].drop(self.target, axis=1)
            # there is no y_test, as it is part of a submission to kaggle

            return X, y, X_submission
        else:
            pass
    
    def get_submission(self):
        '''
        get_submission
        Method that returns the submission dataset with the same transformation as the development dataset
        
        '''
        return self.data.loc[index_submission,:]
    
    # IMPORT
    @staticmethod
    def from_pickle(import_path, file_name):
      '''
      from_pickle
      static method that returns a EDA instance from the previous EDA process

      Parameters
      export_path: path where the pickle is stored
      file_name: name of the pickle, without pkl extension
      '''
      with open(import_path + '/' + file_name + '.pkl', 'rb') as fid:
        return pickle.load(fid)
    
    # EXPORT
    @staticmethod
    def to_pickle(instance, export_path, file_name):
      '''
      to_pickle
      static method that creates a pickle of the EDA instance

      Parameters
      export_path: path where the pickle will be stored
      file_name: name of the pickle, without pkl extension.
      '''
      with open(export_path + '/' + file_name + '.pkl', 'wb') as fid:
        pickle.dump(instance, fid)
      return HTML(f"<i>{file_name}.pkl</i> has been created in <i>{export_path}</i>")
