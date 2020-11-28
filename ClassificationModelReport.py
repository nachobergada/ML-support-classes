import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

class ClassificationModelReport:

    '''
    ClassificationModelReport
    This class is used to both compare train and test metrics of a model, and compare the different metrics
    of all the created models.
    '''

    model = None
    data = [
        # 0 - training
        {
            'X': None, 
            'y': None,
            'y_pred': None, 
        },
        # 1- validation
        {
            'X': None, 
            'y': None,
            'y_pred': None, 
        },
    ]
    metrics = [] # list storing dictionaries of metrics for train [0] and val [1]
    summary = {}

    def init_report(self, model, X_train, y_train, X_val, Y_val):  
        """Stores data and calculates the different metrics, stores it in the local vars and show them on screen"""

        # train
        self.model = model
        self.data[0]["X"] = X_train
        self.data[0]["y"] = y_train
        if "FaissKNeighbors" in str(type(model)):
            self.data[0]["y_pred"] = pd.Series(model.predict(np.ascontiguousarray(X_train.values)), index=X_train.index, name="Class_predicted")
            self.data[0]["y_pred_proba"] = None
        elif "tensorflow.python.keras" in str(type(model)): # prediction returns an array of arrays. We have to convert it. Also, predict returns predict_proba, and predict_classes returns predict
            self.data[0]["y_pred"] = pd.Series([item[0] for item in model.predict_classes(X_train)], index=X_train.index, name="Class_predicted")
            self.data[0]["y_pred_proba"] = None # pd.Series([item[0] for item in model.predict(X_train)], index=X_train.index, name="Class_predicted")
        else:
            self.data[0]["y_pred"] = pd.Series(model.predict(X_train), index=X_train.index, name="Class_predicted")
            if (model.predict_proba(X_train) is not None):
                self.data[0]["y_pred_proba"] = pd.Series(pd.DataFrame(model.predict_proba(X_train), index=X_train.index).iloc[:,-1], name="Class_predicted")
            else:
                self.data[0]["y_pred_proba"] = None

        #test
        self.data[1]["X"] = X_val
        self.data[1]["y"] = y_val
        if "FaissKNeighbors" in str(type(model)):
            self.data[1]["y_pred"] = pd.Series(model.predict(np.ascontiguousarray(X_val.values)), index=X_val.index, name="Class_predicted")
            self.data[1]["y_pred_proba"] = None
        elif "tensorflow.python.keras" in str(type(model)): # prediction returns an array of arrays. We have to convert it. Also, predict returns predict_proba, and predict_classes returns predict
            self.data[1]["y_pred"] = pd.Series([item[0] for item in model.predict_classes(X_val)], index=X_val.index, name="Class_predicted")
            self.data[1]["y_pred_proba"] = None # pd.Series([item[0] for item in model.predict(X_val)], index=X_val.index, name="Class_predicted")
        else:
            self.data[1]["y_pred"] = pd.Series(model.predict(X_val), index=X_val.index, name="Class_predicted")
            if (model.predict_proba(X_test) is not None):
                self.data[1]["y_pred_proba"] = pd.Series(pd.DataFrame(model.predict_proba(X_val), index=X_val.index).iloc[:,-1], name="Class_predicted")
            else:
                self.data[1]["y_pred_proba"] = None

        # Calculate the metrics
        self.metrics = [self.empty_metrics(), self.empty_metrics()] # train and validation
        for i in range(0, len(self.metrics)):
            self.metrics[i]["accuracy"] = round(accuracy_score(self.data[i]["y"], self.data[i]["y_pred"]),4)
            self.metrics[i]["precision"] = round(precision_score(self.data[i]["y"], self.data[i]["y_pred"]),4)
            self.metrics[i]["recall"] = round(recall_score(self.data[i]["y"], self.data[i]["y_pred"]),4)
            self.metrics[i]["f1"] = round(f1_score(self.data[i]["y"], self.data[i]["y_pred"]),4)
            self.metrics[i]["cm"] = confusion_matrix(self.data[i]["y"], self.data[i]["y_pred"])
            if self.data[i]["y_pred_proba"] is not None:
                self.metrics[i]["auc"] = round(roc_auc_score(self.data[i]["y"], self.data[i]["y_pred_proba"]),4)
                self.metrics[i]["tpr"], self.metrics[i]["fpr"], thresholds = roc_curve(self.data[i]["y"], self.data[i]["y_pred_proba"])
  
    def show_report(self):
        '''Shows the report for this model'''

        display(pd.DataFrame(self.metrics, index=["train","validation"])[["accuracy","precision","recall","f1","auc"]])

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        axs[0].title.set_text("Train Confusion Matrix")
        sns.heatmap(self.metrics[0]["cm"], annot=True, annot_kws={"size":9}, fmt='g', ax=axs[0]) # fmt = format of annot, in that case, plain notation (g)
        axs[1].title.set_text("Validation Confusion Matrix")
        sns.heatmap(self.metrics[1]["cm"], annot=True, annot_kws={"size":9}, fmt='g', ax=axs[1]) # fmt = format of annot, in that case, plain notation (g)
        plt.show()

        if (self.data[0]["y_pred_proba"] is not None) & (self.data[1]["y_pred_proba"] is not None):
            fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
            axs[0].title.set_text("Train ROC Curve")
            plot_roc_curve(self.model, self.data[0]["X"], self.data[0]["y"], ax=axs[0])
            axs[1].title.set_text("Validation ROC Curve")
            plot_roc_curve(self.model, self.data[1]["X"], self.data[1]["y"], ax=axs[1])

    def plot_roc_curve_custom(self, y = None, y_pred_prob = None): # deprecated
        """Plots the roc curve of the trained model"""
        if y is None: y, y_pred_prob = self.y, self.y_pred_prob

        fpr, tpr, threshold = roc_curve(y, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.margins(0.1)
        plt.show()
  
    def show_cv_results(self, cv_results, metric):
        '''
        Compares the results of a GridSearchCV

        parameters:
        - cv_results -> the cv_results_ object stored in the GridSearchCV instance (GridSearchCV.cv_results_)
        - metric -> the name of the metric to compare
        '''
        # The detailed results of the Cross-Validation process are stored the follwing way
        # cv_results_ is a dictionary
        # Metric results are stored for each Kfold split in the following keys: 
        # - split*number_of_fold*_test_*metric*
        # - split*number_of_fold*_train_*metric*
        # For each kfold split there is an array with length j. The length j corresponds to all combinations of the parameters fed to GridSearchCV
        # The different combinations of hypermparameters are stored in this array, from 0 to j-1

        print ("********************************")
        print ("METRIC: {}".format(metric.upper()))
        print ("********************************")

        fig, axes = plt.subplots(1, len(cv_results["params"]), sharey = True, figsize=(5*len(cv_results["params"]), 5))
        fig.subplots_adjust(hspace=0.3)
        # plt.xticks(np.arange(gscv.n_splits_), np.arange(1,gscv.n_splits_+1))

        # We gather the best model that GridSearchCV has calculated
        # e.g. 'rank_test_accuracy': array([2, 1, 3], dtype=int32)
        iModel_best = cv_results["rank_test_" + metric].tolist().index(1) # 1 is the first position in ranking, the best model in test dataset

        # We loop through the hyperparameters combination
        models_scores = pd.DataFrame(columns=["model","split","train","test"])
        for iModel, model in enumerate(cv_results["params"]): # params is an array of dictionaries
            split_metric_scores = None # this list will store the train and test metric scores for each of the splits
            has_metrics = True
            for iSplit in range(0, gscv.n_splits_):
                # if we find that there are no values, we must inform the result as nan
                if np.isnan(cv_results["split" + str(iSplit) + "_train_" + metric][iModel]): has_metrics = False
                if has_metrics:
                    # store the metric
                    split_metric_scores = pd.DataFrame([[
                        iModel, 
                        int(iSplit),
                        cv_results["split" + str(iSplit) + "_train_" + metric][iModel], 
                        cv_results["split" + str(iSplit) + "_test_" + metric][iModel]
                    ]], columns=models_scores.columns)
                else:
                    # store nan
                    split_metric_scores = pd.DataFrame([[
                        iModel, 
                        int(iSplit),
                        np.nan, 
                        np.nan
                    ]], columns=models_scores.columns)
                models_scores = models_scores.append(split_metric_scores, ignore_index=True)
            # and we plot the values for train and test for the different splits in the model
            axes[iModel].title.set_text("MODEL {}".format(iModel))
            if iModel == iModel_best: axes[iModel].set_facecolor('wheat') #highlight the selected model by GridSearchCV
            if has_metrics: sns.lineplot(data=models_scores[models_scores["model"]==iModel].iloc[:,1:].set_index("split"), ax=axes[iModel])
        plt.show()

        # And one last plot to compare the models
        fig, axes = plt.subplots(1, 2, sharey = True, sharex = True, figsize=(10,5))
        fig.subplots_adjust(wspace=0.3)
        axes[0].title.set_text("MODEL COMPARISON - Train")
        sns.boxplot(data=models_scores, x="model", y="train", ax=axes[0], showmeans=True)
        axes[1].title.set_text("MODEL COMPARISON - Test")
        sns.boxplot(data=models_scores, x="model", y="test", ax=axes[1], showmeans=True)
        plt.show()

    def show_nnhistory_results(self, nnhistory, metrics):
        '''
        Plots the results returned by the history object of keras tensor flow model.

        parameters:
        nnhistory -> keras.fit output
        metrics -> string of the metrics used for validation

        returns:
        plots entropy loss per epoch evolution and the corresponding metric, both for train and test datasets
        '''
        
        fig = plt.figure(figsize=(6,5))
        plt.title('Binary Cross Entropy Loss')
        plt.plot(nnhistory.history['loss'], color='blue', label='train')
        plt.plot(nnhistory.history['val_loss'], color='orange', label='test')
        legend = plt.legend(loc='upper left', shadow=True, fontsize='large')
        plt.show()
        
        if 1==2: # currently disabled. TODO: Metrics do not behave correctly. I need to check why.
            if "list" in str(type(metrics)):
                fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics)*6, 5), sharex = True)
                for i, m in enumerate(metrics):
                    axes[i].set_title('Classification ' + m.upper())
                    axes[i].plot(nnhistory.history[m], color='blue', label='train')
                    axes[i].plot(nnhistory.history['val_' + m], color='orange', label='test')
                    axes[i].legend(loc='upper left', shadow=True, fontsize='large')
                plt.show()
            else:
                plt.title('Classification ' + metrics.upper())
                plt.plot(nnhistory.history[metrics], color='blue', label='train')
                plt.plot(nnhistory.history['val_' + metrics], color='orange', label='test')
                legend = plt.legend(loc='upper left', shadow=True, fontsize='large')
                plt.show()
            
  
    def get_model_params(self, cv_results, model_key):
        '''
        Returns the params of the model using the key in the GridSearchCV cv_results_ object.

        parameters:
        cv_results -> GridSearchCV.cv_results_
        model_key -> the key of the selected model on cv_results["params"]

        returns:
        dictionary with the model parameters
        '''
        return {model_key:cv_results["params"][model_key]}

    def store_report(self, key = "", name = ""):
        """Add the VALIDATION metrics of the current model to the dictionary of compared models"""

        self.summary[key] = {
            "name" : name,
            "model" : self.model, 
            "y": self.data[1]["y"], 
            "y_pred": self.data[1]["y_pred"], 
            "y_pred_proba": self.data[1]["y_pred_proba"], 
            "accuracy" : self.metrics[1]["accuracy"], 
            "precision": self.metrics[1]["precision"], 
            "recall": self.metrics[1]["recall"], 
            "f1": self.metrics[1]["f1"], 
            "auc" : self.metrics[1]["auc"], 
            "tpr": self.metrics[1]["tpr"], 
            "fpr": self.metrics[1]["fpr"], 
            "cm": self.metrics[1]["cm"], 
        }

    def compare_models(self, order_by="accuracy"):
        """Plots the metrics of the different models that have been stored"""
        df_summary = pd.DataFrame(self.summary).T.reset_index()
        return df_summary[["name","accuracy","recall","precision","f1","auc"]].sort_values(by=order_by, ascending=False)

    def empty_metrics(self):
        return {
          'accuracy': None, 
          'precision': None, 
          'recall': None, 
          'f1': None, 
          'cm': None, 
          'auc': None, 
          'fpr': None, 
          'tpr': None, 
        }
