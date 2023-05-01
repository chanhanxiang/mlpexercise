import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve

class metricplot(object):
    def __init__(self, true, predicted):
        self.true = true
        self.predicted = predicted

    def evaluate(true, predicted):
        '''Print evaluation metrics for Accuracy, Preision, Recall, F1 and ROC AUC'''  
        accuracy = accuracy_score(true, predicted)
        precision = precision_score(true, predicted)
        recall = recall_score(true, predicted)
        f1 = f1_score(true, predicted)
        roc_auc = roc_auc_score(true, predicted)
        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F1:', f1)
        print('ROC AUC:', roc_auc)
        print('__________________________________')

class dataplot(object):
    def __init__(self, X_test, y_test, model):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test


    def curves(X_test, y_test, model): 
        '''Generate graphs for precison and recall'''
        y_pred_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

        # For precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

        # ax1 for ROC, ax2 for precision-recall

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (18, 5))
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.plot(fpr, tpr)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.plot()

        ax2.plot(recall, precision, marker='.', label='Model')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision Recall Curve')
        ax2.plot()

    def coefplot(X_test, model):
        '''Plot to generate rank features with the highest sum of weighted coefficient'''
        importances = model.coef_
        importances = importances[0]
        importances = abs(importances)
        features = X_test.columns.values
        indices = np.argsort(importances)

        plt.figure(figsize=(18, 22))
        ax = plt.subplot()
        plt.barh(range(len(indices)), importances[indices], color='r', align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(features[indices])
        plt.xlabel('Relative Importance')
        plt.title('Coefficient plot', fontsize=20)
        plt.show()


    def feaplot(X_test, model):
        '''Feature importance by Gini impurity'''
        importances = model.feature_importances_
        features = X_test.columns
        indices = np.argsort(importances)

        feature_scores = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=True)

        plt.figure(figsize=(18, 22))
        ax = feature_scores.plot.barh(y ='Score', color='r', align='center')
        plt.xlabel('Relative Importance')
        plt.tick_params(axis='x', labelsize=15)
        #plt.tick_params(axis='y', labelsize=15)
        plt.title('Feature importance plot', size=15)

        plt.show()

    def permutation(X_test, model):
        '''Generation and visualisation of Permutation Importance Plot'''
        importances = model.importances_mean
        importances = abs(importances)
        features = X_test.columns
        indices = np.argsort(importances)

        plt.figure(figsize=(18, 22))
        ax = plt.subplot()
        plt.barh(range(len(indices)), importances[indices], color='r', align='center')
        ax.set_yticks(range(len(indices)), features[indices])
        plt.xlabel('Relative Importance')
        plt.title('Permutation importance plot', fontsize=20)
        plt.show()