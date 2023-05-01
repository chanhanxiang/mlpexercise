import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, RidgeClassifierCV 
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def prepare_data(df_new):
    X = df_new.drop(columns="rainpred")
    y = df_new["rainpred"].values

    # Apply SMOTE and undersampling
    over = SMOTE(sampling_strategy=0.65)
    under = RandomUnderSampler(sampling_strategy=1.0)
    X, y = over.fit_resample(X, y)
    X, y = under.fit_resample(X, y)
    
    # Scale the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True)

    return X_train, X_test, y_train, y_test

def define_models():
    models = [make_pipeline(StandardScaler(), LogisticRegression()),
            make_pipeline(StandardScaler(), Perceptron()), 
            make_pipeline(StandardScaler(), SGDClassifier(loss='log', n_iter_no_change=100)), 
            make_pipeline(StandardScaler(), RidgeClassifierCV()), 
            make_pipeline(StandardScaler(), GaussianNB()), 
            make_pipeline(StandardScaler(), BernoulliNB()),
            make_pipeline(StandardScaler(), DecisionTreeClassifier()),
            make_pipeline(StandardScaler(), AdaBoostClassifier(n_estimators = 100)),
            make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators = 100)),  
            make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 100)), 
            make_pipeline(StandardScaler(), BaggingClassifier(n_estimators = 100))]

    return models, 


def run_models(df_new):
    models = define_models()
    model_names = ['Logistic Regression', 'Perceptron', 'Stochastic Gradient Descent', 'Ridge Classifier', \
        'GaussianNB', 'BernoulliNB', 'Decision Tree Classifier', \
        'AdaBoostClassifier', 'GradientBoostingClassifier', 'RandomForestClassifier', 'BaggingClassifier']
    X_train, X_test, y_train, y_test = prepare_data(df_new)

    accuracy_test = []
    precision_test = []
    recall_test = []
    f1_test = []
    roc_auc_test = []

    for clf, name in zip(models, model_names):
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        accuracy_test.append(accuracy_score(pred, y_test))
        precision_test.append(precision_score(pred, y_test))
        recall_test.append(recall_score(pred, y_test))
        f1_test.append(f1_score(pred, y_test))
        roc_auc_test.append(roc_auc_score(pred, y_test))


    df_stat_test = pd.DataFrame({'Algorithm' : model_names, 
                        'Accuracy' : accuracy_test, 
                        'Precision': precision_test, 
                        'Recall': recall_test, 
                        'F1' : f1_test, 
                        'ROC AUC': roc_auc_test})

    print(df_stat_test)

