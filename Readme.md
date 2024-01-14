# Readme

Name: Chan Han Xiang

Email: chxuniversity@yahoo.com

### Folder

```bash

  ├── .github
  ├── src
  │   └── mlp_preprocess.py
  │   └── mlp_model.py
  │   └── mlp_main.py
  │   └── db_conn.py
  │   └── preprocess.py
  │   └── eda_vis.py
  │   └── train_test_split.py
  │   └── modelview.py
  ├── README.md
  ├── eda.ipynb
  ├── requirements.txt
  └── run.sh
  └── run_local.sh 

```

*Note: run_local.sh is a supplementary file and not to be regarded as principal item in submission package.*

### Pipeline execution

Run pipeline

1. Run ./run.sh on Gitbash. If error encountered, try entering chmod u+x run.sh instead.

Run with virtualenv on local PC

If run.sh does not lead to launch of Jupyter Notebook, suggest running the shell script using the following approach. Additionally, do note that all files in submission needs to be downloaded to local PC in order to work:

1. Open Git Bash, enter cd and followed by the directory pathway in the same line. Example:
cd '.../path/to/directory'
2. Create local environment (virtualenv) using mkdir (example name: localaiap)
3. pip install jupyterlab and virtualenv 
4. Activate python on virtuanenv
5. Install requirements.txt
6. Activate local environment and install ipykernel
7. Launch jupyter-notebook on local environment

Script for run with virtualenv maybe found at run_local.sh


##### Environment specifications

PC: Windows 10 Enterprise
Git Bash: 2.31.1
Python 3.8.8
Visual Studio Code 1.63.2 with Jupyter notebook


### Notebook sections

1. Import libraries
2. Data preprocessing
3. Exploratory Data Analysis (EDA)
4. Additional data preprocessing
5. Train test split
6. Logistic regression
7. Adaboost Classifier
8. k-Nearest Neighbour
9. Conclusion

## Key findings from EDA

- Target attribute: "Rainfall" is continuous variable. Will be converted to categorical variable, called "rainpred" which classify a day as non-rainy if rainfall <= 1.0mm on given day and a day as rainy if rainfall is >1.0mm. To fulfill business criteria set by fishing company.
- 19 other attributes, of which 8 are categorical and 11 are continuous
- Among categorical variables, "WindGustDir", "WindDir9am", "WindDir3pm" have 18 categories. Performing one-hot encoding would exponentially increase the number of categories in the dataset
- Numerical variables have outliers <10%. "Sunshine", "Humidity3pm", "WindSpeed3pm", "WindSpeed9am" have outliers between 5-10%
- Spearman correlation plot lists out variables with positive and negative correlation respectively
- After generating "rainpred", there is significant class imbalance between days whereby rainfall is  <= 1.0mm and days whereby rainfall is >1.0mm. Oversampling and undersampling shall be applied to rectify this issue.

## Preprocessing

| Item | Preprocessing  | Description |
| :--: | :---------:    | :---------: |
| 1    |  datesplit     | First step: Split the date column into 3 seperate columns of year, month and date</br> Second step: Convert the datatypes from object to integer</br> Data derived from "Date" column. |
| 2    |  rainpred      | Create new column, "rainpred" by classifying Rainfall values that is >1.0mm as 1 (positive),</br> and Rainfall values that are <= 1.0mm as 0 (negative). Data derived from "Rainfall" column. |
| 3    |  absolute      | Negative values are observed in Sunshine column, most likely errorneously entered.</br> To convert them as positive value. |
| 4    |  camelcase     | To fix camelcase characters observed in Pressure9am and Pressure3pm columns. |
| 5    |  replacena     | Fill NA with "No" in "RainToday" column |
| 6    |  dropna        | Drop any residual NA in the rows |
| 7    |  objmap        | Replace No with 0 and Yes with 1 for "RainToday" and "RainTomorrow" columns |
| 8    |  onehotencoding| Apply one-hot encoding to multi-class categorical variables, and re-concatenating them after performing one-hot encoding. |
| 9    |  dropcolumn    | Drop the processed columns from the re-constituted main dataframe. |



## Choice of models

Dataset is split into 60% for training, 20% for validation, 20% for test. Eleven classifier models are selected as candidates - 4 linear, 5  tree/ensemble and 2 naive bayes. The eleven selected supervised learning models are: 'Logistic Regression', 'Perceptron', 'Stochastic Gradient Descent', 'Ridge Classifier', 'GaussianNB', 'BernoulliNB', 'Decision Tree Classifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'RandomForestClassifier', 'BaggingClassifier'. Naive Bayes models were selected as feature variables are assumed to be independent of each other, relative to the target variable. The linear and tree/ensemble models are selected primarily for the presence of feature_importance or coef_ attribute which which were intended for feature selection.

The classifier models are packaged into a dataframe, and run once each on the train, validation and test datasets. Performance scores for accuracy, precision, recall, F1 and ROC_AUC then returned; see Key findings for the scores below. Model selection is determined by score consistency between the train, validation and test datasets and the Based on the train, validation and test scores. Two models are picked from the list: Logistic regression and Adaboost. k-Nearest Neighbour, an unsupervised learning method is used as the 3rd model. Optimal k-value is assessed to be at 8.

### Performance of models

Results returned from the test set for the models run:

|Algorithm	                  |Accuracy	|Precision	|Recall	  |F1	      |ROC AUC  |
| :----------------------:    | :-----: | :-------: | :-----: | :-----: | :-----: |
|Logistic Regression	        |0.931156	|0.898500	  |0.961284	|0.928832	|0.933003 |
|Perceptron	                  |0.897617	|0.888791	  |0.904762	|0.896705	|0.897741 |
|Stochastic Gradient Descent	|0.931598	|0.899382	  |0.961321	|0.929321	|0.933397 |
|Ridge Classifier	            |0.919682	|0.847308	  |0.990712	|0.913416	|0.928664 |
|GaussianNB	                  |0.901589	|0.850838	  |0.946955	|0.896327	|0.905769 |
|BernoulliNB	                |0.868049	|0.894086	  |0.849832	|0.871398	|0.869050 |
|Decision Tree Classifier	    |0.877317	|0.897617	  |0.862595	|0.879758	|0.877940 |
|AdaBoostClassifier	          |0.932480	|0.909091	  |0.953704	|0.930863	|0.933429 |
|GradientBoostingClassifier	  |0.931598	|0.899382	  |0.961321	|0.929321	|0.933397 |
|RandomForestClassifier	      |0.942189	|0.922330 	|0.960478	|0.941018	|0.942888 |
|BaggingClassifier	          |0.937776	|0.911739	  |0.961825	|0.936112	|0.938966 |

Results from the train and validation set can be found in eda.ipynb. All models have very decent metric scores with performance metric score exceeding >0.80 for all metrics obsrved. For the selected models, the metric scores of Accuracy, Precision, Recall and F1 are also plotted onto Confusion matrix, while the ROC and AUC scores are plotted into graph for clearer presentation.

For the linear model, Logistic Regression is selected as it arguably has the best metric scores out of the 4 linear models (though it must be noted that other models have around the same or slightly worse off performance scores). For the tree/ensemble models, Decision Tree Classifier, RandomForestClassifier and BaggingClassifier are not selected as they have initial F1 scores of 1.00 in the training dataset, a sign of overfitting. Adaboostclassifier is chosen over GradientBoostingClassifier as it requires much less time when Recursive Feature Elimination with Cross Validation is performed.

To determine which of the attribute have a significant predictive value in determining rainfall the next day, feature selection is performed. For logistic regression, coefficient plot is generated, for Adaboost the impurity-based feature importance is applied. For k-nearest neighbour, permutation importance is applied. Recursive Feature Elimination with Cross Validation is also applied to Logistic Regression and Adaboost Classifier as a verification step, after one of the variables that appear unplausible ranked at the top of the most predictive variables.

