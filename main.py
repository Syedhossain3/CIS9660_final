
from function import *
import seaborn as sb
from function import *
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from dmba import plotDecisionTree
import sys

## Load dataframe
df = pd.read_csv('winequalityN.csv')

## create duplicate datagrame
df_duplicate = df.copy()

## set Pycharm output windows to visualize all columns
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

##Visualization
##Seaborn
# show_seaborn(df_duplicate)

##Histogram
# show_histogram(df_duplicate)

## heatmap
# show_heatmap(df_duplicate)

##convert qualitative data to dummy data
df_duplicate = convert_columns_to_dummy_data(df_duplicate, "type")

## missing percentage
# get_missing_percentage(df_duplicate)

## We saw that there few null value. we convert them with mean value
df_duplicate.update(df_duplicate.fillna(df_duplicate.mean()))

##REname column name
df_duplicate.columns = df_duplicate.columns.str.replace(' ', '_')

##Finding error base on Heap map. check if more than .7
correlation_visualization_err_detection(df_duplicate)
# total_sulfur_dioxide

##drop
df_duplicate.drop('total_sulfur_dioxide', inplace=True, axis=1)

## Showing Boxplot before removing outlier on each column
# detecting_outlier_boxplot(df_duplicate)
# show_histogram(df_duplicate)

## Show ZScore outlier on each column
# detect_outliers_zscore(df_duplicate)

## Remove outlier using Z-Score
# df_duplicate = remove_outliers_zscore(df_duplicate)

## Show IQR outlier on each column
# detect_outlier_IQR(df_duplicate)

## Remove outlier using IQR
# df_duplicate = remove_outlier_iqr(df_duplicate)

# show_histogram(df_duplicate)
## Showing Boxplot after removing outlier on each column
# detecting_outlier_boxplot(df_duplicate)

##Create top_quality column
df_duplicate['top_quality'] = [1 if x >= 7 else 0 for x in df_duplicate.quality]

Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
       'density', 'pH', 'sulphates', 'alcohol', 'type_white', 'top_quality'],
      dtype='object')
predictors = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
              'density', 'pH', 'sulphates', 'alcohol', 'type_white']
outcome = 'top_quality'

# ## Partition the data
train, validate = train_test_split(df_duplicate, test_size=0.25, random_state=40)

# # training (50)
train, temp = train_test_split(df_duplicate, test_size=0.3, random_state=40)
validate, test = train_test_split(temp, test_size=0.3, random_state=40)

X = train[predictors]
y = train[outcome]

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=1)

## Normalization
# We do normalization on numerical data because our data is unbalanced it means the
# difference between the variable values is high so we convert them into 1 and 0
# train_X = normalize_data(train_X)
# valid_X = normalize_data(valid_X)


# # Logistic Regression
logistic_regression = LogisticRegression(penalty="l2", C=1e42, solver='liblinear', class_weight='balanced')
logistic_regression.fit(train_X, train_y)

score = logistic_regression.score(train_X, train_y)
print(score)
print(pd.DataFrame({'coeff': logistic_regression.coef_[0]}, index=X.columns))

## Classification Summery
classificationSummary(train_y, logistic_regression.predict(train_X))
classificationSummary(valid_y, logistic_regression.predict(valid_X))

## Model Metrics
logistic_regression_prediction_train = logistic_regression.predict_proba(train_X)[:, 1] > 0.5
logistic_regression_prediction_valid = logistic_regression.predict_proba(valid_X)[:, 1] > 0.5
model_matrix(train_y, valid_y, logistic_regression_prediction_train, logistic_regression_prediction_valid)

## Decision Tree Model
decision_tree = DecisionTreeClassifier(max_depth=4)
decision_tree.fit(train_X, train_y)

# plotDecisionTree(DecisionTree, feature_names=train_X.columns)
# Visualize data
visualization_decision_tree(decision_tree, train_X)

importances = decision_tree.feature_importances_
im = pd.DataFrame({'feature': train_X.columns, 'importance': importances})
im = im.sort_values('importance', ascending=False)
print(im)

## Model Matrix
decision_tree_prediction_train = decision_tree.predict(train_X)
decision_tree_prediction_valid = decision_tree.predict(valid_X)
model_matrix(train_y, valid_y, decision_tree_prediction_train, decision_tree_prediction_valid)

## #Random forest
random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(train_X.values, train_y.values.ravel())

## Model Matrix
random_forest_prediction_train = random_forest.predict(train_X)
random_forest_prediction_valid = random_forest.predict(valid_X)
model_matrix(train_y, valid_y, random_forest_prediction_train, random_forest_prediction_valid)

## Gradient Boosting
gradient_boosting = GradientBoostingClassifier(random_state=0)
gradient_boosting.fit(train_X, train_y)
gradient_boosting.predict(valid_X[:2])

gradient_boosting_importances = list(zip(gradient_boosting.feature_importances_, train_X.columns))
pd.DataFrame(importances, index=[x for (_, x) in gradient_boosting_importances]).sort_values(by=0, ascending=False).plot(kind='bar',
                                                                                                       color='b',
                                                                                                      figsize=(20, 8))
plt.show()

## Model Matrix
gradient_boosting_prediction_train = gradient_boosting.predict(train_X)
gradient_boosting_prediction_valid = gradient_boosting.predict(valid_X)
model_matrix(train_y, valid_y, gradient_boosting_prediction_train, gradient_boosting_prediction_valid)

## Baseline AUC analysis
# logistic_regression
fpr, tpr, thresholds = roc_curve(train_y, logistic_regression_prediction_train)
print("LogisticRegression Train: ", str(auc(fpr, tpr)))
fpr, tpr, thresholds = roc_curve(valid_y, logistic_regression_prediction_valid)
print("LogisticRegression Valid: ", str(auc(fpr, tpr)), "\n")
# random_forest
fpr, tpr, thresholds = roc_curve(train_y, random_forest_prediction_train)
print("RandomForest Train: ", str(auc(fpr, tpr)))
fpr, tpr, thresholds = roc_curve(valid_y, random_forest_prediction_valid)
print("RandomForest Valid: ", str(auc(fpr, tpr)), "\n")
# gradient_boosting
fpr, tpr, thresholds = roc_curve(train_y, gradient_boosting_prediction_train)
print("GradientBoostedTree Train: ", str(auc(fpr, tpr)))
fpr, tpr, thresholds = roc_curve(valid_y, gradient_boosting_prediction_valid)
print("GradientBoostedTree Valid: ", str(auc(fpr, tpr)), "\n")
