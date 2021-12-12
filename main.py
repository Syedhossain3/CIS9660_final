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
import sklearn.metrics as metrics
from sklearn.metrics import plot_roc_curve
from numpy import savetxt

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


### Logistic Regression
logistic_regression = LogisticRegression(penalty="l2", C=1e42, solver='liblinear', class_weight='balanced')
logistic_regression.fit(train_X, train_y)

# print(pd.DataFrame({'coeff': logistic_regression.coef_[0]}, index=X.columns))

## Classification Summery
classificationSummary(train_y, logistic_regression.predict(train_X))
classificationSummary(valid_y, logistic_regression.predict(valid_X))

## Model Metrics
logistic_regression_prediction_train = logistic_regression.predict_proba(train_X)[:, 1] > 0.5
logistic_regression_prediction_valid = logistic_regression.predict_proba(valid_X)[:, 1] > 0.5
model_matrix(train_y, valid_y, logistic_regression_prediction_train, logistic_regression_prediction_valid)

### Decision Tree Model
decision_tree = DecisionTreeClassifier(max_depth=4)
decision_tree.fit(train_X, train_y)

# plotDecisionTree(DecisionTree, feature_names=train_X.columns)
# Visualize data
visualization_decision_tree(decision_tree, train_X)

decision_tree_importances = decision_tree.feature_importances_
# feature_importance(decision_tree_importances)
# im = pd.DataFrame({'feature': train_X.columns, 'importance': decision_tree_importances})
# im = im.sort_values('importance', ascending=False)
# print(im)

## Model Matrix
decision_tree_prediction_train = decision_tree.predict(train_X)
decision_tree_prediction_valid = decision_tree.predict(valid_X)
model_matrix(train_y, valid_y, decision_tree_prediction_train, decision_tree_prediction_valid)

###Random forest
random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(train_X.values, train_y.values.ravel())

## Model Matrix
random_forest_prediction_train = random_forest.predict(train_X)
random_forest_prediction_valid = random_forest.predict(valid_X)
model_matrix(train_y, valid_y, random_forest_prediction_train, random_forest_prediction_valid)

### Gradient Boosting
gradient_boosting = GradientBoostingClassifier(random_state=0)
gradient_boosting.fit(train_X, train_y)
gradient_boosting.predict(valid_X[:2])

gradient_boosting_importances = list(zip(gradient_boosting.feature_importances_, train_X.columns))
# feature_importance(gradient_boosting_importances)

## Model Matrix
gradient_boosting_prediction_train = gradient_boosting.predict(train_X)
gradient_boosting_prediction_valid = gradient_boosting.predict(valid_X)
model_matrix(train_y, valid_y, gradient_boosting_prediction_train, gradient_boosting_prediction_valid)

### Baseline AUC analysis
# logistic_regression
baseline_auc_analysis(train_y, logistic_regression_prediction_train, "LogisticRegression Train")
baseline_auc_analysis(valid_y, logistic_regression_prediction_valid, "LogisticRegression Valid")

# random_forest
baseline_auc_analysis(train_y, random_forest_prediction_train, "RandomForest Train")
baseline_auc_analysis(valid_y, random_forest_prediction_valid, "RandomForest Valid")

# gradient_boosting
baseline_auc_analysis(train_y, gradient_boosting_prediction_train, "GradientBoostedTree Train")
baseline_auc_analysis(valid_y, gradient_boosting_prediction_valid, "GradientBoostedTree Valid")

## ROC Curve Analysis
random_forest_roc = roc_curve_analysis(random_forest, valid_X, valid_y)
gradient_boosting_roc = roc_curve_analysis(gradient_boosting, valid_X, valid_y)


Classifier = [logistic_regression, decision_tree]
result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
for cls in Classifier:
    yproba = cls.predict_proba(valid_X)[:, 1]
    # plot_roc_curve(cls, valid_X, valid_y)
    fpr, tpr, thresholds = roc_curve(valid_y, yproba)

    auc = roc_auc_score(valid_y, yproba)

    result_table = result_table.append({'classifiers': cls,
                                        'fpr': fpr,
                                        'tpr': tpr,
                                        'auc': auc}, ignore_index=True)
result_table.set_index('classifiers', inplace=True)
print(result_table)
# result_table.fillna(0)
fig = plt.figure(figsize=(8, 6))

print(result_table.head())

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'],
             result_table.loc[i]['tpr'],
             label=i)
    # label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

plt.plot(random_forest_roc.loc[0, :], random_forest_roc.loc[1, :], label="Random Forest")
plt.plot(gradient_boosting_roc.loc[0, :],  gradient_boosting_roc.loc[1, :], label="Gradiant Boosted Tree")
# plt.plot(nn_roc.loc[0, :], nn_roc.loc[1, :], label="Neural Network")

plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 8}, loc='lower right')

# plt.show()

# final_prediction = logistic_regression.predict(df_duplicate[predictors])
#
# savetxt('testPred.csv', final_prediction, delimiter=',')