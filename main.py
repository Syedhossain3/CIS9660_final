from function import *
from matplotlib import pyplot as plt
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

# ## Partition the data( for supervised tasks)
train, validate = train_test_split(df_duplicate, test_size=0.25, random_state=40)
# print("Training : ", train.shape)
# print("Validation :  ", validate.shape)

# # training (50)
train, temp = train_test_split(df_duplicate, test_size=0.3, random_state=40)
validate, test = train_test_split(temp, test_size=0.3, random_state=40)

X = train[predictors]
y = train[outcome]
#
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=1)

# print(train_X.shape)
# print(train_y)


## Normalization
# We do normalization on numerical data because our data is unbalanced it means the
# difference between the variable values is high so we convert them into 1 and 0
# train_X = normalize_data(train_X)
# valid_X = normalize_data(valid_X)


# # Logistic Regression
logit_reg = LogisticRegression(penalty="l2", C=1e42, solver='liblinear', class_weight='balanced')
logit_reg = LogisticRegression()

logit_reg.fit(train_X, train_y)

score = logit_reg.score(train_X, train_y)
print(score)

print(pd.DataFrame({'coeff': logit_reg.coef_[0]}, index=X.columns))
# # Model Metrics
classificationSummary(train_y, logit_reg.predict(train_X))
classificationSummary(valid_y, logit_reg.predict(valid_X))
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
lr_prediction_train = logit_reg.predict_proba(train_X)[:, 1] > 0.5
lr_prediction_valid = logit_reg.predict_proba(valid_X)[:, 1] > 0.5
print("LR Accuracy on train is:", accuracy_score(train_y, lr_prediction_train))
print("LR Accuracy on test is:", accuracy_score(valid_y, lr_prediction_valid))
print("LR Precision_score train is:", precision_score(train_y, lr_prediction_train, average='micro'))
print("LR Precision_score on test is:", precision_score(valid_y, lr_prediction_valid, average='micro'))
print("LR Recall_score on train is:", recall_score(train_y, lr_prediction_train, average='micro'))
print("LR Recall_score on test is:", recall_score(valid_y, lr_prediction_valid, average='micro'))
print("LR f1_score on train is:", f1_score(train_y, lr_prediction_train, average='micro'))
print("LR f1_score on test is:", f1_score(valid_y, lr_prediction_valid, average='micro'))

# # Decision Tree
DecisionTree = DecisionTreeClassifier(max_depth=4)
DecisionTree.fit(train_X, train_y)
tree.plot_tree(DecisionTree, feature_names=train_X.columns)
plt.show()

# plotDecisionTree(DecisionTree, feature_names=train_X.columns)


#
# # # dot_data = export_graphviz(DecisionTree, filled=True, rounded=True,
# # #                                     class_names=['Setosa',
# # #                                                 'Versicolor',
# # #                                                 'Virginica'],
# # #                                     feature_names=['predictors'],
# # #                                     out_file=None)
# # # graph = graph_from_dot_data(dot_data)
# # # graph.write_png('tree.png')
# # # # graph = graphviz.Source(DecisionTree, format="png")
# # # # graph.render("decision_tree_graphivz")
# # #
# # # # dot_data = StringIO()
# # # # export_graphviz(DecisionTree, out_file=dot_data,
# # # #                 filled=True, rounded=True,
# # # #                 special_characters=True, feature_names = predictors, class_names=['0', '1'])
# # # # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# # # # graph.write_png('wine.png')
# # # # Image(graph.create_png())
# # #
# importances = DecisionTree.feature_importances_
# # #
# im = pd.DataFrame({'feature': train_X.columns, 'importance': importances})
# im = im.sort_values('importance', ascending=False)
# print(im)
# # #
# # #
# dt_prediction_train = DecisionTree.predict(train_X)
# dt_prediction_valid = DecisionTree.predict(valid_X)
#
# print("DT Accuracy score on train is:", accuracy_score(train_y, dt_prediction_train))
# print("DT Accuracy score on test is:", accuracy_score(valid_y, dt_prediction_valid))
# print("DT Precision score on train is:", precision_score(train_y, dt_prediction_train, average='micro'))
# print("DT Precision score on test is:", precision_score(valid_y, dt_prediction_valid, average='micro'))
# print("DT Recall score on train is:", recall_score(train_y, dt_prediction_train, average='micro'))
# print("DT Recall score on test is:", recall_score(valid_y, dt_prediction_valid, average='micro'))
# print("DT F1 score on train is:", f1_score(train_y, dt_prediction_train, average='micro'))

#  # # #Random forest
rf = RandomForestClassifier(random_state=0)
cc_rf = rf.fit(train_X.values, train_y.values.ravel())
rf_prediction_train = cc_rf.predict(train_X)
rf_prediction_valid = cc_rf.predict(valid_X)
# #
# print("RF_Accuracy on train is:",accuracy_score(train_y,rf_prediction_train))
# print("RF_Accuracy on test is:",accuracy_score(valid_y,rf_prediction_valid))
# print("RF_Precision_score train is:",precision_score(train_y,rf_prediction_train, average='micro'))
# print("RF_Precision_score on test is:",precision_score(valid_y,rf_prediction_valid, average='micro'))
# print("RF_Recall_score on train is:",recall_score(train_y,rf_prediction_train, average='micro'))
# print("RF_Recall_score on test is:",recall_score(valid_y,rf_prediction_valid, average='micro'))
# print("RF_f1_score on train is:",f1_score(train_y,rf_prediction_train, average='micro'))
# print("RF_f1_score on test is:",f1_score(valid_y,rf_prediction_valid, average='micro'))

gbm = GradientBoostingClassifier(random_state=0)
gbm.fit(train_X, train_y)
gbm.predict(valid_X[:2])
importances = list(zip(gbm.feature_importances_, train_X.columns))
pd.DataFrame(importances, index=[x for (_, x) in importances]).sort_values(by=0, ascending=False).plot(kind='bar',
                                                                                                       color='b',
                                                                                                       figsize=(20, 8))
# plt.show()
gbt_prediction_train = gbm.predict(train_X)
gbt_prediction_valid = gbm.predict(valid_X)

print("Accuracy on train is:", accuracy_score(train_y, gbt_prediction_train))
print("Accuracy on test is:", accuracy_score(valid_y, gbt_prediction_valid))
print("Precision_score train is:", precision_score(train_y, gbt_prediction_train))
print("Precision_score on test is:", precision_score(valid_y, gbt_prediction_valid))
print("Recall_score on train is:", recall_score(train_y, gbt_prediction_train))
print("Recall_score on test is:", recall_score(valid_y, gbt_prediction_valid))
print("f1_score on train is:", f1_score(train_y, gbt_prediction_train))
print("f1_score on test is:", f1_score(valid_y, gbt_prediction_valid))

# LR
fpr, tpr, thresholds = roc_curve(train_y, lr_prediction_train)
print("LogisticRegression Train: ", str(auc(fpr, tpr)))
fpr, tpr, thresholds = roc_curve(valid_y, lr_prediction_valid)
print("LogisticRegression Valid: ", str(auc(fpr, tpr)), "\n")
# RF
fpr, tpr, thresholds = roc_curve(train_y, rf_prediction_train)
print("RandomForest Train: ", str(auc(fpr, tpr)))
fpr, tpr, thresholds = roc_curve(valid_y, rf_prediction_valid)
print("RandomForest Valid: ", str(auc(fpr, tpr)), "\n")
# GBT
fpr, tpr, thresholds = roc_curve(train_y, gbt_prediction_train)
print("GradientBoostedTree Train: ", str(auc(fpr, tpr)))
fpr, tpr, thresholds = roc_curve(valid_y, gbt_prediction_valid)
print("GradientBoostedTree Valid: ", str(auc(fpr, tpr)), "\n")
