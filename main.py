from function import *
from matplotlib import pyplot as plt
import seaborn as sns
from function import *
import pandas as pd
import numpy as np

## Load dataframe
df = pd.read_csv('winequalityN.csv')
# print(df.shape)
## create duplicate datagrame
df_duplicate = df.copy()

## set Pycharm output windows to visualize all columns
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

##convert qualitative data to dummy data
df_duplicate = convert_columns_to_dummy_data(df_duplicate, "type")

#drop collaborative field
df_duplicate.drop(["type"], axis = 1)

print(df_duplicate.head())

# ## drop all null from row
# df_duplicate = df_duplicate.dropna()
# # print(df_duplicate.shape)
# # (6497, 13)
#
# ##REname column name
# df_duplicate.columns = df_duplicate.columns.str.replace(' ', '_')
# # print(df_duplicate.describe())
#
# ## Showing Boxplot before removing outlier on each column
# # detecting_outlier_boxplot(df_duplicate)
#
# # show_histogram(df_duplicate)
# ## Show ZScore outlier on each column
# # detect_outliers_zscore(df_duplicate)
#
# ## Remove outlier using Z-Score
# df_duplicate = remove_outliers_zscore(df_duplicate)
# # print(df_duplicate.shape)
# # (5955, 12)
#
# ## Show IQR outlier on each column
# # detect_outlier_IQR(df_duplicate)
#
# ## Remove outlier using IQR
# df_duplicate = remove_outlier_iqr(df_duplicate)
# # print(df_duplicate.shape)
# # (4622, 12)
#
# # show_histogram(df_duplicate)
# ## Showing Boxplot after removing outlier on each column
# # detecting_outlier_boxplot(df_duplicate)
#
# Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
#        'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality'],
#       dtype='object')
# predictors = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
#               'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']
# outcome = 'quality'
#
# #6 Partition the data( for supervised tasks)
# train, validate = train_test_split(df_duplicate, test_size=0.25, random_state=1)
# # print("Training : ", train.shape)
# # print("Validation :  ", validate.shape)
#
# # training (50)
# train, temp = train_test_split(df_duplicate, test_size=0.3, random_state=1)
# validate, test = train_test_split(temp, test_size=0.3, random_state=1)
#
# X = train[predictors]
# y = train[outcome]
#
# train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=1)
#
# # Logistic Regression
# # logit_reg = LogisticRegression(penalty="l2", C=1e42, solver='liblinear', class_weight='balanced')
# # logit_reg = LogisticRegression()
# #
# # logit_reg.fit(train_X, train_y)
# #
# # score = logit_reg.score(train_X, train_y)
# # print(score)
# #
# # print(pd.DataFrame({'coeff': logit_reg.coef_[0]}, index=X.columns))
# # # regressionSummary(train_y, logit_reg.predict(train_X))
# # # Model Metrics
# # classificationSummary(train_y, logit_reg.predict(train_X))
# # classificationSummary(valid_y, logit_reg.predict(valid_X))
# # # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# # lr_prediction_train = logit_reg.predict_proba(train_X)[:, 1] > 0.5
# # lr_prediction_valid = logit_reg.predict_proba(valid_X)[:, 1] > 0.5
# # print("LR Accuracy on train is:", accuracy_score(train_y, lr_prediction_train))
# # print("LR Accuracy on test is:", accuracy_score(valid_y, lr_prediction_valid))
# # print("LR Precision_score train is:", precision_score(train_y, lr_prediction_train, average='micro'))
# # print("LR Precision_score on test is:", precision_score(valid_y, lr_prediction_valid, average='micro'))
# # print("LR Recall_score on train is:", recall_score(train_y, lr_prediction_train, average='micro'))
# # print("LR Recall_score on test is:", recall_score(valid_y, lr_prediction_valid, average='micro'))
# # print("LR f1_score on train is:", f1_score(train_y, lr_prediction_train, average='micro'))
# # print("LR f1_score on test is:", f1_score(valid_y, lr_prediction_valid, average='micro'))
# # # # # Decision Tree
# # DecisionTree = DecisionTreeClassifier(max_depth = 4)
# # DecisionTree.fit(train_X, train_y)
# #
# # plotDecisionTree(DecisionTree, feature_names=train_X.columns)
# # fig, ax = plt.subplots(figsize=(8, 10))
# # tree.plot_tree(DecisionTree, fontsize=6)
# # plt.show()
# # # # dot_data = export_graphviz(DecisionTree, filled=True, rounded=True,
# # # #                                     class_names=['Setosa',
# # # #                                                 'Versicolor',
# # # #                                                 'Virginica'],
# # # #                                     feature_names=['predictors'],
# # # #                                     out_file=None)
# # # # graph = graph_from_dot_data(dot_data)
# # # # graph.write_png('tree.png')
# # # # # graph = graphviz.Source(DecisionTree, format="png")
# # # # # graph.render("decision_tree_graphivz")
# # # #
# # # # # dot_data = StringIO()
# # # # # export_graphviz(DecisionTree, out_file=dot_data,
# # # # #                 filled=True, rounded=True,
# # # # #                 special_characters=True, feature_names = predictors, class_names=['0', '1'])
# # # # # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# # # # # graph.write_png('wine.png')
# # # # # Image(graph.create_png())
# # # #
# # importances = DecisionTree.feature_importances_
# # # #
# # im = pd.DataFrame({'feature': train_X.columns, 'importance': importances})
# # im = im.sort_values('importance',ascending=False)
# # # print(im)
# # # #
# # # #
# # dt_prediction_train = DecisionTree.predict(train_X)
# # dt_prediction_valid = DecisionTree.predict(valid_X)
# # #
# # print("DT Accuracy score on train is:", accuracy_score(train_y, dt_prediction_train))
# # print("DT Accuracy score on test is:", accuracy_score(valid_y, dt_prediction_valid))
# # print("DT Precision score on train is:", precision_score(train_y, dt_prediction_train, average='micro'))
# # print("DT Precision score on test is:", precision_score(valid_y, dt_prediction_valid, average='micro'))
# # print("DT Recall score on train is:", recall_score(train_y, dt_prediction_train, average='micro'))
# # print("DT Recall score on test is:", recall_score(valid_y, dt_prediction_valid, average='micro'))
# # print("DT F1 score on train is:", f1_score(train_y, dt_prediction_train, average='micro'))
# #
# # # # #Random forest
# # rf = RandomForestClassifier(random_state=0)
# # cc_rf = rf.fit(train_X.values, train_y.values.ravel())
# # rf_prediction_train = cc_rf.predict(train_X)
# # rf_prediction_valid = cc_rf.predict(valid_X)
# # #
# # print("RF_Accuracy on train is:",accuracy_score(train_y,rf_prediction_train))
# # print("RF_Accuracy on test is:",accuracy_score(valid_y,rf_prediction_valid))
# # print("RF_Precision_score train is:",precision_score(train_y,rf_prediction_train, average='micro'))
# # print("RF_Precision_score on test is:",precision_score(valid_y,rf_prediction_valid, average='micro'))
# # print("RF_Recall_score on train is:",recall_score(train_y,rf_prediction_train, average='micro'))
# # print("RF_Recall_score on test is:",recall_score(valid_y,rf_prediction_valid, average='micro'))
# # print("RF_f1_score on train is:",f1_score(train_y,rf_prediction_train, average='micro'))
# # print("RF_f1_score on test is:",f1_score(valid_y,rf_prediction_valid, average='micro'))
