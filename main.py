from function import *
from matplotlib import pyplot as plt
import seaborn as sb
from function import *
import pandas as pd
import numpy as np
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

## drop all null from row
df_duplicate = df_duplicate.dropna()
# print(df_duplicate.shape)
# (6463, 14)
#
##REname column name
df_duplicate.columns = df_duplicate.columns.str.replace(' ', '_')
# print(df_duplicate.describe())

##Finding error base on Heap map. check if more than .7
correlation_visualization_err_detection(df_duplicate)
# total_sulfur_dioxide
##drop
df_duplicate.drop('total_sulfur_dioxide', inplace=True, axis=1)

print(df_duplicate.head(20))

## Showing Boxplot before removing outlier on each column
# detecting_outlier_boxplot(df_duplicate)

# show_histogram(df_duplicate)
## Show ZScore outlier on each column
# # detect_outliers_zscore(df_duplicate)
#
## Remove outlier using Z-Score
df_duplicate = remove_outliers_zscore(df_duplicate)
# # print(df_duplicate.shape)
# # (5955, 12)
#
# ## Show IQR outlier on each column
# # detect_outlier_IQR(df_duplicate)
#
## Remove outlier using IQR
df_duplicate = remove_outlier_iqr(df_duplicate)
# # print(df_duplicate.shape)
# # (4622, 12)
#
# # show_histogram(df_duplicate)
# ## Showing Boxplot after removing outlier on each column
# # detecting_outlier_boxplot(df_duplicate)
#
##Create a new dummy code
df_duplicate['best_quality']=[1 if x>=7 else 0 for x in df_duplicate.quality]
# print(df_dummies)
# df_duplicate['best_quality'] = np.where(df_duplicate.quality >= 7, 1, 0)

# Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
#        'density', 'pH', 'sulphates', 'alcohol', 'quality', 'type_white'],
#       dtype='object')
# predictors = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
#               'density', 'pH', 'sulphates', 'alcohol', 'type_white']
# outcome = 'quality'
#
# # ## Partition the data( for supervised tasks)
# train, validate = train_test_split(df_duplicate, test_size=0.25, random_state=1)
# # print("Training : ", train.shape)
# # print("Validation :  ", validate.shape)
#
# # # training (50)
# train, temp = train_test_split(df_duplicate, test_size=0.3, random_state=1)
# validate, test = train_test_split(temp, test_size=0.3, random_state=1)
#
# X = train[predictors]
# y = train[outcome]
#
# train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=1)
#
# # # Logistic Regression
# logit_reg = LogisticRegression(penalty="l2", C=1e42, solver='liblinear', class_weight='balanced')
# logit_reg = LogisticRegression()
#
# logit_reg.fit(train_X, train_y)
#
# score = logit_reg.score(train_X, train_y)
# print(score)
#
# print(pd.DataFrame({'coeff': logit_reg.coef_[0]}, index=X.columns))
