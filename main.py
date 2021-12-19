import matplotlib.pyplot
from numpy import savetxt
from function import *
from function import *
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


## Load dataframe
df = pd.read_csv('winequalityN.csv')

## create duplicate datagrame
df_duplicate = df.copy()

## set Pycharm output windows to visualize all columns
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

##Visualization
##Seaborn
show_seaborn(df_duplicate)

##Histogram
show_histogram(df_duplicate)

## heatmap
show_heatmap(df_duplicate)

##convert qualitative data to dummy data
df_duplicate = convert_columns_to_dummy_data(df_duplicate, "type")

## missing percentage
# get_missing_percentage(df_duplicate)

## We saw that there few null value. we convert them with mean value
df_duplicate.update(df_duplicate.fillna(df_duplicate.mean()))

##Rename columns name
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
model_matrix(train_y, valid_y, logistic_regression_prediction_train, logistic_regression_prediction_valid, "Logistic Regression")

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
model_matrix(train_y, valid_y, decision_tree_prediction_train, decision_tree_prediction_valid, "Decision Tree")

###Random forest
random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(train_X.values, train_y.values.ravel())

# im = pd.DataFrame({'feature': train_X.columns, 'importance': random_forest.feature_importances_})
# im = im.sort_values('importance', ascending=False)
# print(im)

## Model Matrix
random_forest_prediction_train = random_forest.predict(train_X)
random_forest_prediction_valid = random_forest.predict(valid_X)
model_matrix(train_y, valid_y, random_forest_prediction_train, random_forest_prediction_valid, "Random forest")

##Naive Bayes
naive_bayes = GaussianNB()
naive_bayes.fit(train_X, train_y)

# predict probabilities
# naive_bayes_prediction_train = naive_bayes.predict_proba(train_X)
# naive_bayes_prediction_valid = naive_bayes.predict_proba(valid_X)

##Model Metrics
naive_bayes_prediction_train = naive_bayes.predict(train_X)
naive_bayes_prediction_valid = naive_bayes.predict(valid_X)
model_matrix(train_y, valid_y, naive_bayes_prediction_train, naive_bayes_prediction_valid, "Naive Bayes")


### Gradient Boosting
gradient_boosting = GradientBoostingClassifier(random_state=0)
gradient_boosting.fit(train_X, train_y)
gradient_boosting.predict(valid_X[:2])

gradient_boosting_importances = list(zip(gradient_boosting.feature_importances_, train_X.columns))
feature_importance(gradient_boosting_importances)

## Model Matrix
gradient_boosting_prediction_train = gradient_boosting.predict(train_X)
gradient_boosting_prediction_valid = gradient_boosting.predict(valid_X)
model_matrix(train_y, valid_y, gradient_boosting_prediction_train, gradient_boosting_prediction_valid, "Gradient Boosting Tree")

## Neural Network
# Need to complete Neural Network

### Baseline AUC analysis
# logistic_regression
baseline_auc_analysis(train_y, logistic_regression_prediction_train, "LogisticRegression Train")
baseline_auc_analysis(valid_y, logistic_regression_prediction_valid, "LogisticRegression Valid")

# decision_tree
baseline_auc_analysis(train_y, decision_tree_prediction_train, "Decision Tree Train")
baseline_auc_analysis(valid_y, decision_tree_prediction_valid, "Decision Tree Valid")

# naive_bayes
baseline_auc_analysis(train_y, naive_bayes_prediction_train, "Naive Bayes Train")
baseline_auc_analysis(valid_y, naive_bayes_prediction_valid, "Naive Bayes Valid")

# random_forest
baseline_auc_analysis(train_y, random_forest_prediction_train, "RandomForest Train")
baseline_auc_analysis(valid_y, random_forest_prediction_valid, "RandomForest Valid")

# gradient_boosting
baseline_auc_analysis(train_y, gradient_boosting_prediction_train, "GradientBoostedTree Train")
baseline_auc_analysis(valid_y, gradient_boosting_prediction_valid, "GradientBoostedTree Valid")

## ROC Curve Analysis
random_forest_roc = roc_curve_analysis(random_forest, valid_X, valid_y)
gradient_boosting_roc = roc_curve_analysis(gradient_boosting, valid_X, valid_y)
logistic_regression_roc = roc_curve_analysis(logistic_regression, valid_X, valid_y)
decision_tree_roc = roc_curve_analysis(decision_tree, valid_X, valid_y)

classifier = [logistic_regression, decision_tree, naive_bayes]
roc_cure_analysis_classifier_result(classifier, valid_X, valid_y, random_forest_roc, gradient_boosting_roc, "Random Forest", "Gradiant Boosted")

##LIME with gradient_boosting
lime_data_visualization(train_X, valid_X, gradient_boosting, "lime.html")

