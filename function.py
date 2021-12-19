import math
import matplotlib.pyplot as plt
import csv
import collections
import pydotplus
from ast import Index
from json import encoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from lime import lime_tabular
import matplotlib
import model2 as model2
import pandas as pd
import numpy as np
from scipy import stats
from jedi.api.refactoring import inline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from IPython.display import display
import seaborn as sb
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from bokeh.models import Y
from numpy import int64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, BayesianRidge
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn import metrics
from dmba import classificationSummary, gainsChart, liftChart, regressionSummary
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lime import lime_tabular
from dmba import plotDecisionTree
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import plot_roc_curve
import graphviz
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as metrics
from sklearn.metrics import plot_roc_curve
import h2o
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import math



def get_missing_percentage(df):
    return print(df.isnull().sum() / (len(df)) * 100)


def get_number_of_wine_by_country(df_column, x_label, y_label):
    plt.figure(figsize=(16, 7))
    sns.set(style="darkgrid")
    sns.barplot(x=df_column.value_counts()[:10].index, y=df_column.value_counts()[:10].values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return plt.show()


# Average Points
def get_average_points_by_country(df_column, x_lable, y_lable, title):
    plt.figure(figsize=(16, 7))
    g = sns.barplot(x=df.groupby("country").mean().sort_values(by="points", ascending=False).points.index[:10],
                    y=df.groupby("country").mean().sort_values(by="points", ascending=False).points.values[:10],
                    palette="gist_ncar")
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.title(title)
    ax = g
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),
                    textcoords='offset points')
    return plt.show()


## Finding outlier
def detecting_outlier_boxplot(df):
    for col in df:
        plt.boxplot(df[col], vert=False)
        plt.title("Detecting outliers using Boxplot")
        plt.xlabel(col)
        plt.show()


## Remove outlier in dataframe
def remove_outliers_zscore(data_frame):
    abs_z_scores = np.abs(stats.zscore(data_frame))
    data_frame = data_frame[(abs_z_scores < 3).all(axis=1)]
    return data_frame


## Detect outlier in dataframe
def detect_outliers_zscore(data_frame):
    abs_z_scores = np.abs(stats.zscore(data_frame))
    return print(np.where(abs_z_scores > 3))


##IQR for each column
def detect_outlier_iqr(data_frame):
    q1 = data_frame.quantile(0.25)
    q3 = data_frame.quantile(0.75)
    iqr = q3 - q1
    print(iqr)
    print((data_frame < (q1 - 1.5 * iqr)) | (data_frame > (q3 + 1.5 * iqr)))


##Remove the outliers from the dataset
def remove_outlier_iqr(data_frame):
    q1 = data_frame.quantile(0.25)
    q3 = data_frame.quantile(0.75)
    iqr = q3 - q1
    data_frame = data_frame[~((data_frame < (q1 - 1.5 * iqr)) | (data_frame > (q3 + 1.5 * iqr))).any(axis=1)]
    return data_frame


##Historgram
def show_histogram(data_frame):
    data_frame.hist(bins=20, figsize=(10, 10))
    # show graph
    plt.show()


##Seaborn
def show_seaborn(data_frame):
    sb.pairplot(data_frame)
    # show graph
    plt.show()


## heatmap
def show_heatmap(data_frame):
    # correlation by visualization
    plt.figure(figsize=[18, 7])
    # plot correlation
    sb.heatmap(data_frame.corr(), annot=True)
    plt.show()


## Convert Qualitative data to dummy data
def convert_columns_to_dummy_data(data_frame, column_name):
    data_frame = pd.get_dummies(data_frame, columns=[column_name], drop_first=True)
    return data_frame


##Correlation Visualization Error Detection
def correlation_visualization_err_detection(data_frame):
    colm = []
    # loop for columns
    for i in range(len(data_frame.corr().keys())):
        for j in range(i):
            if abs(data_frame.corr().iloc[i, j]) > 0.7:
                colm = data_frame.corr().columns[i]
                return print(colm)


##Normatize
def normalize_data(data_frame):
    norm = MinMaxScaler()
    norm_fit = norm.fit(data_frame)
    return norm_fit.transform(data_frame)


##Model Matrix
def model_matrix(data_frame_train_y, data_frame_valid_y, object_train, object_valid, model_name):
    print(model_name + ": accuracy on train is:", accuracy_score(data_frame_train_y, object_train))
    print(model_name + ": accuracy on test is:", accuracy_score(data_frame_valid_y, object_valid))
    print(model_name + ": precision_score train is:", precision_score(data_frame_train_y, object_train))
    print(model_name + ": precision_score on test is:", precision_score(data_frame_valid_y, object_valid))
    print(model_name + ": Recall_score on train is:", recall_score(data_frame_train_y, object_train))
    print(model_name + ": Recall_score on test is:", recall_score(data_frame_valid_y, object_valid))
    print(model_name + ": f1_score on train is:", f1_score(data_frame_train_y, object_train))
    print(model_name + ": f1_score on test is:", f1_score(data_frame_valid_y, object_valid))


## Visualization DecisionTree
def visualization_decision_tree(object_tree, data_frame):
    dot_data = tree.export_graphviz(object_tree,
                                    feature_names=data_frame.columns,
                                    out_file=None,
                                    filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    colors = ('turquoise', 'orange')
    edges = collections.defaultdict(list)
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))
    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])
    graph.write_png('decision_tree.png')


## Feature Importance
def feature_importance(object_importance):
    pd.DataFrame(object_importance, index=[x for (_, x) in object_importance]).sort_values(by=0, ascending=False).plot(
        kind='bar', color='b', figsize=(20, 8))
    plt.show()


## Baseline AUC analysis
def baseline_auc_analysis(data_frame, object_model_prediction, object_model_name):
    fpr, tpr, thresholds = roc_curve(data_frame, object_model_prediction)
    if 'Valid' in object_model_name:
        print(object_model_name + ": ", str(auc(fpr, tpr)), "\n")
    else:
        print(object_model_name + ": ", str(auc(fpr, tpr)))


#### ROC Curve Analysis
def roc_curve_analysis(object_model_name, data_frame_x, data_frame_y):
    object_proba = object_model_name.predict_proba(data_frame_x)[:, 1]
    object_roc = roc_curve(data_frame_y, object_proba)
    return pd.DataFrame(object_roc)


##Roc curve analysis
def roc_cure_analysis_classifier_result(classifier, valid_X, valid_y, random_forest_roc,
                                        gradient_boosting_roc, random_forest_str, gradiant_boosted_str):
    result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
    for cls in classifier:
        y_probability = cls.predict_proba(valid_X)[:, 1]
        # plot_roc_curve(cls, valid_X, valid_y)
        fpr, tpr, thresholds = roc_curve(valid_y, y_probability)

        auc = roc_auc_score(valid_y, y_probability)

        result_table = result_table.append({'classifiers': cls,
                                            'fpr': fpr,
                                            'tpr': tpr,
                                            'auc': auc}, ignore_index=True)
    result_table.set_index('classifiers', inplace=True)
    fig = plt.figure(figsize=(8, 6))

    # print(classifier_result_table.head())

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label=i)
        # label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot(random_forest_roc.loc[0, :], random_forest_roc.loc[1, :], label=random_forest_str)
    plt.plot(gradient_boosting_roc.loc[0, :], gradient_boosting_roc.loc[1, :], label=gradiant_boosted_str)

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 8}, loc='lower right')

    plt.show()


def gradient_boosting(df):
    import h2o
    from h2o.estimators import H2OGradientBoostingEstimator
    h2o.init(nthreads=-1, min_mem_size_GB=8)

    # Dataset:
    wh2o = h2o.H2OFrame(df)
    train, valid = wh2o.split_frame(ratios=[.8], seed=-1)

    # Set the predictors and response; set the factors:
    wh2o['points'] = wh2o['points'].asfactor()
    predictors = ['country', 'points', 'province', 'variety', 'winery', 'region_1']
    response = "log_price_bins"
    # Build and train the model:
    w_gbm = H2OGradientBoostingEstimator(nfolds=100,
                                         seed=-1,
                                         distribution='multinomial',
                                         max_depth=10,
                                         sample_rate=0.6000000000000001,
                                         learn_rate=0.06,
                                         col_sample_rate=1.0,
                                         auc_type='WEIGHTED_OVR',
                                         max_runtime_secs=600
                                         )

    w_gbm.train(x=predictors, y=response, training_frame=train)
    # Eval performance:
    perf = w_gbm.model_performance()
    # Generate predictions on a test set (if necessary):
    pred = w_gbm.predict(valid)
    return valid, pred


def create_heatmap(x, y, size, color):
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    size_scale = 200

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x15 grid
    ax = plt.subplot(plot_grid[:, :-1])  # Use the leftmost 14 columns of the grid for the main plot

    ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size * size_scale,  # Vector of square sizes, proportional to size parameter
        c=color.apply(value_to_color),  # Vector of square colors, mapped to color palette
        marker='s'  # Use square as scatterplot marker
    )

    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    # Add color legend on the right side of the plot
    ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost column of the plot

    col_x = [0] * len(palette)  # Fixed x coordinate for the bars
    bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars

    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5] * len(palette),  # Make bars 5 units wide
        left=col_x,  # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )
    ax.set_xlim(1, 2)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.grid(False)  # Hide grid
    ax.set_facecolor('white')  # Make background white
    ax.set_xticks([])  # Remove horizontal ticks
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
    ax.yaxis.tick_right()  # Show vertical ticks on the right

    n_colors = 256  # Use 256 colors for the diverging color palette
    palette = sns.diverging_palette(20, 220, n=n_colors)  # Create the palette
    color_min, color_max = [-1, 1]  # Range of values that will be mapped to the palette, i.e. min and max possible correlation


## LIME
def lime_data_visualization(data_frame_train_X, data_frame_valid_X, model_name, output_file):
    explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(train_X),
                                             mode="regression",
                                             feature_names=data_frame_train_X.columns,
                                             categorical_features=[0])
    exp = explainer.explain_instance(data_row=data_frame_valid_X.iloc[4],
                                predict_fn=model_name.predict)
    exp.save_to_file(output_file)