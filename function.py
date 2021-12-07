import math
import matplotlib.pyplot as plt
import csv
from ast import Index
from json import encoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
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


def percntage_missing(df):
    # '''prints out columns with missing values with its %'''
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct != 0:
            print('{} => {}%'.format(col, round(pct, 2)))


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
def detect_outlier_IQR(data_frame):
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
    sb.pairplot(df_duplicate)
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
    pd.get_dummies(data_frame, columns=[column_name], drop_first=True)
