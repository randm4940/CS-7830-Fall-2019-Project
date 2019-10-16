import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


def read_heart_data():
    return pd.read_csv('../../heart.csv')


def print_count_of_target(target, df):
    print('\nTarget value count:')
    print(df.target.value_counts())
    sns.countplot(x=target, data=df)
    plt.show()


def feature_reduction(df, target):
    # Before feature reduction to remove features that don't impact the output significantly
    tree = ExtraTreesClassifier(n_estimators=100)
    tree.fit(df, target)
    model = SelectFromModel(tree, prefit=True)
    feature_index = model.get_support()
    data_new = model.transform(df)
    return [data_new, feature_index]


def feature_scaling(df):
    # Use StandardScaler to provide Gaussian distribution of data (0 centering and 1 standard deviation)
    scaler = StandardScaler()
    scaler.fit(df)
    data = scaler.transform(df)
    return data


if __name__ == "__main__":
    df = read_heart_data()
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 120)
    print(df.head(10))

    # show how many of each target we have
    print_count_of_target('target', df)

    # get data without target values
    data = df.drop('target', axis=1)

    # do some feature scaling
    transformed_data = feature_scaling(data)

    # now look at feature reduction
    target = df['target'].values
    reduced_data, feature_index = feature_reduction(transformed_data, target)
    print(reduced_data.shape)

    # print the list of features that were selected from the feature reduction
    reduced_features = []
    feature_names = list(data.columns.values)
    for was_selected, feature in zip(feature_index, feature_names):
        if was_selected:
            reduced_features.append(feature)

    print('Selected Features:')
    print(str(reduced_features))
    print('Done')





