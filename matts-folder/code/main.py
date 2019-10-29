import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def read_heart_data():
    return pd.read_csv('../../heart.csv')


def print_count_of_target(target, df):
    print('\nTarget value count:')
    print(df.target.value_counts())
    sns.countplot(x=target, data=df)
    plt.show()


def run_pca(x_scaled):
    pca = PCA()
    pca.fit(x_scaled)
    x_pca = pca.transform(x_scaled)
    print("shape of X_pca", x_pca.shape)

    # Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('Explained Variance')
    plt.show()

    ex_variance = np.var(x_pca, axis=0)
    ex_variance_ratio = ex_variance / np.sum(ex_variance)
    print (ex_variance_ratio)

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
    return scaler.transform(df)


if __name__ == "__main__":
    df = read_heart_data()
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 120)
    print(df.head(10))

    # show how many of each target we have
    # print_count_of_target('target', df)

    # get data without target values
    data = df.drop('target', axis=1)

    # do some feature scaling
    transformed_data = feature_scaling(data)

    # now look at feature reduction
    target = df['target'].values

    # PCA
    run_pca(transformed_data)

    # comment this out for now
    # reduced_data, feature_index = feature_reduction(transformed_data, target)
    # print(reduced_data.shape)

    # print the list of features that were selected from the feature reduction
    reduced_features = []
    feature_names = list(data.columns.values)
    for was_selected, feature in zip(feature_index, feature_names):
        if was_selected:
            reduced_features.append(feature)

    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        test_size = 0.33, random_state = 42)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)

    clf.fit(X_train, y_train)

    print('Selected Features:')
    print(str(reduced_features))
    print('Done')





