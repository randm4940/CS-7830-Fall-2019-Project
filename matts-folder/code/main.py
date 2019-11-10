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
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def read_heart_data():
    return pd.read_csv('../../heart.csv')


def parameter_optimization(df):
    import warnings
    warnings.simplefilter('once', ConvergenceWarning)

    shuffled_data = df.sample(frac=1)
    data = shuffled_data.drop('target', axis=1)
    target = shuffled_data['target'].values
    data_values = feature_scaling(data)
    X_train, X_test, y_train, y_test = train_test_split(data_values, target,
                                                        test_size=0.30, random_state=1)

    mlp = MLPClassifier(max_iter=2000)
    parameter_space = {
        'hidden_layer_sizes': [(5,), (6,), (7,), (8,), (9,), (5, 3),  (5, 4),  (6, 3),  (6, 4), (5, 3, 2),
                               (5, 3, 3), ],
        'activation': ['tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0001, 0.05, 0, 0.01, 1, 10, 20, 25, 30],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01, 0.05, 0.1, 0.5]
    }

    from sklearn.model_selection import GridSearchCV
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)

    # Best parameter set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    y_true, y_pred = y_test, clf.predict(X_test)

    from sklearn.metrics import classification_report
    print('Results on the test set:')
    print(classification_report(y_true, y_pred))


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

    # One-hot encode chest pain
    df['cp'][df['cp'] == 0] = 'typical angina'
    df['cp'][df['cp'] == 1] = 'atypical angina'
    df['cp'][df['cp'] == 2] = 'non-anginal pain'
    df['cp'][df['cp'] == 3] = 'asymptomatic'
    df['cp'] = df['cp'].astype('object')
    df = pd.get_dummies(df)

    pd.set_option('display.max_columns', 25)
    pd.set_option('display.width', 200)
    print(df.head(20))

    # This takes a long time, but goes through an exhaustive search over different parameter values:
    # parameter_optimization(df)

    # Get data values
    shuffled_data = df.sample(frac=1)
    data = shuffled_data.drop('target', axis=1)
    target = shuffled_data['target'].values
    data_values = feature_scaling(data)


    # PCA
    # run_pca(transformed_data)

    # comment this out for now
    # reduced_data, feature_index = feature_reduction(transformed_data, target)
    # print(reduced_data.shape)

    # print the list of features that were selected from the feature reduction
    # reduced_features = []
    # feature_names = list(data.columns.values)
    # for was_selected, feature in zip(feature_index, feature_names):
    #     if was_selected:
    #         reduced_features.append(feature)

    X_train, X_test, y_train, y_test = train_test_split(data_values, target,
                                                        test_size=0.33, random_state=1)

    clf = MLPClassifier(activation='tanh', alpha=0.0001, hidden_layer_sizes=(5, 3, 2), learning_rate='constant',
                        learning_rate_init=0.1, solver='sgd')

    clf.fit(X_train, y_train)

    print("Training set score: %f" % clf.score(X_train, y_train))
    print("Test set score: %f" % clf.score(X_test, y_test))

    # print('Selected Features:')
    # print(str(reduced_features))

    print('Done')


