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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import shap  #for SHAP values


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
        'hidden_layer_sizes': [(6,), (7,), (8,), (6, 3),  (6, 4), (5, 3, 2),
                               (7, 3), ],
        'activation': ['tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0001, 0.05, 0.01, 1, 10, 20],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01, 0.05, 0.1, 0.5],
        'early_stopping': [True, False]
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
    tree = ExtraTreesClassifier(n_estimators=200)
    tree.fit(df, target)
    model = SelectFromModel(tree, prefit=True)
    feature_index = model.get_support()
    data_new = model.transform(df)
    return [data_new, feature_index]


def feature_importance(x, y):
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(x, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(x.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(x.shape[1]), indices)
    plt.xlim([-1, x.shape[1]])
    plt.show()
    print('Done')


def feature_scaling(df):
    # Use StandardScaler to provide Gaussian distribution of data (0 centering and 1 standard deviation)
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler.transform(df)


def train(df):
    # Get data values
    f1_scores = []
    accuracies = []

    for i in range(0, 10):
        shuffled_data = df.sample(frac=1)
        data = shuffled_data.drop('target', axis=1)
        target = shuffled_data['target'].values
        data_values = feature_scaling(data)

        # Feature Importance
        # feature_importance(data_values, target)

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
                                                            test_size=0.30, random_state=1)

        clf = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(7, 3), learning_rate='constant',
                            learning_rate_init=0.001, solver='sgd', max_iter=2000)

        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)

        print('------------------')
        print('Iteration', i + 1)
        from sklearn.metrics import classification_report
        print("Training set score: %f" % clf.score(X_train, y_train))

        print('Results on the test set:')
        print(classification_report(y_true, y_pred))

        conf_matrix = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
        print('\nSensitivity : ', sensitivity)

        specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
        print('Specificity : ', specificity)

        f1 = f1_score(y_true, y_pred)
        print('F1 Score: ', f1)
        f1_scores.append(f1)

        accuracy = accuracy_score(y_true, y_pred)
        accuracies.append(accuracy)
        print('Accuracy: ', accuracy)


        print('------------------\n')
        # print('Selected Features:')
        # print(str(reduced_features))

    avg_f1_score = np.mean(f1_scores)
    print('Average F1 Score: ', avg_f1_score)
    avg_accuracy = np.mean(accuracies)
    print('Average Accuracy: ', avg_accuracy)

if __name__ == "__main__":
    np.random.seed(1)
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

    train(df)


    print('Done')


