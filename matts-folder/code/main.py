import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import scikitplot as skplt
import seaborn as sns
from scipy import interp
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


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
                               (7, 3), (8, 3), (8, 3, 2)],
        'activation': ['tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0001, 0.05, 0.01, 1, 10, 20],
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
    pca = PCA(n_components=10)
    pca.fit(x_scaled)
    x_pca = pca.transform(x_scaled)
    return x_pca
    # print("shape of X_pca", x_pca.shape)
    #
    # # Plotting the Cumulative Summation of the Explained Variance
    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of Components')
    # plt.ylabel('Variance (%)')  # for each component
    # plt.title('Explained Variance')
    # plt.show()

    # ex_variance = np.var(x_pca, axis=0)
    # ex_variance_ratio = ex_variance / np.sum(ex_variance)
    # print (ex_variance_ratio)

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
    # Store average data values
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    # sensitivities = []
    specificities = []
    roc_auc_scores = []

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(0, 5):
        shuffled_data = df.sample(frac=1)
        data = shuffled_data.drop('target', axis=1)
        target = shuffled_data['target'].values
        data_values = feature_scaling(data)
        # PCA
        # data_values = run_pca(data_values)

        # Feature Importance
        # feature_importance(data_values, target)

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
                                                            test_size=0.20, random_state=1)

        clf = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(8, 3), learning_rate='constant',
                            learning_rate_init=0.001, solver='sgd', max_iter=2000)

        clf.fit(X_train, y_train)

        y_true, y_pred = y_test, clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        weights = clf.coefs_
        np.savetxt('../documents/nn-weights.txt', weights, fmt='%s')

        print('------------------')
        print('Iteration', i + 1)
        print("Training set score: %f" % clf.score(X_train, y_train))

        # print('Results on the test set:')
        # print(classification_report(y_true, y_pred))

        conf_matrix = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        # ROC AUC Curve
        # if i == 1:
        #     skplt.metrics.plot_roc(y_true, y_proba)
        #     plt.show()
        #
        #     # Plot confusion matrix
        #     plt.figure(figsize=(10, 7))
        #     sns.heatmap(conf_matrix, annot=True)
        #     ax = plt.axes()
        #     ax.set_title('MLP Confusion Matrix')
        #     plt.show()

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        precision = precision_score(y_true, y_pred)
        print('\nPrecision:', precision)
        precisions.append(precision)

        recall = recall_score(y_true, y_pred)
        print('Recall: ', recall)
        recalls.append(recall)

        # sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
        # print('Sensitivity : ', sensitivity)
        # sensitivities.append(sensitivity)

        specificity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
        print('Specificity : ', specificity)
        specificities.append(specificity)

        roc_score = roc_auc_score(y_true, y_pred)
        print('\nROC AUC: ', roc_score)
        roc_auc_scores.append(roc_score)

        f1 = f1_score(y_true, y_pred)
        print('F1 Score: ', f1)
        f1_scores.append(f1)

        accuracy = accuracy_score(y_true, y_pred)
        accuracies.append(accuracy)
        print('Accuracy: ', accuracy)

        print('------------------\n')
        # print('Selected Features:')
        # print(str(reduced_features))

    # Print averages
    avg_precision = np.mean(precisions)
    print('Average Precision: ', avg_precision)
    avg_recall = np.mean(recalls)
    print('Average Recall: ', avg_recall)
    # avg_sensitivity = np.mean(sensitivities)
    # print('Average Sensitivity: ', avg_sensitivity)
    avg_specificity = np.mean(specificities)
    print('Average Specificity: ', avg_specificity)
    avg_roc_auc_score = np.mean(roc_auc_scores)
    print('\nAverage ROC AUC: ', avg_roc_auc_score)
    avg_f1_score = np.mean(f1_scores)
    print('Average F1 Score: ', avg_f1_score)
    avg_accuracy = np.mean(accuracies)
    print('Average Accuracy: ', avg_accuracy)

    # Plot ROC
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MLP ROC')
    plt.legend(loc="lower right")
    plt.show()


def heatmap_rf():
    conf_matrix = [[20, 2], [5, 34]]

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True)
    ax = plt.axes()
    ax.set_title('Random Forest Confusion Matrix')
    plt.show()

def heatmap_knn():
    conf_matrix = [[29, 4], [21,  7]]

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True)
    ax = plt.axes()
    ax.set_title('Fuzzy C-Means Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    np.random.seed(1)
    df = read_heart_data()

    # One-hot encode chest pain
    df['cp'][df['cp'] == 0] = 'typical angina'
    df['cp'][df['cp'] == 1] = 'atypical angina'
    df['cp'][df['cp'] == 2] = 'non-anginal pain'
    df['cp'][df['cp'] == 3] = 'asymptomatic'

    # One-hot encode thal
    df['thal'][df['thal'] == 1] = 'normal'
    df['thal'][df['thal'] == 2] = 'fixed defect'
    df['thal'][df['thal'] == 3] = 'reversable defect'
    df['thal'] = df['thal'].astype('object')

    # One-hot encode rest ecg
    df['restecg'][df['restecg'] == 0] = 'normal'
    df['restecg'][df['restecg'] == 1] = 'ST-T wave abnormality'
    df['restecg'][df['restecg'] == 2] = 'left ventricular hypertrophy'

    df = pd.get_dummies(df)

    pd.set_option('display.max_columns', 25)
    pd.set_option('display.width', 200)
    print(df.head(20))

    # This takes a long time, but goes through an exhaustive search over different parameter values:
    # parameter_optimization(df)

    train(df)
    heatmap_rf()
    heatmap_knn()

    print('Done')


