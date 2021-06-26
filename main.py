import pandas as pd
import numpy as np
from scipy import stats
import re
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
import itertools
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm






def findBestHpPerModel(model, train_x, train_y, hyper_params):
    # Set the parameters by cross-validation
    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                      'C': [1, 10, 100, 1000]},
    #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    scores = ['accuracy']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            model(), hyper_params, scoring=score
        )
        clf.fit(train_x, train_y)


        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()

        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        # y_true, y_pred = y_test, clf.predict(X_test)
        # print(classification_report(y_true, y_pred))
        # print()
        # validate_acc = accuracy_score(y_true, y_pred, normalize=True)
        # print(f'Accuracy on validation : {validate_acc}')

        return clf.best_score_, clf.best_params_ #validate_acc

















def drop_numerical_outliers(df, z_thresh=3):
    '''
     remove outliers according to the z-score method
    :param df :
    :param z_thresh:
    :return: updated df
    '''


    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh) \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)
    return df

def normalization(train, test):

    for col in train.columns:
        if col!="Creditability":
            train_min = train[col].min()
            train_max = train[col].max()
            train[col] = ((train[col] - train_min) / (train_max - train_min))
            test[col] = ((test[col] - train_min) / (train_max - train_min))

    return train, test

def getAccuracy(predicts, validate_y):
    size = validate_y.shape[0]
    acc = sum(predicts == validate_y) / size
    return acc



def findSubsets(s, n):
    return list(itertools.combinations(s, n))


def featureSelectionAllCombinations(train_x, train_y, time_left):
    time_for_period = time_left
    train_x_feature_selection, train_y_feature_selection = \
        train_x[:int(train_x.shape[0] // 1.2)], train_y[:int(train_x.shape[0] // 1.2)]
    validate_x_feature_selection, validate_y_feature_selection = \
        train_x[int(train_x.shape[0] // 1.2):], train_y[int(train_x.shape[0] // 1.2):]
    validate_acc = 0
    n = len(list(train_x.columns))//2
    best_subset_features = []
    for num_of_features in range(1,len(train_x.columns)):
        if num_of_features != 1:
            #print(num_of_features)
            time_for_last_period = time.perf_counter() - start
            time_for_period-=time_for_last_period
            #print(f"time for period {time_for_period}, time for last period {time_for_last_period}")
            if time_for_period < 0:#n*time_for_last_period:
                return best_subset_features, validate_acc
        start = time.perf_counter()
        selected_features = findSubsets(train_x.columns, num_of_features)
        for i in range(len(selected_features)):
            classifier = KNeighborsClassifier(n_neighbors=25)
            classifier.fit(train_x_feature_selection[list(selected_features[i])],
                           train_y_feature_selection)
            predicts = classifier.predict(validate_x_feature_selection[list(selected_features[i])])
            accuracy = getAccuracy(predicts, validate_y_feature_selection)
            if accuracy > validate_acc:
                validate_acc = accuracy
                best_subset_features = list(selected_features[i])

    return best_subset_features, validate_acc

def forwardLocalSearch(train_x, train_y):
    selected_features = []
    selected_features.append(train_x.columns[0])
    features = train_x.columns.drop(train_x.columns[0])
    train_x_feature_selection, train_y_feature_selection = \
        train_x[:int(train_x.shape[0]//1.2)] , train_y[:int(train_x.shape[0]//1.2)]
    validate_x_feature_selection, validate_y_feature_selection  = \
        train_x[int(train_x.shape[0]//1.2):], train_y[int(train_x.shape[0]//1.2):]

    classifier =KNeighborsClassifier(n_neighbors=25)
    classifier.fit(train_x_feature_selection[selected_features],
                   train_y_feature_selection)
    predicts = classifier.predict(validate_x_feature_selection[selected_features])
    accuracy = getAccuracy(predicts, validate_y_feature_selection)
    validate_acc = accuracy

    for feature in features:
        selected_features.append(feature)
        classifier = KNeighborsClassifier(n_neighbors=25)
        classifier.fit(train_x_feature_selection[selected_features],
                       train_y_feature_selection)
        predicts = classifier.predict(validate_x_feature_selection[selected_features])
        accuracy = getAccuracy(predicts,validate_y_feature_selection)
        if accuracy > validate_acc:
            validate_acc = accuracy
        else:
            selected_features.remove(feature)

    print(f"best subset {selected_features} accuracy: {validate_acc}")
    return selected_features, validate_acc


def BackwardLocalSearch(train_x,train_y):
    selected_features = list(train_x.columns)
    features = train_x.columns
    train_x_feature_selection, train_y_feature_selection = \
        train_x[:int(train_x.shape[0] // 1.2)], train_y[:int(train_x.shape[0] // 1.2)]
    validate_x_feature_selection, validate_y_feature_selection = \
        train_x[int(train_x.shape[0] // 1.2):], train_y[int(train_x.shape[0] // 1.2):]

    classifier = RandomForestClassifier(max_depth=10, min_samples_split=4, n_estimators=25)
    classifier.fit(train_x_feature_selection[selected_features],
                   train_y_feature_selection)
    predicts = classifier.predict(validate_x_feature_selection[selected_features])
    accuracy = getAccuracy(predicts, validate_y_feature_selection)
    validate_acc = accuracy

    for feature in features:
        selected_features.remove(feature)
        classifier = RandomForestClassifier(max_depth=10, min_samples_split=4, n_estimators=25)
        classifier.fit(train_x_feature_selection[selected_features],
                       train_y_feature_selection)
        predicts = classifier.predict(validate_x_feature_selection[selected_features])
        accuracy = getAccuracy(predicts, validate_y_feature_selection)
        if accuracy > validate_acc:
            validate_acc = accuracy
        else:
            selected_features.append(feature)

    print(f"best subset {selected_features} accuracy: {validate_acc}")
    return selected_features, validate_acc




def featureSelection(train_x, train_y, time_for_feature_selection):
    start = time.perf_counter()
    forward_best_subset, forward_acc = forwardLocalSearch(train_x,train_y)
    time_for_forward_pass = time.perf_counter() - start
    if time_for_feature_selection < 2.2 * time_for_forward_pass:
        return forward_best_subset
    backward_best_subset, backward_acc = BackwardLocalSearch(train_x,train_y)
    time_for_backward_pass = time.perf_counter() - (start + time_for_forward_pass)

    all_comb_best_subset, all_comb_acc = featureSelectionAllCombinations(train_x,train_y,
                                                  time_for_feature_selection-time_for_forward_pass-time_for_backward_pass)
    max_acc = max(forward_acc,backward_acc,all_comb_acc)
    print(time_for_feature_selection - start > 0)
    if(max_acc == forward_acc):
        print("forward")
        return forward_best_subset
    elif max_acc== backward_acc:
        print("backward")
        return backward_best_subset
    else:
        print("all_comb")
        return all_comb_best_subset


def BlackBoxModel(data_name =None, time_limit = 600):
    start = time.perf_counter()
    df = pd.read_csv(data_name, sep=',', header=0)
    train, test = \
        np.split(df.sample(frac=1, random_state=1),
                 [int(.85 * len(df))])

    #train = drop_numerical_outliers(train)
    train, test = normalization(train, test)
    train_y, test_y = train["Creditability"], test["Creditability"]
    train_x, test_x = train.drop(["Creditability"], axis=1), test.drop(["Creditability"], axis=1)
    end =time.perf_counter()

    print(f"Time Left: {time_limit - (end-start)}")

    time_left = time_limit-time.perf_counter()
    time_scale = range(1,100,2)
    acc_list = []
    for i in range(1,100,2):
        best_subset = featureSelection(train_x,train_y, i)
        models = [KNeighborsClassifier, RandomForestClassifier, AdaBoostClassifier, svm.SVC]#], MLPClassifier, DecisionTreeClassifier]
        KNeighborsClassifier_HP         = [{'n_neighbors': [10,25,50,75]}]


        DecisionTreeClassifier_HP       = [{'criterion':["gini", "entropy"], 'splitter' : ["best", "random"] ,
                                 'min_samples_split': [2,4,6,8], 'min_samples_leaf':[2,4,6]
                                 ,'min_weight_fraction_leaf': [0.2,0.4]}]


        MLPClassifier_HP                = [{'activation': ['identity', 'logistic', 'tanh', 'relu'],
                                            'learning_rate_init': [0.01,0.02, 0.005],'solver':[ 'sgd', 'adam'], 'max_iter': [10], 'hidden_layer_sizes': [(3,3),(3,2),(4,2),(5, 2), (5,3)]}]


        RandomForestClassifier_HP       = [{"max_depth": [200],'min_samples_split':[4,6],'max_features': ['auto', 'sqrt'],
                                            "random_state": [0], "n_estimators": [125,250],'criterion':["entropy","gini"]}]

        AdaBoostClassifier_HP           =  [{'n_estimators' : [25, 75], 'learning_rate' : [0.4, 0.8]}]

        svm.SVC_HP                      = [{'kernel': ['rbf'], 'gamma': [1e-4],
                                            'C': [2,4,6]}]
        best_acc = 0
        best_model = ""
        best_hyper_parameters = {}
        for model in models:
            hyper_parameters = []
            if model == KNeighborsClassifier:
                hyper_parameters = KNeighborsClassifier_HP
            elif model == RandomForestClassifier:
                hyper_parameters = RandomForestClassifier_HP
            elif model == MLPClassifier:
                hyper_parameters = MLPClassifier_HP
            elif model == DecisionTreeClassifier:
                hyper_parameters = DecisionTreeClassifier_HP
            elif model == AdaBoostClassifier:
                hyper_parameters = AdaBoostClassifier_HP
            elif model == svm.SVC:
                hyper_parameters = svm.SVC_HP

            acc, best_hyper_params_per_model  = findBestHpPerModel(model, train_x[best_subset],train_y, hyper_parameters)
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_hyper_parameters = best_hyper_params_per_model


        print(f"best model is {best_model}, acc: {best_acc}, best parameters: {best_hyper_parameters}")

        model = best_model(**best_hyper_parameters)
        model.fit(train_x,train_y)
        print(f"classification report: = {classification_report(model.predict(test_x), test_y)}")
        acc_list.append(getAccuracy(model.predict(test_x),test_y))
        print(acc_list[-1])

    plt.plot(time_scale, acc_list, color="b")
    plt.xlabel("time (sec)")
    plt.ylabel("Accuracy")
    plt.show()


def main():

    start = time.perf_counter()
    BlackBoxModel("German.csv", 300)
    print(f"TOTAL TIME: {time.perf_counter() - start}")













if __name__== "__main__":
    main()