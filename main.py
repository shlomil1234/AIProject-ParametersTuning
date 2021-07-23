from Configuration import DATASETS,LABELS, COLORS, GRAPHS, \
    FEATURE_SELECTION_SCHEDULING, K, DIRECTORY, \
    TIME_GRAPH_DIRECTORY
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

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
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier

import random



def filling_missing_values(train, test):
    '''
    data imputation using mean
    :param train:
    :param validate:
    :param test:
    :return: updated train, validate, test
    '''
    float_columns = train.columns
    for col in float_columns:
        #dropping outliers to calculate more representative mean
        q_low = train[col].quantile(0.002)
        q_hi = train[col].quantile(0.998)
        train_col_filtered = train[col][(train[col] <= q_hi) & (train[col] >= q_low)]
        values = train_col_filtered.dropna().mean()
        train[col].fillna(value=values, inplace=True)
        test[col].fillna(value=values, inplace=True)

    return train, test


def anyTimeForwardSearch(train_x,train_y,model, hyper_params, k_factor, time_left = None):
    features = random.sample(list(train_x.columns), len(list(train_x.columns)))
    all_features = features.copy()

    selected_features = []

    max_acc, max_features = 0, []
    acc = 0
    for feature in features:
        min_acc = 1
        sampling_size = len(train_x.columns) // 6
        selected_features.append(feature)
        features_to_use = [fe for fe in all_features if fe not in selected_features]
        if len(features_to_use) < sampling_size:
            sampling_size = len(features_to_use)
        for k in range(k_factor):
            features_to_use = random.sample(features_to_use, len(features_to_use))
            selected_features += features_to_use[0:sampling_size]

            total_acc = sum(cross_val_score(model(**hyper_params), scoring='accuracy', X=train_x[selected_features], y=train_y,
                                    cv=5))/5
            if total_acc < min_acc:
                min_acc = total_acc
            if total_acc > max_acc:
                max_acc = total_acc
                max_features = selected_features.copy()
            for elem in features_to_use[0:sampling_size]:
                selected_features.remove(elem)


        if min_acc > acc:
            acc = min_acc
        else:
            selected_features.remove(feature)

    acc_max_features = sum(cross_val_score(model(**hyper_params), scoring='accuracy', X=train_x[max_features], y=train_y,
                                    cv=5))/5
    acc_selected_features = sum(cross_val_score(model(**hyper_params), scoring='accuracy', X=train_x[selected_features], y=train_y,
                                    cv=5))/5
    if acc_max_features < acc_selected_features:
        return selected_features, acc
    else:
        return max_features, max_acc


def findBestHpPerModel(model, train_x, train_y, hyper_params):

    scores = ['accuracy']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            model(), hyper_params, scoring=scores[0], cv=3
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

        return clf.best_score_, clf.best_params_ #validate_acc


def featureSelectionOnlyCombCheck(train_x, train_y, time_search, model, best_hyper_params):
    start = time.perf_counter()

    return featureSelectionAllCombinations(train_x,train_y,
                                    time_search,model, best_hyper_params)



def getBestSubsetPerModel(model, best_hyper_params, train_x,train_y,time_search):
    return featureSelection(train_x, train_y, time_search, model, best_hyper_params)




def drop_numerical_outliers(df, z_thresh=3):
    '''
     remove outliers according to z-score method
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

def normalization(train, test, y_label_name = "Creditability"):

    for col in train.columns:
        if col!=y_label_name:
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


def featureSelectionAllCombinations(train_x, train_y, time_left,model, hyper_params):
    start = time.perf_counter()
    time_for_period = time_left
    best_subset_features = None
    acc = 0
    all_subsets = []
    for i in range(1,min(10,len(train_x.columns))):
        all_subsets+=findSubsets(train_x.columns,i)

    while time.perf_counter() - start < time_for_period - 0.5:
        features = random.choice(all_subsets)
        total_acc = cross_val_score(model(**hyper_params),scoring='accuracy', X=train_x[list(features)],y=train_y, cv=5)
        if sum(total_acc)/5 > acc:
            acc = sum(total_acc)/5
            best_subset_features = list(features)

    return best_subset_features, acc

def forwardLocalSearch(train_x, train_y, model, hyper_params):
    features = random.sample(list(train_x.columns), len(list(train_x.columns)))

    selected_features = []

    acc = 0
    for feature in features:
        selected_features.append(feature)
        total_acc = cross_val_score(model(**hyper_params),scoring='accuracy', X=train_x[selected_features],y=train_y, cv=3)
        if sum(total_acc)/3 > acc:
            acc=sum(total_acc)/3
        else:
            selected_features.remove(feature)
    return selected_features, acc

def BackwardLocalSearch(train_x,train_y,model, hyper_params):
    selected_features = list(train_x.columns)
    features = random.sample(list(train_x.columns), len(list(train_x.columns)))
    acc = sum(cross_val_score(model(**hyper_params),scoring='accuracy', X=train_x[selected_features],y=train_y, cv=3))/3
    for feature in features:
        selected_features.remove(feature)
        total_acc = cross_val_score(model(**hyper_params),scoring='accuracy', X=train_x[selected_features],y=train_y, cv=3)
        if sum(total_acc)/3 > acc:
            acc = sum(total_acc)/3
        else:
            selected_features.append(feature)

    return selected_features, acc




def featureSelection(train_x, train_y, time_for_feature_selection, model =None, best_hyper_params = None):
    start = time.perf_counter()
    anytime_forward_subset, anytime_forward_acc = anyTimeForwardSearch(train_x,train_y,model, best_hyper_params, 3)
    print(f"anytime: {anytime_forward_acc} \n {anytime_forward_subset}")
    forward_best_subset, forward_acc = forwardLocalSearch(train_x,train_y,model, best_hyper_params)
    time_for_forward_pass = time.perf_counter() - start
    print(f"time for forwars pass = {time_for_forward_pass}")
    if time_for_feature_selection - time_for_forward_pass < time_for_forward_pass:
        return forward_best_subset
    backward_best_subset, backward_acc = BackwardLocalSearch(train_x,train_y,model, best_hyper_params)
    time_for_backward_pass = time.perf_counter() - (start + time_for_forward_pass)
    print(f"time for backward pass = {time_for_backward_pass}")
    all_comb_acc = 0
    if time_for_feature_selection-time_for_backward_pass-time_for_forward_pass > 15:
        all_comb_best_subset, all_comb_acc = featureSelectionAllCombinations(train_x,train_y,
                                                  time_for_feature_selection-time_for_forward_pass-time_for_backward_pass,model, best_hyper_params)
        print("enter to all combination feature selection")
    max_acc = max(forward_acc,backward_acc,all_comb_acc)
    print(f"forward acc = {forward_acc} backward acc = {backward_acc} allcomb acc = {all_comb_acc}")
    if(max_acc == forward_acc):
        print("forward")
        return forward_best_subset
    elif max_acc== backward_acc:
        print("backward")
        return backward_best_subset
    else:
        print("all comb")
        return all_comb_best_subset


def getBestModel(train_x,train_y, time_for_tunning_params, best_subset = None):
    if best_subset != None:
        train_x = train_x[best_subset]

    models = [KNeighborsClassifier, RandomForestClassifier, AdaBoostClassifier,
              svm.SVC , DecisionTreeClassifier, GaussianNB, GaussianProcessClassifier, MLPClassifier]

    DecisionTreeClassifier_HP = [{'criterion': ["gini", "entropy"], 'splitter': ["best", "random"],
                                  'min_samples_split': [2, 4], 'min_samples_leaf': [4, 6]
                                     , 'min_weight_fraction_leaf': [ 0.2, 0.4]}]

    GaussianNB_HP = [{}]
    GaussianProcessClassifier_HP = [{'random_state': [0], 'multi_class': ['one_vs_rest','one_vs_one'],
                                     'n_restarts_optimizer': [2,4], 'max_iter_predict': [50,75,100] }]
    MLPClassifier_HP = [{'activation': ['identity', 'logistic', 'tanh', 'relu'],
                         'learning_rate_init': [0.01, 0.02, 0.005,0.03,0.025], 'solver': ['sgd', 'adam'],
                         'max_iter': [10], 'hidden_layer_sizes': [(3, 3), (3, 2), (4, 2), (5, 2), (5, 3),(2,2)]}]

    KNeighborsClassifier_HP = [{'n_neighbors': [10, 20, 25, 30], 'weights': ["uniform", "distance"], 'algorithm': ["auto"," ball_tree",
                                                                                                                   "kd_tree", "brute"]}]

    RandomForestClassifier_HP = [{"max_depth": [200], 'min_samples_split': [2,4], 'max_features': ['auto', 'sqrt'],
                                  "random_state": [0], "n_estimators": [25,50,75], 'criterion': ["entropy", "gini"]}]

    AdaBoostClassifier_HP = [{'n_estimators': [25, 50, 75], 'learning_rate': [0.01,0.02,0.05,0.1]}]

    svm.SVC_HP = [{'kernel': ['rbf'], 'gamma': [1e-4],
                   'C': [2, 4, 6]}]
    best_acc = 0
    best_model = ""
    best_hyper_parameters = {}
    start = time.perf_counter()
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
        elif model == GaussianNB:
            hyper_parameters = GaussianNB_HP
        elif model == GaussianProcessClassifier:
            hyper_parameters = GaussianProcessClassifier_HP
        acc, best_hyper_params_per_model = findBestHpPerModel(model, train_x, train_y, hyper_parameters)
        end = time.perf_counter()
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_hyper_parameters = best_hyper_params_per_model
        if end-start > time_for_tunning_params - 10:
            break

    return  best_model, best_hyper_parameters, best_acc


def plotTimePerK(k_range, time_per_k, graph):
    plt.title(f"Time per K for data {GRAPHS[graph]}")
    plt.plot(k_range, time_per_k, color=COLORS[0])
    plt.xlabel("k")
    plt.ylabel("Feature selection time")
    plt.savefig(TIME_GRAPH_DIRECTORY + GRAPHS[graph], bbox_inches='tight', format="png")
    plt.show()


def BlackBoxModel(data_name =None, label=None, time_limit = 600, graph=None):
    start = time.perf_counter()
    df = pd.read_csv(data_name, sep=',', header=0)
    train, test = \
        np.split(df.sample(frac=1, random_state=1),
                 [int(.80 * len(df))])

    train, test = filling_missing_values(train, test)
    train, test = normalization(train, test,y_label_name=label)
    train_y, test_y = train[label], test[label]
    train_x, test_x = train.drop([label], axis=1), test.drop([label], axis=1)
    end =time.perf_counter()

    print(f"Time Left: {time_limit - (end-start)}")

    time_left = time_limit-time.perf_counter()
    # time_scale = range(20,61,5)
    time_scale = range(1,11)
    test_acc_list = []
    train_acc_list = []

    if FEATURE_SELECTION_SCHEDULING:
        best_model, best_hyper_parameters, best_acc = getBestModel(train_x=train_x, train_y=train_y)
    else:
        best_model, best_hyper_parameters = KNeighborsClassifier, {'n_neighbors': 50}

    time_per_k = []
    k_range = range(1,K)

    for i in k_range:
        start = time.perf_counter()
        best_subset, train_acc = anyTimeForwardSearch(train_x, train_y, model=best_model,
                                                      hyper_params=best_hyper_parameters, k_factor=i)
        time_per_k.append(time.perf_counter() - start)

        if FEATURE_SELECTION_SCHEDULING:
            model = best_model(**best_hyper_parameters)
        else:
            best_model, best_hyper_parameters, best_acc = getBestModel(train_x=train_x[best_subset], train_y=train_y)
            model = best_model(**best_hyper_parameters)

        model.fit(train_x[best_subset], train_y)
        print(f"classification report: = {classification_report(model.predict(test_x[best_subset]), test_y)}")
        test_acc_list.append(getAccuracy(model.predict(test_x[best_subset]), test_y))
        train_acc_list.append(train_acc)


        print(f"percents: {i/(K-1)*100}% done!")

    plotTimePerK(k_range, time_per_k,graph)

    return time_scale, train_acc_list, test_acc_list, model



def main():
    for i,data in enumerate(DATASETS):
        start = time.perf_counter()
        time_scale, train_acc_list, test_acc_list, _ = BlackBoxModel(data,LABELS[i], 300,i)
        print(f"TOTAL TIME: {time.perf_counter() - start}")
        plt.title("Accuracy per K")
        plt.plot(range(1,12), train_acc_list, color=COLORS[i])
        plt.plot(range(1,12), test_acc_list, color=COLORS[i+1])
        plt.xlabel("k")
        plt.ylabel("Accuracy")
        plt.savefig(DIRECTORY + GRAPHS[i] + ".png", bbox_inches='tight')
        plt.show()

        test_acc_list = [0] + [test_acc_list[k]- test_acc_list[0] for k in range(1,len(test_acc_list))]
        plt.plot(range(0,11), test_acc_list, color=COLORS[i])
        plt.title("Improvement according to the first run")
        plt.xlabel("k-1")
        plt.ylabel("Accuracy changed on test")
        plt.savefig( DIRECTORY + GRAPHS[i] +"_Improvement.png", bbox_inches='tight')
        plt.show()
        print(f"------------------Data {data} Done!-------------------")











if __name__== "__main__":
    main()