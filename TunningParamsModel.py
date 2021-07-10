from Configuration import INPUT, FEATURE_SELECTION_TIME_RATIO,MODEL_AND_PARAMS_TIME_RATIO, \
    NUM_FEATURES, NUM_SAMPLES
from main import anyTimeForwardSearch, filling_missing_values, normalization, getBestModel
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import time

def findLabel(dataset):
    if dataset == "datasets/German.csv":
        return "Creditability"
    #TODO: continute


def GetBestModel(dataset, label, time_for_feature_selection):
    df = pd.read_csv(dataset, sep=',', header=0)
    train, test = \
        np.split(df.sample(frac=1, random_state=1),
                 [int(.80 * len(df))])
    train, test = filling_missing_values(train, test)
    train, test = normalization(train, test, y_label_name=label)
    train_y, test_y = train[label], test[label]
    train_x, test_x = train.drop([label], axis=1), test.drop([label], axis=1)
    best_model, best_hyper_parameters = KNeighborsClassifier, {'n_neighbors': 50}
    flag = True
    k_factor = 1
    best_acc = 0
    best_subset = []
    s = time.perf_counter()
    last_period = 0
    while flag:
        if k_factor != 1:
            last_period = end - start
        start = time.perf_counter()
        temp_subset, train_acc = anyTimeForwardSearch(train_x, train_y, model=best_model,
                                           hyper_params=best_hyper_parameters, k_factor=k_factor)
        end = time.perf_counter()
        if end - s > time_for_feature_selection - last_period*3:
            flag = False
        if train_acc > best_acc:
            best_acc = train_acc
            best_subset = temp_subset.copy()
        last_period = end - start
        k_factor+=1

    print(f"time for feautre selection: {time.perf_counter() - s}")
    start = time.perf_counter()
    best_model, best_hyper_parameters, best_acc = getBestModel(train_x=train_x[best_subset], train_y=train_y)
    print(f"time for choose models and params: {time.perf_counter() - start}")

    model = best_model(**best_hyper_parameters)
    model.fit(train_x[best_subset], train_y)
    print(f"classification report: = {classification_report(model.predict(test_x[best_subset]), test_y)}")
    return model

def tunningParameters(dataset, total_time):
    label = findLabel(dataset)
    time_for_feature_selection = total_time * FEATURE_SELECTION_TIME_RATIO
    time_for_model_and_params = total_time * MODEL_AND_PARAMS_TIME_RATIO

    #k_factor =  max(round(time_for_feature_selection / (0.003*NUM_SAMPLES[1] + 0.1 * NUM_FEATURES[1])), 1)
    model = GetBestModel(dataset=dataset,label=label,time_for_feature_selection=time_for_feature_selection)





def main():
    dataset, total_time = INPUT
    tunningParameters(dataset, total_time)


if __name__ == "__main__":
    main()