from Configuration import INPUT, FEATURE_SELECTION_TIME_RATIO,MODEL_AND_PARAMS_TIME_RATIO, \
    NUM_FEATURES, NUM_SAMPLES, COLORS
from main import anyTimeForwardSearch, filling_missing_values, normalization, getBestModel, getAccuracy
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time
import matplotlib.pyplot as plt

def findLabel(dataset):
    if dataset == "datasets/German.csv":
        return "Creditability"
    elif dataset == "datasets/water_potability.csv":
        return "Potability"
    elif dataset == "datasets/nba_logreg.csv":
        return "TARGET_5Yrs"
    elif dataset == "datasets/wine.csv":
        return "quality"
    elif dataset == "datasets/diabetes.csv":
        return "Outcome"
    elif dataset == "datasets/data-ori.csv":
        return "SOURCE"
    return "complication"


def GetBestModelAlgorithm(dataset, label, time_for_feature_selection, time_for_tunning_params):
    start_tunning = time.perf_counter()
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
    end_feature_selection_time = time.perf_counter()
    time_for_tunning_params += time_for_feature_selection-(end_feature_selection_time-start_tunning)
    print(f"----time for tunning----{time_for_tunning_params}")
    best_model, best_hyper_parameters, best_acc = getBestModel(train_x=train_x[best_subset],
                                                               train_y=train_y,
                                                               time_for_tunning_params= time_for_tunning_params
                                                               )
    print(f"time for choose models and params: {time.perf_counter() - start}")

    model = best_model(**best_hyper_parameters)
    model.fit(train_x[best_subset], train_y)
    # print(f"classification report: = {classification_report(model.predict(test_x[best_subset]), test_y)}")
    acc = getAccuracy(model.predict(test_x[best_subset]), test_y)
    return model , acc

def tunningParameters(dataset, total_time, factor_time):
    label = findLabel(dataset)
    time_for_feature_selection = total_time * FEATURE_SELECTION_TIME_RATIO * factor_time
    time_for_model_and_params = total_time * MODEL_AND_PARAMS_TIME_RATIO * (1-factor_time)

    #k_factor =  max(round(time_for_feature_selection / (0.003*NUM_SAMPLES[1] + 0.1 * NUM_FEATURES[1])), 1)
    model, acc = GetBestModelAlgorithm(dataset=dataset,label=label,
                         time_for_feature_selection=time_for_feature_selection,
                         time_for_tunning_params=time_for_model_and_params
                         )

    return model, acc


def makeGraph(x, y, title, x_label, y_label):
    plt.title(f"{title}")
    plt.plot(x, y, color=COLORS[0])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig("tunningtime/German.png", bbox_inches='tight', format="png")
    plt.show()


def main():
    dataset, total_time = INPUT
    k = 0
    ratio_set = [0.6,0.8]
    plt.title(f"Accuracy per time and ratio")
    plt.xlabel("time")
    plt.ylabel("accuracy")
    for ratio in ratio_set:
        test_acc = []
        iter_time = []
        for i in range(1, 10):
            model , acc = tunningParameters(dataset, total_time*i, ratio)
            test_acc.append(acc)
            iter_time.append(total_time * i)
        plt.plot(iter_time, test_acc, color=COLORS[k])
        k += 1
        print(f"Ratio {ratio} done!")

    plt.savefig("tunningtime/German.png", bbox_inches='tight', format="png")
    plt.show()

    #makeGraph(iter_time,test_acc,"Accuracy per tunning time","tunning time", "Test accuracy")




if __name__ == "__main__":
    main()