
from Configuration import INPUT, FEATURE_SELECTION_TIME_RATIO,MODEL_AND_PARAMS_TIME_RATIO, \
    NUM_FEATURES, NUM_SAMPLES, COLORS
from Configuration import MODELS, HYPER_PARAMS, START_PROB, PROB_FACTOR
import pandas as pd
import numpy as np
from main import anyTimeForwardSearch, filling_missing_values, normalization, getAccuracy
from TunningParamsModel import findLabel
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


import random

def get_best_alg(best_acc_per_model):
    best_acc = best_acc_per_model[0]
    best_alg_num = 0
    for i in range(1,len(MODELS)):
        if best_acc_per_model[i] > best_acc:
            best_acc = best_acc_per_model[i]
            best_alg_num = i

    return best_alg_num

def getBestModel(train_x, train_y, time_for_tunning_params):
    start_time = time.perf_counter()
    model_mapping, model_d_acc = search_params_mapping()
    best_acc_per_model = [0] * 8
    model_best_params = [0] * 8
    model_best_param_set = [0] * 8
    prob = START_PROB
    cycle_time = 0
    while time_for_tunning_params- (time.perf_counter()-start_time) > cycle_time:
        cycle_time_start = time.perf_counter()
        if random.random() < prob:
            best_model = random.randrange(len(MODELS))

            params_set = []
            for i in range(len(model_mapping[best_model])):
                params_set.append(random.randrange(model_mapping[best_model][i]))

            params_dict = {}
            for i, key in enumerate(HYPER_PARAMS[best_model][0].keys()):
                params_dict[key] = HYPER_PARAMS[best_model][0][key][params_set[i]]

            avg_acc = sum(cross_val_score(MODELS[best_model](**params_dict),train_x,train_y,scoring='accuracy', cv=3))/3
            if avg_acc > best_acc_per_model[best_model]:
                best_acc_per_model[best_model] = avg_acc
                model_best_params[best_model] = params_dict
                model_best_param_set[best_model] = params_set
        else:
            best_model = get_best_alg(best_acc_per_model)
            if model_best_param_set[best_model] != 0:
                max_param = max(model_d_acc[best_model])
                if max_param != 0 and random.random() < prob:
                    param_index_to_change = model_d_acc[best_model].index(max_param)
                else:
                    param_index_to_change = random.randrange(len(model_best_param_set[best_model]))
                params_set = model_best_param_set[best_model]
                if model_mapping[best_model][param_index_to_change] > model_best_param_set[best_model][param_index_to_change]+1:
                    params_set[param_index_to_change]+=1
                else:
                    params_set[param_index_to_change]-=1
                params_dict = {}
                for i, key in enumerate(HYPER_PARAMS[best_model][0].keys()):
                    params_dict[key] = HYPER_PARAMS[best_model][0][key][params_set[i]]
                avg_acc = sum(
                    cross_val_score(MODELS[best_model](**params_dict), train_x, train_y, scoring='accuracy', cv=3)) / 3
                if avg_acc > best_acc_per_model[best_model]:
                    d_acc = avg_acc - best_acc_per_model[best_model]
                    best_acc_per_model[best_model] = avg_acc
                    model_best_params[best_model] = params_dict
                    model_best_param_set[best_model] = params_set
                    model_d_acc[best_model][param_index_to_change] = d_acc

            else:
                params_set = []
                for i in range(len(model_mapping[best_model])):
                    params_set.append(random.randrange(model_mapping[best_model][i]))
                params_dict = {}
                for i, key in enumerate(HYPER_PARAMS[best_model][0].keys()):
                    params_dict[key] = HYPER_PARAMS[best_model][0][key][params_set[i]]
                avg_acc = sum(
                    cross_val_score(MODELS[best_model](**params_dict), train_x, train_y, scoring='accuracy', cv=3)) / 3
                if avg_acc > best_acc_per_model[best_model]:
                    best_acc_per_model[best_model] = avg_acc
                    model_best_params[best_model] = params_dict
                    model_best_param_set[best_model] = params_set


        prob*=PROB_FACTOR
        cycle_time = time.perf_counter() - cycle_time_start

    max_acc = max(best_acc_per_model)
    model_index = best_acc_per_model.index(max_acc)
    return MODELS[model_index], model_best_params[model_index], max_acc






def search_params_mapping():
    model_mapping = {}
    model_d_acc = {}
    for i in range(len(MODELS)):
        model_mapping[i] = []
        model_d_acc[i] = []
        for key in HYPER_PARAMS[i][0].keys():
            model_mapping[i].append(len(HYPER_PARAMS[i][0][key]))
            model_d_acc[i].append(0)

    return model_mapping, model_d_acc

def tunningParameters(dataset, total_time, factor_time):
    label = findLabel(dataset)
    time_for_feature_selection = total_time * FEATURE_SELECTION_TIME_RATIO * factor_time
    time_for_model_and_params = total_time * MODEL_AND_PARAMS_TIME_RATIO * (1-factor_time)

    model, acc = GetBestModelAlgorithm(dataset=dataset,label=label,
                         time_for_feature_selection=time_for_feature_selection,
                         time_for_tunning_params=time_for_model_and_params
                         )

    return model, acc


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
    print(f"time for choose features, models and params: {time.perf_counter() - start}")

    model = best_model(**best_hyper_parameters)
    model.fit(train_x[best_subset], train_y)
    # print(f"classification report: = {classification_report(model.predict(test_x[best_subset]), test_y)}")
    acc = getAccuracy(model.predict(test_x[best_subset]), test_y)
    return model , acc


def main():
    dataset, total_time = INPUT
    k = 0
    ratio_set = [0.1 * i for i in range(1,10)]


    # for ratio in ratio_set:
    #     test_acc = []
    #     iter_time = []
    #     for i in range(1, 6):
    #         model, acc = tunningParameters(dataset, total_time * i, ratio)
    #         test_acc.append(acc)
    #         iter_time.append(total_time * i)
    #     plt.plot(iter_time, test_acc, color=COLORS[k])
    #     k += 1
    #     print(f"Ratio {ratio} done!")

    for i in range(4,7):
        test_acc = []
        dataset_name = dataset.split("/")[1]
        plt.title(f"Dataset: {dataset_name} Accuracy per ratio, time = {total_time * i} sec")
        plt.xlabel("ratio")
        plt.ylabel("accuracy")
        for ratio in ratio_set:
            model, acc = tunningParameters(dataset, total_time * i, ratio)
            test_acc.append(acc)

        plt.plot(ratio_set, test_acc, color=COLORS[k])
        plt.savefig("accuracy_per_ratio/" + dataset_name + "_time_" +str(total_time * i) +  ".png", bbox_inches='tight', format="png")
        plt.show()
        k += 1







if __name__== "__main__":
    main()





