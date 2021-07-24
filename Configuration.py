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

DATASETS=[ "datasets/water_potability.csv", "datasets/German.csv" ,"datasets/diabetes.csv" ,
           "datasets/wine.csv", "datasets/nba_logreg.csv" ]

LABELS = ["Potability","Creditability","Outcome", "quality", "TARGET_5Yrs"]

COLORS = ['b' ,'g', 'r', 'c','m', 'k']

#Graph saving plots names
GRAPHS=[ "water_potability", "German" ,"diabetes" ,
           "wine", "nba_logreg"]

# true means feature selection after selecting parameters. false means feature selection before selecting parameters.
FEATURE_SELECTION_SCHEDULING = False

#anytime algorithm parameter
K= 12

#directory for saving results
DIRECTORY = "anytime_feature_selection_first/"

TIME_GRAPH_DIRECTORY = "TimePerK/"

#Input for the the tunning model
INPUT = ("datasets/water_potability.csv", 50)

#Time per each step in the algorithm
FEATURE_SELECTION_TIME_RATIO  =   1
MODEL_AND_PARAMS_TIME_RATIO   =   1

NUM_SAMPLES = [2785, 1000, 768, 1600, 1340]
NUM_FEATURES = [9, 20, 8, 11, 19]

DecisionTreeClassifier_HP = [{'criterion': ["gini", "entropy"], 'splitter': ["best", "random"],
                              'min_samples_split': [2, 3, 4], 'min_samples_leaf': [2, 4, 6]
                                 , 'min_weight_fraction_leaf': [0.2, 0.4]}]

GaussianNB_HP = [{}]
GaussianProcessClassifier_HP = [{'random_state': [0], 'multi_class': ['one_vs_rest', 'one_vs_one'],
                                 'n_restarts_optimizer': [2, 3, 4], 'max_iter_predict': [50, 75, 100]}]
MLPClassifier_HP = [{'activation': ['identity', 'logistic', 'tanh', 'relu'],
                     'learning_rate_init': [0.01, 0.02, 0.005, 0.03, 0.025], 'solver': ['sgd', 'adam'],
                     'max_iter': [10], 'hidden_layer_sizes': [(3, 3), (3, 2), (4, 2), (5, 2), (5, 3), (2, 2)]}]

KNeighborsClassifier_HP = [
    {'n_neighbors': [10, 15, 20, 25, 30], 'weights': ["uniform", "distance"], 'algorithm': ["auto", "ball_tree",
                                                                                        "kd_tree", "brute"]}]

RandomForestClassifier_HP = [{"max_depth": [100,200,250], 'min_samples_split': [2,3, 4], 'max_features': ['auto', 'sqrt'],
                              "random_state": [0], "n_estimators": [25, 50, 75,100], 'criterion': ["entropy", "gini"]}]

AdaBoostClassifier_HP = [{'n_estimators': [25, 50, 75], 'learning_rate': [0.01, 0.015, 0.02, 0.05, 0.1]}]

SVC_HP = [{'kernel': ['rbf', 'sigmoid', 'poly', 'linear'], 'gamma': [1e-4], 'degree': [2,3,4], 'shrinking': [True, False],
               'C': [2, 4, 6]}]

MODELS = [KNeighborsClassifier, RandomForestClassifier, AdaBoostClassifier,
          svm.SVC, DecisionTreeClassifier, GaussianNB, GaussianProcessClassifier, MLPClassifier]

HYPER_PARAMS = [KNeighborsClassifier_HP, RandomForestClassifier_HP, AdaBoostClassifier_HP, SVC_HP,
                DecisionTreeClassifier_HP, GaussianNB_HP, GaussianProcessClassifier_HP, MLPClassifier_HP]

START_PROB = 0.9

PROB_FACTOR = 0.95