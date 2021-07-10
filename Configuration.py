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
INPUT = ("datasets/German.csv", 200)

#Time per each step in the algorithm
FEATURE_SELECTION_TIME_RATIO  =   0.5
MODEL_AND_PARAMS_TIME_RATIO   =   1-FEATURE_SELECTION_TIME_RATIO

NUM_SAMPLES = [2785, 1000, 768, 1600, 1340]
NUM_FEATURES = [9, 20, 8, 11, 19]