DATASETS=[ "datasets/nba_logreg.csv","datasets/water_potability.csv", "datasets/German.csv" ,"datasets/diabetes.csv" ,
           "datasets/wine.csv" ]

LABELS = ["TARGET_5Yrs","Potability","Creditability","Outcome",  "quality"]

COLORS = ['b' ,'g', 'r', 'c','m', 'k']

#Graph saving plots names
GRAPHS=[ "nba_logreg","water_potability", "German" ,"diabetes" ,
           "wine" ]

# true means feature selection after selecting parameters. false means feature selection before selecting parameters.
FEATURE_SELECTION_SCHEDULING = True

#anytime algorithm parameter
K= 12

#directory for saving results
DIRECTORY = "anytime_feature_selection_after/"