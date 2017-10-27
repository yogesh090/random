import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

# Regularized Greedy Forest
from rgf.sklearn import RGFClassifier     # https://github.com/fukatani/rgf_python

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Preprocessing (Baseline)
id_test = test['id'].values
print("data loaded")

print(train.shape)
#print(train.info())
print(train.columns)
print(test.shape)

#Use only required columns, from xgb feature importance
train_features = [
    "ps_car_13",  #            : 1571.65 / shadow  609.23
    "ps_reg_03",  #            : 1408.42 / shadow  511.15
    "ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
    "ps_ind_03",  #            : 1219.47 / shadow  230.55
    "ps_ind_15",  #            :  922.18 / shadow  242.00
    "ps_reg_02",  #            :  920.65 / shadow  267.50
    "ps_car_14",  #            :  798.48 / shadow  549.58
    "ps_car_12",  #            :  731.93 / shadow  293.62
    "ps_car_01_cat",  #        :  698.07 / shadow  178.72
    "ps_car_07_cat",  #        :  694.53 / shadow   36.35
    "ps_ind_17_bin",  #        :  620.77 / shadow   23.15
    "ps_car_03_cat",  #        :  611.73 / shadow   50.67
    "ps_reg_01",  #            :  598.60 / shadow  178.57
    "ps_car_15",  #            :  593.35 / shadow  226.43
    "ps_ind_01",  #            :  547.32 / shadow  154.58
    "ps_ind_16_bin",  #        :  475.37 / shadow   34.17
    "ps_ind_07_bin",  #        :  435.28 / shadow   28.92
    "ps_car_06_cat",  #        :  398.02 / shadow  212.43
    "ps_car_04_cat",  #        :  376.87 / shadow   76.98
    "ps_ind_06_bin",  #        :  370.97 / shadow   36.13
    "ps_car_09_cat",  #        :  214.12 / shadow   81.38
    "ps_car_02_cat",  #        :  203.03 / shadow   26.67
    "ps_ind_02_cat",  #        :  189.47 / shadow   65.68
    "ps_car_11",  #            :  173.28 / shadow   76.45
    "ps_car_05_cat",  #        :  172.75 / shadow   62.92
    "ps_calc_09",  #           :  169.13 / shadow  129.72
    "ps_calc_05",  #           :  148.83 / shadow  120.68
    "ps_ind_08_bin",  #        :  140.73 / shadow   27.63
    "ps_car_08_cat",  #        :  120.87 / shadow   28.82
    "ps_ind_09_bin",  #        :  113.92 / shadow   27.05
    "ps_ind_04_cat",  #        :  107.27 / shadow   37.43
    "ps_ind_18_bin",  #        :   77.42 / shadow   25.97
    "ps_ind_12_bin",  #        :   39.67 / shadow   15.52
    "ps_ind_14",  #            :   37.37 / shadow   16.65
]

#create X,y,T
X = train[train_features].values
y = train.loc[:,'target'].values
T = test[train_features].values

#create the models
# LightGBM params
lgb_params_1 = {
    'learning_rate': 0.01,
    'n_estimators': 1250,
    'max_bin': 10,
    'subsample': 0.8,
    'subsample_freq': 10,
    'colsample_bytree': 0.8, 
    'min_child_samples': 500
}

lgb_params_2 = {
    'learning_rate': 0.005,
    'n_estimators': 3700,
    'subsample': 0.7,
    'subsample_freq': 2,
    'colsample_bytree': 0.3,  
    'num_leaves': 16
}

lgb_params_3 = {
   'objective':'binary:logistic',
   'learning_rate':0.02,
    'n_estimators':1000,
    'max_depth':4,
    'subsample':0.9,
    'colsample_bytree':0.9,  
    'min_child_weight':10
}

lgb_params_4 = {
   'objective':'binary:logistic',
   'learning_rate':0.02,
    'n_estimators':1000,
    'max_depth':4,
    'subsample':0.9,
    'colsample_bytree':0.9,  
    'min_child_weight':10
}

lgb_model_1 = LGBMClassifier(**lgb_params_1)
lgb_model_2 = LGBMClassifier(**lgb_params_2)
lgb_model_3 = XGBClassifier(**lgb_params_3)
#base_models = (lgb_model_1, lgb_model_2, lgb_model_3)
base_models = (lgb_model_1, lgb_model_2)

log_model = LogisticRegression()
stacker = log_model
print("models created")

#now we have the data with equal set of positives and negatives
#lets check cross validation scores
n_splits=3
folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=15).split(X, y))

S_train = np.zeros((X.shape[0], len(base_models)))
S_test = np.zeros((T.shape[0], len(base_models)))
for i, clf in enumerate(base_models):
    S_test_i = np.zeros((T.shape[0], n_splits))
    for j, (train_idx, test_idx) in enumerate(folds):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_holdout = X[test_idx]
        print(X_train.shape)
        # Get positive examples
        pos = pd.Series(y_train == 1)
        # Add positive examples
        X_train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_train[pos])])
        y_train = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_train[pos])])
        # Shuffle data
        idx = np.arange(len(X_train))
        np.random.shuffle(idx)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]

        print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_holdout)[:,1]                

        S_train[test_idx, i] = y_pred
        S_test_i[:, j] = clf.predict_proba(T)[:,1]
        
    S_test[:, i] = S_test_i.mean(axis=1)

results = cross_val_score(stacker, S_train, y, cv=5, scoring='roc_auc')
print("Stacker score: %.5f" % (results.mean()))
print(results)
print("S train size is : ", S_train.shape)
stacker.fit(S_train, y)
res = stacker.predict_proba(S_test)[:,1]

print(res)

print(results)
print(res)

sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = res
sub.to_csv('stacked_result_strat_upsample.csv', index=False)

print('completed')