import pandas as pd
import numpy as np
import pickle
import datetime as dt
import os
from functions.model_testing import evaluate_prediction

X_columns = [
    'step', 'typeId', 'amount',
    'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
    'incoerenceBalanceOrig', 'incoerenceBalanceDest',
    'errorBalanceOrig', 'errorBalanceDest'
]
Y_col = 'isFraud'
INT_COLUMNS = ['step', 'isFraud', 'isFlaggedFraud']
FLOAT_COLUMNS = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
MODELS_PATH = 'models/'
NEWDATA_PATH = 'data/new_data/'
THRESHOLDS = {
    'Accuracy': 0.99,
    'Precision': 0.99,
    'Recall': 0.99,
}
LOOKBACK = 10


def get_recent_module(models_path=MODELS_PATH):
    return max(os.listdir(models_path))


def check_new_data(recent_model_name, newdata_path=NEWDATA_PATH):
    d = recent_model_name[-14: -4]
    recent_data_path = d+'.csv'
    return [p for p in os.listdir(newdata_path) if p > recent_data_path]


def add_features(df):
    types_dict = {'TRANSFER': 0, 'PAYMENT': 1, 'CASH_IN': 2, 'DEBIT': 3, 'CASH_OUT': 4}
    df['typeId'] = df['type'].apply(lambda x: types_dict[x])
    df['incoerenceBalanceOrig'] = np.where(
        df.oldbalanceOrg==df.newbalanceOrig, 
        np.where(df.amount!=0, 1, 0), 
        np.where(df.amount==0, 1, 0)
    )
    df['incoerenceBalanceDest'] = np.where(
        df.oldbalanceDest==df.newbalanceDest, 
        np.where(df.amount!=0, 1, 0), 
        np.where(df.amount==0, 1, 0)
    )
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    return df


def call_model_and_predict(row, models_path=MODELS_PATH):
    if type(row) == pd.core.series.Series:
        row = row.to_frame().T
        for col in INT_COLUMNS:
            row[col] = row[col].astype(int)
        for col in FLOAT_COLUMNS:
            row[col] = row[col].astype(float)
    model = pickle.load(open(models_path+get_recent_module(models_path), 'rb'))
    prediction = model.predict(add_features(row)[X_columns])
    if len(prediction) == 1:
        return prediction[0]
    return prediction
    

def update_model(**kwargs):
    """
    This process is assuming that new transaction data will be uploaded in coherent format in fonlder NEWDATA_PATH.
    The process:
    1. picks more recent version of the model in MODELS_PATH
    2. checks if there are transactions more recent than most recent model version
    3. evaluate performance of the most recent model version on new data
    4. if performance is above THRESHOLDS new model is just a copy of current version
    5. if berformance is below THRESHOLDS a training dataset is created using most recent data whithin LOOKBACK.
    6. a new XGBClassifier model is trained on this dataset and saved in MODELS_PATH
    """
    recent_model_name = get_recent_module(MODELS_PATH)
    new_data_list = check_new_data(recent_model_name, NEWDATA_PATH)
    print(new_data_list)
    if len(new_data_list) > 0:
        currentmodel = pickle.load(open(MODELS_PATH+recent_model_name, 'rb'))
        for newdata_path in new_data_list:
            d = newdata_path[-14: -4]
            print(d)
            df = pd.read_csv(NEWDATA_PATH+newdata_path)
            actuals = list(df[Y_col])
            predictions = list(currentmodel.predict(add_features(df)[X_columns]))
            row = evaluate_prediction(actuals, predictions)
            print(row)
            if (
                row[-3] < THRESHOLDS['Accuracy'] or row[-2] < THRESHOLDS['Precision'] or row[-1] < THRESHOLDS['Recall']
            ):
                data_list = os.listdir(newdata_path)
                if len(data_list) > 1:
                    for i in range(-min(LOOKBACK, len(data_list)), 0):
                        df = df.append(pd.read_csv(NEWDATA_PATH+data_list[i]))
                Y = df[Y_col]
                X = df[X_columns]
                weights = (Y == 0).sum() / (Y == 1).sum()
                new_model = XGBClassifier(max_depth = 4, scale_pos_weight = weights, n_jobs = 4)
                new_model.fit(X, Y)
                pickle.dump(new_model, open(MODELS_PATH+d+'.sav', 'wb'))
            else:
                pickle.dump(currentmodel, open(MODELS_PATH+d+'.sav', 'wb'))
    return
            
            
            
