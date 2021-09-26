import json
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
# import catboost as ctb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data_preprocessing import main

with open('lightgbm_config.json') as json_file:
    lgb_config = json.load(json_file)

with open('xgboost_config.json') as json_file:
    xgb_config = json.load(json_file)

def get_test_train_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def lgb_train_loop(X, y):
    df = lgb.Dataset(X, y)
    lgb_model = lgb.train(lgb_config, df, num_boost_round=1500)

    return lgb_model

def xgb_train_loop(X,y):
    df = xgb.DMatrix(X, y)
    xgb_model = xgb.train(xgb_config, df, num_boost_round=200)

    return xgb_model

def get_ensemble_predictions(lgb_model, xgb_model, test, test_columns):
    xgb_preds = xgb_model.predict(xgb.DMatrix(test[test_columns]))
    lgb_preds = lgb_model.predict(test[test_columns])

    pred_xgb = np.exp(xgb_preds)
    pred_lgb = np.exp(lgb_preds)

    ensemble_preds = (0.6*pred_lgb + 0.4*pred_xgb)

    return ensemble_preds

def create_submission_df(test, ensemble_preds):
    sub_df = pd.DataFrame()
    sub_df['id'] = test.id
    sub_df['test_duration'] = ensemble_preds

    return sub_df

if __name__ == '__main__':

    # get preprocessed data
    X, y, test = main()
    test_columns = X.columns
    X_train, X_test, y_train, y_test = get_test_train_split(X, y)

    lgb_model = lgb_train_loop(X, y)
    xgb_model = xgb_train_loop(X, y)

    ensemble_preds = get_ensemble_predictions(lgb_model, xgb_model, test, test_columns)

    sub_df = create_submission_df(test, ensemble_preds)
    sub_df.to_csv('submission_{}.csv'.format(time.strftime("%Y%m%d%H%M")), index=False)