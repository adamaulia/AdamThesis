import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns 
from xgboost import XGBRegressor
from sklearn.svm import SVR
import xgboost as xgb
from datetime import date
from scipy.stats import skew
from sklearn.model_selection import cross_validate
from scipy import stats

def model_perform(X_train,y_train, X_test, y_test, model, name, verbose = 0 ):
    # train 
    model.fit(X_train, y_train)

    # test 
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)


    if verbose == 1 :
        print(' MAE {} '.format(mean_absolute_error(y_pred_test,y_test)))
        print(' MSE {} '.format(mean_squared_error(y_pred_test,y_test)))
        print(' R2 {} '.format(r2_score(y_train,y_pred_train)))
    else : 
        pass 
    
    result = {}
    result['mae'] = np.round(mean_absolute_error(y_pred_test,y_test),5)
    result['mse'] = np.round(mean_squared_error(y_pred_test,y_test),5)
    result['R2'] = np.round(r2_score(y_train,y_pred_train),5)
    result['name'] = name
    result['feature_size'] = X_train.shape[1]
    result['train_size'] = X_train.shape[0]
    result['test_size'] = X_test.shape[0]
    result['pearson'] = np.round(stats.pearsonr(y_pred_test, y_test)[0],5)

    return model, y_pred_test, y_pred_train, result 

def show_error_pattern(y_pred, y_test):
    result_test = pd.DataFrame()
    result_test['score'] = y_pred
    result_test['type'] = 'predict'
    result_test['idx'] = np.arange(result_test.shape[0])

    result_test2 = pd.DataFrame()
    result_test2['score'] = np.squeeze(y_test)
    result_test2['type'] = 'test'
    result_test2['idx'] = np.arange(result_test2.shape[0])

    df_result = pd.concat([result_test2,result_test])

    sns.lineplot(data=df_result, x="idx", y='score', hue="type")


def bulk_train(df_input, drop_column, target_column, dataset_name,  rf_param, xgb_param, svr_param, verbose = 0,):

    error_report = []
    model_dict = {}


    # remove 0 rating
    df_input = df_input[df_input['rating']>0]

    # scaled the values 
    features_columns = df_input.drop(drop_column,axis=1).columns
    features_columns = list(features_columns) + [target_column]
    scaler = MinMaxScaler()
    df_input_scale = pd.DataFrame(scaler.fit_transform(df_input[features_columns]), columns = features_columns)
    


    # split train and test 
    X_train, X_test, y_train, y_test = train_test_split(df_input_scale.drop(target_column, axis=1), df_input_scale[target_column], test_size=0.12, random_state=42)

    if rf_param : 
        regr = RandomForestRegressor(**rf_param)
    else :
        regr = RandomForestRegressor(random_state=0)
    model_regr, y_pred_test_regr, y_pred_train_regr, result_rf   = model_perform(X_train, y_train, X_test, y_test, regr, name='rf', verbose = verbose )
    error_report.append(result_rf)

    if xgb_param:
        xgbr = XGBRegressor(**xgb_param)
    else :
        xgbr = XGBRegressor(random_state=0)

    model_xgbr, y_pred_test_xgbr, y_pred_train_xgbr, result_xgb  = model_perform(X_train, y_train, X_test, y_test, xgbr, name='xgb', verbose = verbose)
    error_report.append(result_xgb)

    # SVR 
    if svr_param:
        svr = SVR(**svr_param)
    else :
        svr = SVR(kernel='poly')
        
    model_svr, y_pred_test_svr, y_pred_train_svr, result_svr  = model_perform(X_train, y_train, X_test, y_test, svr, name='svr',verbose = verbose)
    error_report.append(result_svr)

    model_dict['rf'] = model_regr
    model_dict['xgbr'] = result_xgb
    model_dict['svr'] = result_svr
    
    df_report = pd.DataFrame(error_report)
    df_report['dataset'] = dataset_name
    return  model_dict, df_report


def bulk_train_k_fold(df_input, drop_column, target_column, dataset_name, rf_param, xgb_param, svr_param, verbose = 0):

        
    # remove 0 rating
    df_input = df_input[df_input['rating'] > 0]

    # scaled the values 
    features_columns = df_input.drop(drop_column,axis=1).columns
    features_columns = list(features_columns) + [target_column]
    scaler = MinMaxScaler()
    df_input_scale = pd.DataFrame(scaler.fit_transform(df_input[features_columns]), columns = features_columns)

    # split train and test 
    X_train, X_test, y_train, y_test = train_test_split(df_input_scale.drop(target_column, axis=1), df_input_scale[target_column], test_size=0.12, random_state=42)

    # cross validation 
    scoring = ['neg_mean_absolute_error','neg_mean_squared_error','r2']

    if rf_param:
        regr = RandomForestRegressor(**rf_param)
    else :
        regr = RandomForestRegressor(random_state=0)
        
    result_rf = cross_validation(regr, X_train, y_train, scoring = scoring, cv =5 )
    df_rf = pd.DataFrame(result_rf)
    df_rf['name'] = 'rf'

    if xgb_param:
        xgbr = XGBRegressor(**xgb_param)
    else :
        xgbr = XGBRegressor(random_state=0)

    result_xgb = cross_validation(xgbr, X_train, y_train, scoring = scoring, cv =5 )
    df_xgb = pd.DataFrame(result_xgb)
    df_xgb['name'] = 'xgb'


    if svr_param:
        svr = SVR(**svr_param)
    else :
        svr = SVR(kernel='poly')

    result_svr = cross_validation(svr, X_train, y_train, scoring = scoring, cv =5 )
    df_svr = pd.DataFrame(result_svr)
    df_svr['name'] = 'svr'

    df_result = pd.concat([df_rf,df_xgb,df_svr])
    df_result['dataset'] = dataset_name

    # Singe train and test 
    single_result = []

    # random forest
    tmp_result = {} 
    regr.fit(X_train,y_train)
    y_pred = regr.predict(X_test)
    tmp_result['mae'] = mean_absolute_error(y_pred, y_test)
    tmp_result['mse'] = mean_squared_error(y_pred, y_test)
    tmp_result['pearson'] = stats.pearsonr(y_pred, y_test)[0]
    tmp_result['name'] = 'rf'
    tmp_result['dataset'] = dataset_name
    tmp_result['feature_size'] = len(X_train.columns)
    tmp_result['train_size'] = X_train.shape[0]
    tmp_result['test_size'] = X_test.shape[0]
    single_result.append(tmp_result)

    # random forest
    tmp_result = {} 
    xgbr.fit(X_train,y_train)
    y_pred = xgbr.predict(X_test)
    tmp_result['mae'] = mean_absolute_error(y_pred, y_test)
    tmp_result['mse'] = mean_squared_error(y_pred, y_test)
    tmp_result['pearson'] = stats.pearsonr(y_pred, y_test)[0]
    tmp_result['name'] = 'xgb'
    tmp_result['dataset'] = dataset_name
    tmp_result['feature_size'] = len(X_train.columns)
    tmp_result['train_size'] = X_train.shape[0]
    tmp_result['test_size'] = X_test.shape[0]
    single_result.append(tmp_result)

    #svr 
    tmp_result = {} 
    svr.fit(X_train,y_train)
    y_pred = svr.predict(X_test)
    tmp_result['mae'] = mean_absolute_error(y_pred, y_test)
    tmp_result['mse'] = mean_squared_error(y_pred, y_test)
    tmp_result['pearson'] = stats.pearsonr(y_pred, y_test)[0]
    tmp_result['name'] = 'svr'
    tmp_result['dataset'] = dataset_name
    tmp_result['feature_size'] = len(X_train.columns)
    tmp_result['train_size'] = X_train.shape[0]
    tmp_result['test_size'] = X_test.shape[0]
    single_result.append(tmp_result)

    df_result_singel = pd.DataFrame(single_result)


    return df_result, df_result_singel


def cross_validation(model, X, y, scoring, cv=5):

    results = cross_validate(estimator=model,
                               X=X,
                               y=y,
                               cv=cv,
                               scoring=scoring,
                               return_train_score=True)

    return results