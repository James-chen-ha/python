# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:53:56 2020

@author: Lab408
"""
#123
#########################KFOLD_Lightgbm#######################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import datetime
import gc

DATA_PATH ='/Users/Lab408/Desktop/try_model_ashrae_energy_prediction_kaggle/'
##Load data
train_df = pd.read_csv(DATA_PATH + 'small_data_train_energy.csv')
# Remove outliers移除異常值
train_df = train_df [ train_df['building_id'] != 1099 ]#只要是Building_id=1099就刪掉
train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')#query:只留當前的
building_df = pd.read_csv(DATA_PATH + 'building_metadata_forsmalldata.csv')
weather_df = pd.read_csv(DATA_PATH + 'weather_train_smalldata.csv')
##################主要功能程式#################
##Utility Functions##
# Original code from https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling by @aitude

def fill_weather_dataset(weather_df):
    
    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"#(年-月-日 小時:分鐘:秒)
    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)
    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]
    missing_hours = []
    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df,new_rows])
        weather_df = weather_df.reset_index(drop=True) 
          
    
    # Add new Features，(Add Day,Week & Month Columns)
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month
    
    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id','day','month'])
    
#Fill air_temperature#########################################################
    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])
    weather_df.update(air_temperature_filler,overwrite=False)
#Fill Cloud Coverage##########################################################
    # Step 1
    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()
    # Step 2
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])#fillna
    weather_df.update(cloud_coverage_filler,overwrite=False)
#Fill Dew Temperature#########################################################
    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])
    weather_df.update(due_temperature_filler,overwrite=False)
#Fill Sea level Pressure######################################################
    # Step 1
    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
    # Step 2
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])

    weather_df.update(sea_level_filler,overwrite=False)
#Fill Wind Direction##########################################################
    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])
    weather_df.update(wind_direction_filler,overwrite=False)
#Fill Wind Speed##############################################################
    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])
    weather_df.update(wind_speed_filler,overwrite=False)
#Fill precip_depth_1_hr#######################################################
    # Step 1
    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
    # Step 2
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler,overwrite=False)
#Remove Extra Features########################################################
    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)
        
    return weather_df#改過的weather_df，改完後回傳回去
###############################################################################
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
      
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df

###############################################################################
def features_engineering(df):
    
    # Sort by timestamp按時間戳排序
    df.sort_values("timestamp")#
    df.reset_index(drop=True)#
    
    # Add more features
    df["timestamp"] = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S")#to_datetime用法轉換為python格式
    df["hour"] = df["timestamp"].dt.hour
    df["weekend"] = df["timestamp"].dt.weekday
    df['square_feet'] =  np.log1p(df['square_feet'])#log1p用法
    
    # Remove Unused Columns
    drop = ["timestamp","sea_level_pressure", "wind_direction", "wind_speed","year_built","floor_count"]##為何要刪掉這麼多??
    df = df.drop(drop, axis=1)#remove
    gc.collect()
    
    # Encode Categorical Data 分類數據的編碼
    le = LabelEncoder()
    df["primary_use"] = le.fit_transform(df["primary_use"])#先進行計算再進行轉換
    
    return df
#######################################
##Missing Weather Data Handling
#Fill Weather Information
#using this kernel to handle missing weather information.(上面的程式)
weather_df = fill_weather_dataset(weather_df)

#Memory Reduction減少記憶體
train_df = reduce_mem_usage(train_df,use_float16=True)
building_df = reduce_mem_usage(building_df,use_float16=True)
weather_df = reduce_mem_usage(weather_df,use_float16=True)
#Merge Data
#We need to add building and weather information into training dataset.
train_df = train_df.merge(building_df, left_on='building_id',right_on='building_id',how='left')#
train_df = train_df.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])
del weather_df
gc.collect()

#######Features Engineering########
train_df = features_engineering(train_df)
train_df.head(20)
#Features & Target Variables
target = np.log1p(train_df["meter_reading"])#target用log1p會比較好
features = train_df.drop('meter_reading', axis = 1)#
#del train_df
#return train_df
gc.collect()

######KFOLD LIGHTGBM Model###########
#from sklearn.model_selection import GridSearchCV
categorical_features = ["building_id", "site_id", "meter", "primary_use", "weekend"]#分類特徵
params = {#grid_search
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 3000,
    "learning_rate": 0.01,
    "feature_fraction": 0.7,
#    "reg_lambda": 3.7,
    "metric": "rmse",
    "max_depth":  80,
#    "bagging_freq": 5,
    "bagging_fraction": 0.1,
    "num_boost_round": 3100,
#   "verbosity": -1,
   'reg_alpha': 9,
   'reg_lambda': 9,
   'min_split_gain': 0.11,
   'min_child_weight': 70
}
 
params = {#hyperopt
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 2290,
    "learning_rate": 0.0506,
    "feature_fraction": 0.2186,
#    "reg_lambda": 3.7,
    "metric": "rmse",
    "max_depth":  24,
#    "bagging_freq": 5,
    "bagging_fraction": 0.517,
    "num_boost_round": 2144,
#   "verbosity": -1,
   'reg_alpha': 2,
   'reg_lambda': 4,
   'min_split_gain': 0.05219,
   'min_child_weight': 20
}

params = {#貝葉斯優化1
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 1476,
    "learning_rate": 0.0521,
    "feature_fraction": 0.8556,
#    "reg_lambda": 3.7,
    "metric": "rmse",
    "max_depth":  24,
#    "bagging_freq": 5,
    "bagging_fraction": 0.373,
    "num_boost_round": 100,#改看看變2500、3000
#   "verbosity": -1,
   'reg_alpha': 4.7264,
   'reg_lambda': 7.122,
   'min_split_gain': 0.077,
   'min_child_weight': 7.38
}
params = {#貝葉斯優化2
    "objective": "regression",
    "boosting": "gbdt",
    "num_leaves": 2500,
    "learning_rate": 0.01,
    "feature_fraction": 0.9,
#    "reg_lambda": 3.7,
    "metric": "rmse",
    "max_depth":  20,
#    "bagging_freq": 5,
    "bagging_fraction": 0.9,
    "num_boost_round": 3000,
#   "verbosity": -1,
   'reg_alpha': 1,
   'reg_lambda': 1,
   'min_split_gain': 0.009,
   'min_child_weight': 60
}
##TRY               
#params_range = {
#                'num_leaves': (40, 1500),
#                'feature_fraction': (0.7, 0.9),
#                'bagging_fraction': (0.1, 0.9),
#                'bagging_freq':(1, 9),
#                'max_depth': (10, 11),
#                'lambda_l1': (2, 5),
#                'lambda_l2': (2, 5),
#                'min_split_gain': (0.001, 0.1),
#                'min_child_weight': (5, 50),
#                'learning_rate' : (.01,.07)
#                }
#params = {
#    'max_depth': range(3, 8),
#    'num_leaves':range(50, 170)
#}
#gsearch1 = GridSearchCV(num_boost_round, params_range, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
#gsearch1.fit(df_train, y_train)
#gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
##TRY
folds = 3
seed = 2000
shuffle = False
kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)
#kfold交叉驗證############k fold cv############################################
models = []
for train_index,test_index in kf.split(features):#train、test做交叉驗證
    train_features = features.loc[train_index]
    train_target = target.loc[train_index]
    test_features = features.loc[test_index]
    test_target = target.loc[test_index]
    ##
    Y_train=train_target#grid search用
    ##
    d_training = lgb.Dataset(train_features, label=train_target,categorical_feature=categorical_features, free_raw_data=False)
    ##
    X_train=train_features#grid search用
    ##
    d_test = lgb.Dataset(test_features, label=test_target,categorical_feature=categorical_features, free_raw_data=False)
    
    model = lgb.train(params, train_set=d_training,  num_boost_round=3100, valid_sets=[d_training,d_test], verbose_eval=25, early_stopping_rounds=20)

    models.append(model)
    del train_features, train_target, test_features, test_target, d_training, d_test
    gc.collect()

#------------------------生產完資料後跑gsearch得到最佳化參數----------------------
#gsearch = GridSearchCV(params, param_grid=params_range, scoring='roc_auc', cv=3)#待觀察有無使用到

#model_lgb = lgb.LGBMRegressor(objective= "regression",
#    boosting= "gbdt",
#    num_leaves= 1469,
#    learning_rate= 0.04,
#    feature_fraction= 0.83,
#    metric= "rmse",
#    max_depth=  10,
#    bagging_freq= 5,
#    bagging_fraction= 0.69,
#    num_boost_round= 1000,)
#gsearch = GridSearchCV(estimator=model_lgb,  param_grid=params_range, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)                   
#gsearch.fit(資料Dataframe,類似陣列array的變數)
#gsearch.fit(X_train,Y_train)
#gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_
#=============================gsearch結束======================================
del features, target
gc.collect()
#############################Important Features################################
for model in models:
    lgb.plot_importance(model)
    plt.show()
####################Load Test Data#############################################
test_df = pd.read_csv(DATA_PATH + 'test_smallest_data.csv')
row_ids = test_df["row_id"]
test_df.drop("row_id", axis=1, inplace=True)#刪除
test_df = reduce_mem_usage(test_df)#減少記憶體容量
##Merge Building Data##########################################################
test_df = test_df.merge(building_df,left_on='building_id',right_on='building_id',how='left')#
del building_df
gc.collect()
##Fill Weather Information#####################################################
weather_df = pd.read_csv(DATA_PATH + 'weather_test_smalldata.csv')
weather_df = fill_weather_dataset(weather_df)
weather_df = reduce_mem_usage(weather_df)
#Merge Weather Data############################################################
test_df = test_df.merge(weather_df,how='left',on=['timestamp','site_id'])
del weather_df
gc.collect()
##Features Engineering#########################################################
test_df = features_engineering(test_df)
test_df.head(20)
##Prediction###################################################################
results = []
for model in models:
    if  results == []:
        results = np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)#expm1代表全部-1
    else:
        results += np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)#為什麼要除???每個model算出來的東西都不一樣，所以為了比較會除上model的數量?
    del model
    gc.collect()
    
del test_df, models
gc.collect()
############################submission#########################################
results_df = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(results, 0, a_max=None)})#生成dataframe
del row_ids,results
gc.collect()
results_df.to_csv(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\sample_submission_smalldataa.csv', index=False,float_format='%.4f')
results_df=pd.read_csv(r'C:\Users\lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\sample_submission_smalldataa.csv')
results_df.head(20)







