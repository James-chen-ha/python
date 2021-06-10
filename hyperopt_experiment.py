# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:24:20 2020

@author: Lab408
"""
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
train_df = train_df [ train_df['building_id'] != 1099 ]
train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
building_df = pd.read_csv(DATA_PATH + 'building_metadata_forsmalldata.csv')
weather_df = pd.read_csv(DATA_PATH + 'weather_train_smalldata.csv')
##################主要功能程式#################
##Utility Functions##
# Original code from https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling by @aitude

def fill_weather_dataset(weather_df):
    
    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)
    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]

    missing_hours = []#
    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df,new_rows])

        weather_df = weather_df.reset_index(drop=True)           

    # Add new Features
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month
    
    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id','day','month'])

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])
    weather_df.update(air_temperature_filler,overwrite=False)
#Fill Cloud Coverage
    # Step 1
    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()
    # Step 2
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])

    weather_df.update(cloud_coverage_filler,overwrite=False)
#Fill Dew Temperature
    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])
    weather_df.update(due_temperature_filler,overwrite=False)
#Fill Sea level Pressure
    # Step 1
    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
    # Step 2
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])

    weather_df.update(sea_level_filler,overwrite=False)
#Fill Wind Direction
    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])
    weather_df.update(wind_direction_filler,overwrite=False)
#Fill Wind Speed
    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])
    weather_df.update(wind_speed_filler,overwrite=False)
#Fill precip_depth_1_hr
    # Step 1
    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
    # Step 2
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler,overwrite=False)
#Remove Extra Features
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
    
    # Encode Categorical Data編碼分類數據
    le = LabelEncoder()
    df["primary_use"] = le.fit_transform(df["primary_use"])#先進行計算再進行轉換
    
    return df
#######################################
##Missing Weather Data Handling
#Fill Weather Information
#using this kernel to handle missing weather information.(上面的程式)
weather_df = fill_weather_dataset(weather_df)

#Memory Reduction
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
##
##try
#def lightgbm_factory(argsDict):
#    argsDict = argsDict_tranform(argsDict)
#    categorical_features = ["building_id", "site_id", "meter", "primary_use", "weekend"]
#    params = {
#    "objective": "regression",
#    "boosting": "gbdt",
#    "num_leaves": 1469,
#    "learning_rate": 0.04,
#    "feature_fraction": 0.83,
#    "reg_lambda": 3.7,
#    "metric": "rmse",
#    "max_depth":  10,
#    "bagging_freq": 5,
#    "bagging_fraction": 0.69,
#    "num_boost_round": 1000,
#   "verbosity": -1,
#   'reg_alpha': 0.1,
#   'reg_lambda': 2  
#    } 
###       
folds = 2
seed = 1000
shuffle = True
kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)       
#kfold交叉驗證############k fold cv############################################
#categorical_features = ["building_id", "site_id", "meter", "primary_use", "weekend"]
models = []
for train_index,test_index in kf.split(features):
    train_features = features.loc[train_index]
    train_target = target.loc[train_index]
    test_features = features.loc[test_index]
    test_target = target.loc[test_index]
    #
    Y_train=train_target
    #
#    d_training = lgb.Dataset(train_features, label=train_target,categorical_feature=categorical_features, free_raw_data=False)
    #
    X_train=train_features
    #
#    d_test = lgb.Dataset(test_features, label=test_target,categorical_feature=categorical_features, free_raw_data=False)
#    model_lgb = lgb.train(params, train_set=d_training,  num_boost_round=1000, valid_sets=[d_training,d_test], verbose_eval=25, early_stopping_rounds=5)
#    models.append(model_lgb)
#    del train_features, train_target, test_features, test_target, d_training, d_test
    gc.collect()
###############################################################################
####################################特徵工程
import numpy as np
import pandas as pd

def GetNewDataByPandas():
    y = X_train       #m*n
    X = Y_train         #n*1
    return X, y
####################################數據分割

from sklearn.model_selection import train_test_split
# Read wine quality data from file
X, y = GetNewDataByPandas()

# split data to [[0.8,0.2],01]
X, X_predict, y, Y_predict = train_test_split(X, y, test_size=0.10, random_state=100)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

import lightgbm as lgb

train_data = lgb.Dataset(data=y_train,label=x_train)
test_data = lgb.Dataset(data=y_test,label=x_test)

#####################################hyperopt的参数空间
from hyperopt import fmin, tpe, hp, partial

# 自定义hyperopt的参数空间
#space = {"max_depth": hp.randint("max_depth", 15),#0~15隨機整數
#         "num_trees": hp.randint("num_trees", 300),#num_trees=num_iterations
#         'learning_rate': hp.uniform('learning_rate', 1e-3, 5e-1),#搜索1e-3, 5e-1之間的範圍
#         "bagging_fraction": hp.randint("bagging_fraction", 5),
#         "num_leaves": hp.randint("num_leaves", 1280),
#         "num_boost_round": hp.randint("num_boost_round", 2000),#num_trees=num_iterations=num_boost_round
#         'feature_fraction': hp.uniform('feature_fraction', 2e-2, 3e-1 ),
#         'min_split_gain':hp.uniform('min_split_gain', 1e-2, 2e-1),
#         "min_child_weight":hp.randint("min_child_weight", 50),
#         "reg_lambda":hp.randint("reg_lambda", 5),
#         "reg_alpha":hp.randint("reg_alpha", 5)
#         }
space = {"max_depth": hp.randint("max_depth", 10, 20),#0~15隨機整數
#space_2測試，不一定要用e        "num_trees": hp.randint("num_trees", 300),#num_trees=num_iterations
         'learning_rate': hp.uniform('learning_rate', .01 , .09),#搜索1e-3, 5e-1之間的範圍
         "bagging_fraction": hp.uniform("bagging_fraction", 0.1 , 0.9),
         "num_leaves": hp.randint("num_leaves", 1000, 2500),
         "num_boost_round": hp.randint("num_boost_round", 100, 3000),#num_trees=num_iterations=num_boost_round
         'feature_fraction': hp.uniform('feature_fraction', 0.3, 0.9 ),
         'min_split_gain':hp.uniform('min_split_gain', 0.009, 0.11),
         "min_child_weight":hp.randint("min_child_weight", 5, 60),
         "reg_lambda":hp.randint("reg_lambda",1 , 6),
         "reg_alpha":hp.randint("reg_alpha",1 , 6)
         }
#params_range = {
#                'num_leaves': (1000, 2500),
#                'feature_fraction': (0.3, 0.9),
#                'bagging_fraction': (0.1, 0.9),
#                'bagging_freq':(1, 9),
#                'max_depth': (10, 20),
#                'lambda_l1': (1, 6),
#                'lambda_l2': (1, 6),
#                'min_split_gain': (0.009, 0.11),
#                'min_child_weight': (5, 60),
#                'learning_rate' : (.01,.07)
#                }


def argsDict_tranform(argsDict, isPrint=False):#argsdict:參數作為字典、isPrint:可打印
    argsDict["max_depth"] = argsDict["max_depth"] + 5
    argsDict['num_boost_round'] = argsDict['num_boost_round'] + 150
    argsDict["learning_rate"] = argsDict["learning_rate"] * 0.02 + 0.05########改看看數字，看結果的變化
    argsDict["bagging_fraction"] = argsDict["bagging_fraction"] * 0.1 + 0.5
    argsDict["num_leaves"] = argsDict["num_leaves"] * 3 + 10
    #argsDict["num_boost_round"] = argsDict["num_boost_round"] + 100
    argsDict["feature_fraction"] = argsDict["feature_fraction"] * 0.2 + 0.05
    argsDict["min_split_gain"] = argsDict["min_split_gain"] * 00.2 + 0.05
    argsDict["min_child_weight"] = argsDict["min_child_weight"] + 5
    argsDict["reg_lambda"] = argsDict["reg_lambda"] + 1 
    argsDict["reg_alpha"] = argsDict["reg_alpha"] + 1
    
    if isPrint:
        print(argsDict)
    else:
        pass

    return argsDict
######################################lightgbm模型工厂用于生产我们需要的model，而分数获取器则是为了解耦。
######################################这样在实际的测试工作中可以更加方便地套用代码。
###################################這裡不一定要執行##############################
from sklearn.metrics import mean_squared_error

def lightgbm_factory(argsDict):
    argsDict = argsDict_tranform(argsDict)

    params = {#'nthread': -1,  # 进程数
              'max_depth': argsDict['max_depth'],  # 最大深度
              'num_boost_round': argsDict['num_boost_round'],  # 树的数量
              'eta': argsDict['learning_rate'],  # 学习率
              'bagging_fraction': argsDict['bagging_fraction'],  # 采样数
              'num_leaves': argsDict['num_leaves'],  # 终点节点最小样本占比的和
              'objective': 'regression',
              'feature_fraction': 0.8,  # 样本列采样
              'nmu_boost_round': 2000, 
              'reg_lambda': 2,
              'reg_alpha': 2,
              'bagging_seed': 1000,  # 随机种子,light中默认为100
              'min_split_gain': argsDict['min_split_gain'],
              'min_child_weight': argsDict['min_child_weight']
              }    
    params['metric'] = ['rmse']

    model_lgb = lgb.train(params, train_data, num_boost_round=2000, valid_sets=[test_data],early_stopping_rounds=100)

    return get_tranformer_score(model_lgb)

def get_tranformer_score(tranformer):

    model = tranformer
    prediction = model.predict(Y_predict, num_iteration=model.best_iteration)

    return mean_squared_error(X_predict, prediction)
###################################### 开始使用hyperopt进行自动调参
algo = partial(tpe.suggest, n_startup_jobs=1)
best = fmin(lightgbm_factory, space, algo=algo, max_evals=20, pass_expr_memo_ctrl=None)#利用fmin進行優化
#####################################結果
RMSE = lightgbm_factory(best)
print('best :', best)
###################################可以執行到這裡就好############################
print('best param after transform :')
argsDict_tranform(best,isPrint=True)
print('rmse of the best lightgbm:', np.sqrt(RMSE))
#plt.show()
