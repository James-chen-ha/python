import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns##data visualization資料視覺化
import warnings
import gc##garbage collector interface
warnings.simplefilter('ignore')
matplotlib.rcParams['figure.dpi'] = 100
sns.set()
building = pd.read_csv(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle/building_metadata_forsmalldata.csv')#load data
weather_train = pd.read_csv(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle/weather_train_smalldata.csv')
weather_test = pd.read_csv(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle/weather_test_smalldata.csv')
train = pd.read_csv(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle/small_data_train_energy.csv')
train.head()
#test_1 = pd.read_csv('C:/Users/fishi_000/Desktop/try_model_ashrae_energy_prediction_kaggle/test_smallest_data.csv')
test = pd.read_csv(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle/test_smallest_data.csv')
##1/102020
#temp=
building.isnull().any()
weather_train.isnull().any()
weather_test.isnull().any()
train.isnull().any()
test.isnull().any()
train.head()
##merge the dataset
train = train.merge(building, on='building_id', how='left')#merge用法( ,on= , how= )
test = test.merge(building, on='building_id', how='left')
train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')
test = test.merge(weather_test, on=['site_id', 'timestamp'], how='left')
del weather_train, weather_test, building #删除变量
gc.collect();
##
# Saving some memory
d_types = {'building_id': np.int16,
          'meter': np.int8,
          'site_id': np.int8,
          'primary_use': 'category',
          'square_feet': np.int32,
          'year_built': np.float16,
          'floor_count': np.float16,
          'air_temperature': np.float32,
          'cloud_coverage': np.float16,
          'dew_temperature': np.float32,
          'precip_depth_1_hr': np.float16,
          'sea_level_pressure': np.float32,
          'wind_direction': np.float16,
          'wind_speed': np.float32}

#np.dtype(np.int16)
for feature in d_types:
    train[feature] = train[feature].astype(d_types[feature])
    test[feature] = test[feature].astype(d_types[feature])
    
train["timestamp"] = pd.to_datetime(train["timestamp"])
test["timestamp"] = pd.to_datetime(test["timestamp"])
gc.collect();
################
#########lGBM##################
import gc
import os
from pathlib import Path
import random
import sys
from tqdm import tqdm_notebook as tqdm
from os.path import join as pjoin
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import display, HTML
# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
### --- models ---
from sklearn import preprocessing
from scipy import sparse 
from sklearn.model_selection import KFold
#from sklearn.cross_validation import KFold
import lightgbm as lgb
#import xgboost as xgb
#import catboost as cb123
##
##
#####To input to LGBM######
##The idea is to align the timestamp by peak air temperature, 
##based on an assumption that the highest air temperature should appear at around 14:00.
##After aligning the timestamp of weather data, LB jumped from 1.12 to 1.11.
##(這個想法是根據最高氣溫應在14:00左右出現的假設，根據最高氣溫來調整時間戳。)
##(調整了天氣數據的時間戳後，LB從1.12跳至1.11。)
#RAW_DATA_DIR = '/kaggle/input/ashrae_energy_prediction_kagglecompelition/'
#RAW_DATA_DIR = '/Users/fishi_000/Desktop/'
#RAW_DATA_DIR ='/Users/fishi_000/Desktop/ashrae_energy_prediction_kagglecompelition/'
#r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle
RAW_DATA_DIR = '/Users/Lab408/Desktop/try_model_ashrae_energy_prediction_kaggle'
#C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle
#C:\Users\Lab408\Desktop
#C:\Users\fishi_000\Desktop
weather_dtypes = {
    'site_id': np.uint8,
    'air_temperature': np.float32,
    'cloud_coverage': np.float32,
    'dew_temperature': np.float32,
    'precip_depth_1_hr': np.float32,
    'sea_level_pressure': np.float32,
    'wind_direction': np.float32,
    'wind_speed': np.float32,
}
weather_train = pd.read_csv(pjoin(RAW_DATA_DIR, 'weather_train_smalldata.csv'),dtype=weather_dtypes,
    parse_dates=['timestamp'])
weather_test = pd.read_csv(pjoin(RAW_DATA_DIR, 'weather_test_smalldata.csv'),dtype=weather_dtypes,
    parse_dates=['timestamp'])

weather = pd.concat([weather_train,weather_test],ignore_index=True)
del weather_train, weather_test
weather_key = ['site_id', 'timestamp']
temp_skeleton = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()
del weather    
data_to_plot = temp_skeleton.copy()
data_to_plot["hour"] = data_to_plot["timestamp"].dt.hour
count = 1
plt.figure(figsize=(25, 15))
for site_id, data_by_site in data_to_plot.groupby('site_id'):
    by_site_by_hour = data_by_site.groupby('hour').mean()
    ax = plt.subplot(4, 4, count)
    plt.plot(by_site_by_hour.index,by_site_by_hour['air_temperature'],'xb-')
    ax.set_title('site: '+str(site_id))
    count += 1
plt.tight_layout()
plt.show()
del data_to_plot
##
##如上圖所示，峰值溫度出現在不同時間的不同位置。 即使在夜晚也有很多，這沒有任何意義。 
##這意味著此處的時間戳數據不在本地時間。 
##由於能源消耗與當地時間有關，因此在使用時間戳之前，無需進行任何更改。 
##我們將計算峰值溫度時間與14:00之間的差異，假設這些結果是偏移量，然後將每個站點的時間戳與其對齊。

##calculate ranks of hourly temperatures within date/site_id chunks
##計算date / site_id塊中的每小時溫度等級(溫度變化)
temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.date])['air_temperature'].rank('average')
##create a dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)
##創建一個site_ids（0-16），在一天之內(0-23)，x平均小時溫度等級(溫度變化)
df_2d = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)
##Subtract the columnID of temperature peak by 14, getting the timestamp alignment gap.
##將溫度峰值的columnID減去14，得到時間戳對齊間隙。
site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)
site_ids_offsets.index.name = 'site_id'
def timestamp_align(df):
    df['offset'] = df.site_id.map(site_ids_offsets)
    df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))
    df['timestamp'] = df['timestamp_aligned']
    del df['timestamp_aligned']
    return df
##Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
##Modified to support timestamp type, categorical type(修改為時間戳類型，分類類型)
##Modified to add option to use float16 or not. feather format does not support float16.
##修改以添加選項以使用或不使用float16。羽毛(feather)格式不支持float16。
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type(跳過日期時間類型或分類類型)
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
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
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df

# Read data...
root = '../lab408/Desktop/try_model_ashrae_energy_prediction_kaggle'

train_df = pd.read_csv(os.path.join(root, 'small_data_train_energy.csv'))
weather_train_df = pd.read_csv(os.path.join(root, 'weather_train_smalldata.csv'))
test_df = pd.read_csv(os.path.join(root, 'test_smallest_data.csv'))
weather_test_df = pd.read_csv(os.path.join(root, 'weather_test_smalldata.csv'))
building_meta_df = pd.read_csv(os.path.join(root, 'building_metadata_forsmalldata.csv'))
sample_submission = pd.read_csv(os.path.join(root, 'sample_submission_smalldataa.csv'))
#C:\Users\fishi_000\Desktop\ashrae_energy_prediction_kagglecompelition
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
print(test_df)
weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])
weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])
#降低記憶體使用
reduce_mem_usage(train_df)
reduce_mem_usage(test_df)
reduce_mem_usage(building_meta_df)
reduce_mem_usage(weather_train_df)
reduce_mem_usage(weather_test_df)
#import feather
#path = 'train.feather'
#feather.write_dataframe(train_df, path)
#train_df = feather.read_dataframe(path)
##########
#########################Read data in feather format
train_df = pd.read_feather(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\small_data_train_energy.ftr')
weather_train_df = pd.read_feather(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\weather_train_smalldata.ftr')
test_df = pd.read_feather(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\test_smallest_data.ftr')
weather_test_df = pd.read_feather(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\weather_test_smalldata.ftr')
building_meta_df = pd.read_feather(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\building_metadata_forsmalldata.ftr')
sample_submission = pd.read_feather(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\sample_submission.ftr')
#########
##時間戳更加一致。只有站點14是不同的。(列出所有site_id一致後的時間標記)
##對應於line723
train_df['date'] = train_df['timestamp'].dt.date
train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])
building_meta_df[building_meta_df.site_id == 0]
train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
debug = False    
#preprocess(train_df)
##
building_site_dict = dict(zip(building_meta_df['building_id'], building_meta_df['site_id']))
site_meter_raw = train_df[['building_id', 'meter', 'timestamp', 'meter_reading']].copy()
site_meter_raw['site_id'] = site_meter_raw.building_id.map(building_site_dict)
del site_meter_raw['building_id']
site_meter_to_plot = site_meter_raw.copy()
site_meter_to_plot["hour"] = site_meter_to_plot["timestamp"].dt.hour
elec_to_plot = site_meter_to_plot[site_meter_to_plot.meter == 0]
##
count = 1
plt.figure(figsize=(25, 50))
for site_id, data_by_site in elec_to_plot.groupby('site_id'):
    by_site_by_hour = data_by_site.groupby('hour').mean()
    ax = plt.subplot(15, 4, count)
    plt.plot(by_site_by_hour.index,by_site_by_hour['meter_reading'],'xb-')
    ax.set_title('site: '+str(site_id))
    count += 1
plt.tight_layout()
plt.show()
del elec_to_plot, site_meter_to_plot, building_site_dict, site_meter_raw

###########start training LGBM? and preprossing..#############################
#print('lightgbm.__file__')
import lightgbm as lgb
#from sklearn import LGBMRegressor
#from lightgbm import LGBMRegressor
#python -c "import lightgbm; print(lightgbm.__version__)"123
#lgb.lightgbm
lgb.LGBMRegressor
def preprocess(df):
    df["hour"] = df["timestamp"].dt.hour
    df["weekend"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek
preprocess(train_df)
##add time feature
df_group = train_df.groupby('building_id')['meter_reading_log1p']
building_mean = df_group.mean().astype(np.float16)
building_median = df_group.median().astype(np.float16)
building_min = df_group.min().astype(np.float16)
building_max = df_group.max().astype(np.float16)
building_std = df_group.std().astype(np.float16)

train_df['building_mean'] = train_df['building_id'].map(building_mean)
train_df['building_median'] = train_df['building_id'].map(building_median)
train_df['building_min'] = train_df['building_id'].map(building_min)
train_df['building_max'] = train_df['building_id'].map(building_max)
train_df['building_std'] = train_df['building_id'].map(building_std)#.map用法?
building_mean.head()
building_max.head()
###
######
#Fill Nan value in weather dataframe by interpolation
#通過插值在天氣數據框中填充Nan值
#weather data has a lot of NaNs!!
#Tried to fill these values by interpolating data.
weather_train_df.head()
weather_train_df.isna().sum()
weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())
weather_train_df = weather_train_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())
#似乎nan的數量已通過插值減少，但某些屬性從未出現在特定的site_id中，而nan仍然保留了這些功能。
######feature engineering add feature#####
###Adding some lag feature
def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in cols:
        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]
        weather_df[f'{col}_max_lag{window}'] = lag_max[col]
        weather_df[f'{col}_min_lag{window}'] = lag_min[col]
        weather_df[f'{col}_std_lag{window}'] = lag_std[col]
add_lag_feature(weather_train_df, window=3)
add_lag_feature(weather_train_df, window=72)
weather_train_df.head()
weather_train_df.columns
####categorize primary_use column to reduce memory on merge...
###對primary_use進行分類，以減少合併時的內存
primary_use_list = building_meta_df['primary_use'].unique()
primary_use_dict = {key: value for value, key in enumerate(primary_use_list)} 
print('primary_use_dict: ', primary_use_dict)
building_meta_df['primary_use'] = building_meta_df['primary_use'].map(primary_use_dict)

gc.collect()
###
reduce_mem_usage(train_df, use_float16=True)
reduce_mem_usage(building_meta_df, use_float16=True)
reduce_mem_usage(weather_train_df, use_float16=True)
###
building_meta_df.head()
##
##類別與特徵列出來
category_cols = ['building_id', 'site_id', 'primary_use','hour', 'weekend', # 'month' , 'dayofweek'
    'building_median']  # , 'meter'
feature_cols = ['square_feet', 'year_built'] + [
    'hour', 'weekend', # 'month' , 'dayofweek'
    'building_median'] + [
    'air_temperature', 'cloud_coverage',
    'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
    'wind_direction', 'wind_speed', 'air_temperature_mean_lag72',
    'air_temperature_max_lag72', 'air_temperature_min_lag72',
    'air_temperature_std_lag72', 'cloud_coverage_mean_lag72',
    'dew_temperature_mean_lag72', 'precip_depth_1_hr_mean_lag72',
    'sea_level_pressure_mean_lag72', 'wind_direction_mean_lag72',
    'wind_speed_mean_lag72', 'air_temperature_mean_lag3', 
    'air_temperature_max_lag3', 
    'air_temperature_min_lag3', 'cloud_coverage_mean_lag3',
    'dew_temperature_mean_lag3', 
    'precip_depth_1_hr_mean_lag3', 'sea_level_pressure_mean_lag3',
    'wind_direction_mean_lag3', 'wind_speed_mean_lag3']
print(feature_cols)
print(category_cols)
###"""Train Light GBM model"""###
def create_X_y(train_df, target_meter):
    target_train_df = train_df[train_df['meter'] == target_meter]
    target_train_df = target_train_df.merge(building_meta_df, on='building_id', how='left')
    target_train_df = target_train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')
    X_train = target_train_df[feature_cols + category_cols]
    y_train = target_train_df['meter_reading_log1p'].values

    del target_train_df
    return X_train, y_train

def fit_lgbm(train, val, devices=(-1,), seed=None, cat_features=None, num_rounds=1000,lr=0.04, bf=0.69):
    #"""Train Light GBM model"""#
    X_train, y_train = train
    X_valid, y_valid = val
    metric = 'l2'
    ## specify your configurations as a dict:指定配置作為字典
    params = {'num_leaves': 1496,
              'objective': 'regression',
#              'max_depth':  9,
              'learning_rate': lr,
              "boosting": "gbdt",
              "bagging_freq": 5,
              "bagging_fraction": bf,
              "feature_fraction": 0.83,
              "metric": metric,
#               "verbosity": -1,
#               'reg_alpha': 0.1,
#               'reg_lambda': 2  
#              'device': 'gpu',
#              'gpu_platform_id': -1,
#              'gpu_device_id': -1
              }             
    device = devices[0]
    if device == -1:
       #print('cpu')
        # use cpu
        pass
    else:
         #use gpu
        print(f'using gpu device_id {device}...')
        params.update({'device': 'gpu', 'gpu_device_id': device})

    params['seed'] = seed

    early_stop = 50
    verbose_eval = 30

    d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features)
    watchlist = [d_train, d_valid]
    print('training LGB:')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)

    # predictions
    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    
    print('best_score', model.best_score)
    log = {'train/mae': model.best_score['training']['l2'],
           'valid/mae': model.best_score['valid_1']['l2']}
    return model, y_pred_valid, log

folds = 3
seed = 2000
shuffle = False
kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)
###############################################################################123123
target_meter = 0
X_train, y_train = create_X_y(train_df, target_meter=target_meter)
y_valid_pred_total = np.zeros(X_train.shape[0])
gc.collect()
print('target_meter', target_meter, X_train.shape)

cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
print('cat_features', cat_features)

models0 = []
for train_idx, valid_idx in kf.split(X_train, y_train):
    train_data = X_train.iloc[train_idx,:], y_train[train_idx]
    valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]

    print('train', len(train_idx), 'valid', len(valid_idx))
###model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])
    model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,
                                        num_rounds=1000, lr=0.04, bf=0.69)
    y_valid_pred_total[valid_idx] = y_pred_valid
    models0.append(model)
    gc.collect()
    if debug:
        break

sns.distplot(y_train)
del X_train, y_train
gc.collect()    
###
def plot_feature_importance(model):
    importance_df = pd.DataFrame(model.feature_importance(),
                                 index=feature_cols + category_cols,
                                 columns=['importance']).sort_values('importance')
    fig, ax = plt.subplots(figsize=(8, 8))
    importance_df.plot.barh(ax=ax)
    fig.show()
###
###############################################################################123123123
target_meter = 1
X_train, y_train = create_X_y(train_df, target_meter=target_meter)
y_valid_pred_total = np.zeros(X_train.shape[0])
gc.collect()
print('target_meter', target_meter, X_train.shape)

cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
print('cat_features', cat_features)

models1 = []
for train_idx, valid_idx in kf.split(X_train, y_train):
    train_data = X_train.iloc[train_idx,:], y_train[train_idx]
    valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]

    print('train', len(train_idx), 'valid', len(valid_idx))
#     model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])
    model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols, num_rounds=1000,
                                       lr=0.04, bf=0.69)
    y_valid_pred_total[valid_idx] = y_pred_valid
    models1.append(model)
    gc.collect()
    if debug:
        break

sns.distplot(y_train)
del X_train, y_train
gc.collect()    
###############################################################################
target_meter = 2
X_train, y_train = create_X_y(train_df, target_meter=target_meter)
y_valid_pred_total = np.zeros(X_train.shape[0])

gc.collect()
print('target_meter', target_meter, X_train.shape)

cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
print('cat_features', cat_features)

models2 = []
for train_idx, valid_idx in kf.split(X_train, y_train):
    train_data = X_train.iloc[train_idx,:], y_train[train_idx]
    valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]

    print('train', len(train_idx), 'valid', len(valid_idx))
#     model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])
    model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,
                                        num_rounds=1000, lr=0.04, bf=0.69)
    y_valid_pred_total[valid_idx] = y_pred_valid
    models2.append(model)
    gc.collect()
    if debug:
        break

sns.distplot(y_train)
del X_train, y_train
gc.collect()
###############################################################################
target_meter = 3
X_train, y_train = create_X_y(train_df, target_meter=target_meter)
y_valid_pred_total = np.zeros(X_train.shape[0])

gc.collect()
print('target_meter', target_meter, X_train.shape)

cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
print('cat_features', cat_features)

models3 = []
for train_idx, valid_idx in kf.split(X_train, y_train):
    train_data = X_train.iloc[train_idx,:], y_train[train_idx]
    valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]

    print('train', len(train_idx), 'valid', len(valid_idx))
###model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])
    model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols, num_rounds=1000,
                                       lr=0.04, bf=0.69)

    y_valid_pred_total[valid_idx] = y_pred_valid
    models3.append(model)
    gc.collect()
    if debug:
        break

sns.distplot(y_train)
del X_train, y_train
gc.collect()
###############################################################################
import pandas as pd
import datetime
import time
import calendar
import pytz
from datetime import datetime as dt
from datetime import date, datetime
print('loading...')
test_df = pd.read_feather(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\test_smallest_data.ftr')#使用line847~867之path_3
weather_test_df = pd.read_feather(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\weather_test_smalldata.ftr')#使用line847~867之path_4
print('preprocessing building...')
test_df = pd.read_csv(os.path.join(root, r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\test_smallest_data.csv'))
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])##先定義timestamp
test_df['date'] = test_df['timestamp'].dt.date##原本，(dt.date)datetime.date=date對象代表一個日期
##try//
preprocess(test_df)
test_df['building_mean'] = test_df['building_id'].map(building_mean)
test_df['building_median'] = test_df['building_id'].map(building_median)
test_df['building_min'] = test_df['building_id'].map(building_min)
test_df['building_max'] = test_df['building_id'].map(building_max)
test_df['building_std'] = test_df['building_id'].map(building_std)

print('preprocessing weather...')
#weather_test_df = timestamp_align(weather_test_df)--執行這行時，會出現:AttributeError: 'DataFrame' object has no attribute 'site_id'
#site_ids_offsets.index.name = 'site_id'
##try\\
weather_test_df = pd.read_feather(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\weather_test_smalldata.ftr')#使用line847~867之path_4
site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)
site_ids_offsets.index.name = 'site_id'
def timestamp_align(df):
    df['offset'] = df.site_id.map(site_ids_offsets)
    df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))
    df['timestamp'] = df['timestamp_aligned']
    del df['timestamp_aligned']
    return df
weather_test = pd.read_csv('C:/Users/Lab408/Desktop/try_model_ashrae_energy_prediction_kaggle/weather_test_smalldata.csv')
#weather_test_df = pd.read_csv('C:/Users/fishi_000/Desktop/try_model_ashrae_energy_prediction_kaggle/weather_test_smalldata.csv')
#weather_test_df = timestamp_align(weather_test_df)
import scipy 
weather_test_df = weather_test_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
weather_test_df.head()
weather_test_df.groupby('site_id').apply(lambda group: group.isna().sum())

##try//
#weather_test_df = timestamp_align(weather_test_df)
add_lag_feature(weather_test_df, window=3)
add_lag_feature(weather_test_df, window=72)

print('reduce mem usage...')
reduce_mem_usage(test_df, use_float16=True)
reduce_mem_usage(weather_test_df, use_float16=True)
gc.collect()
#sample_submission = pd.read_feather('sample_submission.feather')
#import os
#from os.path import join as pjoin
sample_submission = pd.read_csv(os.path.join(root, r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\sample_submission_smalldataa.csv'))
root = r'..\Desktop\try_model_ashrae_energy_prediction_kaggle\sample_submission.ftr'
sample_submission = pd.read_feather(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\sample_submission.ftr')
reduce_mem_usage(sample_submission)
sample_submission.to_feather('sample_submission.feather')

##
def create_X(test_df, target_meter):
    global target_test_df #global用法?
    target_test_df = test_df[test_df['meter'] == target_meter]
    print(target_test_df.shape)
    target_test_df = target_test_df.merge(building_meta_df, on='building_id', how='left')
    print(target_test_df.shape)
    target_test_df = target_test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')
    print(target_test_df.shape)
    X_test = target_test_df[feature_cols + category_cols]
    
    print(target_test_df)
    
    #return target_test_df.head()
    return X_test
create_X(test_df, target_meter=0)

def pred(X_test, models, batch_size=1000000):
    iterations = (X_test.shape[0] + batch_size -1) // batch_size
    print('iterations', iterations)

    y_test_pred_total = np.zeros(X_test.shape[0])
    for i, model in enumerate(models):
        print(f'predicting {i}-th model')
        for k in tqdm(range(iterations)):
            y_pred_test = model.predict(X_test[k*batch_size:(k+1)*batch_size], num_iteration=model.best_iteration)
            y_test_pred_total[k*batch_size:(k+1)*batch_size] += y_pred_test

    y_test_pred_total /= len(models)
    return y_test_pred_total
#from ipywidgets import IntProgress
############meter_0##############123
from ipywidgets import IntProgress
X_test = create_X(test_df, target_meter=0)
print(X_test)
gc.collect()
y_test0 = pred(X_test, models0)
sns.distplot(y_test0)
del X_test
gc.collect()
model.save_model('models0.txt', num_iteration=model.best_iteration) #存取模型
model.save_model(r'C:\Users\Lab408\Desktop\model_save\models0.txt', num_iteration=model.best_iteration) #存取模型，並指定生成路徑
model0 = lgb.Booster(model_file='models0.txt')#讀取模型
############meter_1##############123
X_test = create_X(test_df, target_meter=1)
print(X_test)
gc.collect()
y_test1 = pred(X_test, models1)
sns.distplot(y_test1)
del X_test
gc.collect()
model.save_model('models1.txt', num_iteration=model.best_iteration) #存取模型
model.save_model(r'C:\Users\Lab408\Desktop\model_save\models1.txt', num_iteration=model.best_iteration) #存取模型，並指定生成路徑
model1 = lgb.Booster(model_file='models1.txt')#讀取模型
############meter_2##############123
X_test = create_X(test_df, target_meter=2)
print(X_test)##why is empty?
gc.collect()
y_test2 = pred(X_test, models2)
sns.distplot(y_test2)
del X_test
gc.collect()
model.save_model('models2.txt', num_iteration=model.best_iteration) #存取模型
model.save_model(r'C:\Users\Lab408\Desktop\model_save\models2.txt', num_iteration=model.best_iteration) #存取模型，並指定生成路徑
model2 = lgb.Booster(model_file='models2.txt')#讀取模型
############meter_3##############
X_test = create_X(test_df, target_meter=3)
print(X_test)##why is empty?
gc.collect()
y_test3 = pred(X_test, models3)
sns.distplot(y_test3)
del X_test
gc.collect()
model.save_model('models3.txt', num_iteration=model.best_iteration) #存取模型
model.save_model(r'C:\Users\Lab408\Desktop\model_save\models3.txt', num_iteration=model.best_iteration) #存取模型，並指定生成路徑
model3 = lgb.Booster(model_file='models3.txt')#讀取模型
#################################
##submit
sample_submission.loc[test_df['meter'] == 0, 'meter_reading'] = np.expm1(y_test0)
sample_submission.loc[test_df['meter'] == 1, 'meter_reading'] = np.expm1(y_test1)
sample_submission.loc[test_df['meter'] == 2, 'meter_reading'] = np.expm1(y_test2)
sample_submission.loc[test_df['meter'] == 3, 'meter_reading'] = np.expm1(y_test3)
sample_submission.to_csv(r'C:\Users\Lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\sample_submission_smalldataa.csv', index=False, float_format='%.4f')
sample_submission_smalldataa=pd.read_csv(r'C:\Users\lab408\Desktop\try_model_ashrae_energy_prediction_kaggle\sample_submission_smalldataa.csv')
sample_submission_smalldataa.head()
np.log1p(sample_submission['meter_reading']).hist()
###列出重要的特徵
plot_feature_importance(models0[1])
plot_feature_importance(models1[1])
plot_feature_importance(models2[1])
plot_feature_importance(models3[1])
