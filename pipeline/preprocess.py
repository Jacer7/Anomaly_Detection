from io import StringIO
from pathlib import Path

import pandas as pd
# set a path
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path('./')
DATA_DIR = ROOT_DIR / 'data'
train_filepath = DATA_DIR / 'sensor.csv'
MODELS_DIR = ROOT_DIR / 'models'


def get_sensor_data(file):
    data = pd.read_csv(StringIO(str(file.file.read(), 'utf-8')), encoding='utf-8')
    data.iloc[:, 1:2].fillna((data.iloc[:, 1:2].mean()), inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    dfSensors = data.drop(['machine_status'], axis=1)
    return pd.DataFrame(data=dfSensors)


def sample_sensor_data(df):
    df.index = df.index.strftime('%Y-%m-%d')
    df['timestamp'] = pd.to_datetime(df.index)
    df = df.set_index('timestamp')
    grouped = df.groupby('timestamp').apply(lambda x: (x.iloc[0].min(), x.iloc[0].max()))
    return pd.DataFrame(grouped.explode())


def get_indexed_df(df, removeTimeFlag):
    if removeTimeFlag > 0:
        df.index = df.index.strftime('%Y-%m-%d')
    df['timestamp'] = pd.to_datetime(df.index)
    df = df.set_index('timestamp')
    return df


# remove un-ncessary columns
def remove_col(df, *args):
    return df.drop([*args], axis=1)


# set index to timestamp
def set_index(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    return df


# Inspecting missing values
def calc_percent_NAs(df):
    nans = pd.DataFrame(df.isnull().sum().sort_values(ascending=False) / len(df), columns=['percent'])
    idx = nans['percent'] > 0
    return nans[idx] * 100


# Impute Missing value
def impute_missing(df):
    df.iloc[:, 0:1].fillna((df.iloc[:, 0:1].mean()), inplace=True)
    return df


# label encoder
def encoder(df):
    ylabel = df.iloc[:, -1]
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(ylabel)
    return y


# standard scaling
def scaler(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return scaled


def get_preprocessed(dframe: pd.DataFrame):
    dframe = set_index(dframe)
    dframe = impute_missing(dframe)
    return dframe
