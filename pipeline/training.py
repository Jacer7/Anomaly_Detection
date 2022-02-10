import pickle
from pathlib import Path
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.seasonal import STL

from pipeline.preprocess import encoder, get_preprocessed, remove_col, get_indexed_df

def stl_model(dfsensor):
    data = dfsensor.resample('D').mean().ffill()  # D-days, M-month, A-DEC- anual, Q-DEC-quarterly
    res = STL(data, period=15).fit()
    return res


def get_anomaly_limits(res, coefficient):
    res_mu = res.mean()
    res_dev = res.std()

    lower = res_mu - float(coefficient) * res_dev
    upper = res_mu + float(coefficient) * res_dev
    return lower, upper


def get_anomalies(res, dfsensor, lower, upper):
    dfres = pd.DataFrame(data=res)
    dfres = get_indexed_df(dfres, 0)

    dfsensor = get_indexed_df(dfsensor, 1)

    dfmerged = pd.merge(dfsensor, dfres, left_index=True, right_index=True)
    dfanamoli = dfmerged.loc[(dfmerged['resid'] < lower) | (dfmerged['resid'] > upper)]
    return pd.DataFrame(data=dfanamoli)

def get_y(dfr):
    y = encoder(dfr)
    # print(y)
    return y


def get_X(dfr):
    X = dfr.iloc[:, 0:1]
    # X = scaler(X)
    print(X)
    return X

def fit_model_if(dfr):
    # define random states
    state = 1
    model1 = IsolationForest(n_estimators=200, contamination=0.11, random_state=state, max_samples='auto')
    model1.fit(dfr.values)
    # save the model to disk
    filename = '../models/model_if.pkl'
    pickle.dump(model1, open(filename,'wb'))

def fit_model_lof(dfr):
    model2 = LocalOutlierFactor(n_neighbors=20, contamination=0.11, novelty=True)
    model2.fit(dfr.values)
    # save the model to disk
    filename2 = '../models/model_lof.pkl'
    pickle.dump(model2, open(filename2, 'wb'))
    

if __name__ == '__main__':
    import os
    print(os.getcwd())
    ROOT_DIR = Path('../')
    MODELS_DIR = ROOT_DIR / 'models'
    data = pd.read_csv('../data/uploads/Sensortest.csv')
    #dfr = remove_col(data, 'Unnamed: 0', 'sensor_15')
    dfr = get_preprocessed(data.iloc[:, 0:2])
    fit_model_if(dfr)
    fit_model_lof(dfr)