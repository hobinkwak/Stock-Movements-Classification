import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm

def stationarity_adf_test(data):
    adf = pd.DataFrame(sm.tsa.stattools.adfuller(data)[0:2],
                                 index=['Test Statistics', 'p-value'], columns=['Stationarity_adf'])
    return adf


def stationarity_kpss_test(data):
    kpss = pd.DataFrame(sm.tsa.stattools.kpss(data)[0:2],
                                  index=['Test Statistics', 'p-value'], columns=['Stationarity_kpss'])
    return kpss

def stationary_transform(dataset, pairs):
    dic = pickle.load(open('stationarity_check.pkl','rb'))
    for pair in pairs:
        if pair in dic.keys():
            for col in dic[pair]:
                if dic[pair][col] == 'diff':
                    dataset[pair][0][col] = dataset[pair][0][col].diff()
                if dic[pair][col] == 'log_diff':
                    dataset[pair][0][col] = np.log(dataset[pair][0][col]).diff()
            dataset[pair] = dataset[pair][0].iloc[1:, :], dataset[pair][1].iloc[1:, :]
    return dataset

def minmaxscale(X):
    return (X - X.min()) / (X.max() - X.min())


def process_wics_data(wics):
    df = pd.read_csv(f"{wics}", encoding="euc_kr", index_col=0)
    df.columns = [
        [c.split(".")[0] if ("." in c) else c for c in df.columns],
        df.iloc[0],
    ]
    df = df.iloc[2:]
    df.index = pd.to_datetime(df.index)
    df = df.applymap(
        lambda x: float(str(x).replace(",", "")) if (type(x) != float) else x
    )
    return df


def prc_momentum(data, period):
    return data / data.shift(period)


def get_change(df, period=1, window=[20, 60, 120, 250]):
    return (df.pct_change(period) * np.sign(df.shift(periods=period))).dropna(how='all')


def get_moving_features(data, type='volume'):
    windows = [20, 60, 120, 250]
    features = []
    if type == 'price':
        for window in windows:
            feature = data.apply(lambda x: prc_momentum(x, window)).resample('M').last().dropna(how='all')
            features.append(feature)
    if type == 'volume':
        for window in windows:
            feature = get_change(data.rolling(window).sum().dropna(how='all').resample('M').last())
            scaled_features = feature.apply(lambda x: minmaxscale(x), axis=1)
            features.append(scaled_features)
    if type == 'fwd':
        fwd_1m = get_change(data.resample('M').last()).apply(lambda X: minmaxscale(X), axis=1)
        fwd_3m = get_change(data, 90).dropna(how='all').resample('M').last().apply(lambda X: minmaxscale(X),
                                                                                   axis=1)
        features = [fwd_1m, fwd_3m]
    return features


def feature_row(features, pair, date):
    feature_table = features['macro'].loc[date:]
    for key in list(features.keys())[1:2]:
        feature_table[key] = features[key].apply(lambda x: (x[pair[0]] / x[pair[1]]), axis=1).loc[date:]
    for key in list(features.keys())[2:]:
        feature_table[key] = features[key].apply(lambda x: (x[pair[0]] - x[pair[1]]), axis=1).loc[date:]
    return feature_table


def levinv_etf(df, score_table, underlying, lev=False, inv=False):

    if (type(lev) != type(False)):
        q = score_table.rank(1, 'first').apply(
            lambda x: pd.qcut(x, 5, labels=[-2, -1.5, 0, 1.5, 2], ) if not x.isnull().all() else x, 1)
        df[lev] = df[underlying] * q[underlying].astype(np.float)
    if (type(inv) != type(False)):
        q = score_table.rank(1, 'first').apply(
            lambda x: pd.qcut(x, 5, labels=[2, 1.5, 0, -1.5, -2], ) if not x.isnull().all() else x, 1)
        df[inv] = df[underlying] * q[underlying].astype(np.float)
    return df


