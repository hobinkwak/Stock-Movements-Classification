from itertools import combinations
import pandas as pd

from utils.utils import *


def load_etf():
    etf_data = pd.read_csv(
        "data/etf_data.csv", encoding="euc_kr", parse_dates=["tdate"]
    )
    etf_ohlcv = etf_data.set_index(["tdate", "etf_code", "data_name"])[
        "value"
    ].unstack()
    etf_close = etf_ohlcv["종가"].unstack()
    return etf_close

def load_macro_data():
    macro_data = pd.read_csv('외부데이터/macro_final.csv', index_col='Item Name').iloc[1:, :]
    macro_data.index = pd.to_datetime(macro_data.index)
    macro_data = macro_data.fillna(method='ffill')
    macro_data = (macro_data.resample('m').last() / macro_data.resample('m').first())

    macro_data.columns = ['FOMC정책금리', '한국정책금리', '중국정책금리', '미국국채_1m', '미국국채_3m', '미국국채_6m', '미국국채_1y', '미국국채_5y',
                          '미국국채_10y', '리보_달러_1m', '리보_달러_1y', '리보_달러_3m', '리보_달러_6m', '리보_달러_1w',
                          'DDR4 16G (2G*8) 2666 MHZ', 'NAND 16Gb 2Gx8 SLC', 'DDR4 16G (2G*8) eTT MHZ',
                          'DDR3 4Gb 512Mx8 1600/1866Mbps', 'DDR3 4Gb 512Mx8 eTT',
                          'NAND 8Gb 1Gx8 SLC', 'NAND 64Gb 8Gx8 MLC', 'WTI_1M', 'BRENT_1M', 'DUBAI_ASIA1M',
                          '난방유_선물_NYMEX', '천연가스_선물_NYMEX', '가스오일_선물_IPE', '천연가스_선물_IPE', '금_선물', '은_선물', '알루미늄_선물',
                          '전기동_선물', '납_선물', '니켈_선물', '주석_선물', '아연_선물', '10YR BEI', 'T10Y2Y', 'DFF',
                          'HY Ef Yield', 'Trade DI', 'VIX', 'USDKRW', 'Eco Policy Uncertainty']

    macro_data = macro_data[
        ['FOMC정책금리', '한국정책금리', '중국정책금리', '미국국채_1m', '미국국채_3m', '미국국채_6m', '미국국채_1y', '미국국채_5y', '미국국채_10y', '리보_달러_1m',
         '리보_달러_1y', '리보_달러_3m', '리보_달러_6m', '리보_달러_1w', 'DDR3 4Gb 512Mx8 eTT',
         'NAND 8Gb 1Gx8 SLC', 'WTI_1M', 'BRENT_1M', 'DUBAI_ASIA1M', '난방유_선물_NYMEX', '천연가스_선물_NYMEX', '가스오일_선물_IPE',
         '천연가스_선물_IPE', '금_선물', '은_선물', '알루미늄_선물', '전기동_선물', '납_선물', '니켈_선물', '주석_선물', '아연_선물', '10YR BEI', 'T10Y2Y',
         'HY Ef Yield', 'Trade DI', 'VIX', 'USDKRW', 'Eco Policy Uncertainty']]
    return macro_data



def load_wics_data():
    WICS대_exposure = process_wics_data("./외부데이터/ETF별 업종 exposure.csv")
    WICS업종 = process_wics_data("./외부데이터/WICS 업종별 투자정보 데이터.csv")
    WICS대 = WICS업종[
        [
            "에너지",
            "소재",
            "산업재",
            "경기관련소비재",
            "필수소비재",
            "건강관리",
            "금융",
            "IT",
            "커뮤니케이션서비스",
            "유틸리티",
        ]
    ]
    WICS대 = WICS대.T.drop_duplicates().T
    return WICS대, WICS대_exposure



def features_from_wics(wics):
    """
    wics : WICS대 (from load_wics_data())
    """
    wics_price = wics.xs("종가지수", level=1, axis=1)
    momentums = get_moving_features(wics_price, type='price')

    wics_trd_volume = wics.xs("거래대금", level=1, axis=1)
    trd_volumes = get_moving_features(wics_trd_volume, type='volume')
    wics_retail_volume = wics.xs("개인 순매수대금(일간)", level=1, axis=1).fillna(0)
    retail_volumes = get_moving_features(wics_retail_volume, type='volume')
    wics_for_volume = wics.xs("외국인총합계순매수대금(일간)", level=1, axis=1).fillna(0)
    for_volumes = get_moving_features(wics_for_volume, type='volume')
    wics_inst_volume =  wics.xs("기관 순매수대금(일간)", level=1,axis=1).fillna(0)
    inst_volumes = get_moving_features(wics_inst_volume, type='volume')

    wics_pe =  wics.xs("P/E(FY0)", level=1,axis=1)
    pe_scale = wics_pe.resample('M').last().apply(lambda X: minmaxscale(X), axis=1)

    wics_fwd_pe =  wics.xs("P/E(Fwd.12M)", level=1,axis=1)
    fwd_pe_changes = get_moving_features(wics_fwd_pe, type='fwd')
    wics_fwd_eps =  wics.xs("EPS(Fwd.12M, 지배)", level=1,axis=1)
    fwd_eps_changes =get_moving_features(wics_fwd_eps, type='fwd')

    size_ =  wics.xs("시가총액", level=1,axis=1).resample('M').last()

    features = {
        "macro": load_macro_data(),
        "size": size_,
        "mom_1m": momentums[0],
        "mom_3m": momentums[1],
        "mom_6m": momentums[2],
        "mom_1y": momentums[3],
        "trd_1m": trd_volumes[0],
        "trd_3m": trd_volumes[1],
        "trd_6m": trd_volumes[2],
        "trd_1y": trd_volumes[3],
        "retail_trd_1m": retail_volumes[0],
        "retail_trd_3m": retail_volumes[1],
        "retail_trd_6m": retail_volumes[2],
        "retail_trd_1y": retail_volumes[3],
        "for_trd_1m": for_volumes[0],
        "for_trd_3m": for_volumes[1],
        "for_trd_6m": for_volumes[2],
        "for_trd_1y": for_volumes[3],
        "inst_trd_1m": inst_volumes[0],
        "inst_trd_3m": inst_volumes[1],
        "inst_trd_6m": inst_volumes[2],
        "inst_trd_1y": inst_volumes[3],
        "fwd_pe_1m": fwd_pe_changes[0],
        "fwd_pe_3m": fwd_pe_changes[1],
        "fwd_eps_1m": fwd_eps_changes[0],
        "fwd_eps_3m": fwd_eps_changes[1],
        "pe": pe_scale,
    }

    return wics_price, features


def combination_set(pair, start, end, price, features):
    """
    :param pair: WICS대분류 pair
    :param start: 기간
    :param end: 기간
    :param price: wics_prices (from features_from_wics())
    :param features: features (from features_from_wics())
    """
    comb_price = price[list(pair)]
    comb_ret = (comb_price.resample('m').last() / comb_price.resample('m').first()).loc[start:end]

    feature_table = features['macro'].loc[start:end]
    for key in list(features.keys())[1:6]:
        feature_table[key] = features[key].apply(lambda x: (x[pair[0]] / x[pair[1]]), axis=1).loc[start:end]
    for key in list(features.keys())[6:]:
        feature_table[key] = features[key].apply(lambda x: (x[pair[0]] - x[pair[1]]), axis=1).loc[start:end]

    comb_ret['winner'] = comb_ret.apply(
        lambda x: comb_ret.columns[0] if (x[comb_ret.columns[0]] > x[comb_ret.columns[1]]) else comb_ret.columns[1],
        axis=1)

    feature_table = feature_table.replace([-np.inf, np.inf], np.nan).fillna(method='ffill')
    comb_ret = comb_ret.replace([-np.inf, np.inf], np.nan).fillna(method='ffill')

    feature_table = feature_table.shift(1).iloc[1:]
    comb_ret = comb_ret.iloc[1:]

    X_data = feature_table
    y_data = comb_ret[['winner']].astype('category')

    return X_data, y_data

def load_dataset():
    WICS대,_ = load_wics_data()
    price, features = features_from_wics(WICS대)
    columns = ['에너지', '소재', '산업재', '경기관련소비재', '필수소비재', '건강관리', '금융', 'IT', '커뮤니케이션서비스', '유틸리티']
    pairs = list(combinations(columns, 2))
    total_dataset = {pair : combination_set(pair,'2011-12','2021-05', price, features) for pair in pairs}
    return total_dataset
