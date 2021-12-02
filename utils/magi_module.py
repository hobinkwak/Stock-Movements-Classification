import pandas as pd
import numpy as np


def sig_to_weight(sig_series, long_sig, short_sig, weight):
    long_count = (sig_series == long_sig).sum()
    short_count = (sig_series == short_sig).sum()

    if (long_count != 0) & (short_count != 0):
        sig_series.loc[(sig_series != long_sig) & (sig_series != short_sig)] = np.NaN
        sig_series.loc[sig_series == long_sig] = weight / long_count
        sig_series.loc[sig_series == short_sig] = -1.0 * weight / short_count
    else:
        sig_series.loc[:] = np.NaN

    return sig_series.fillna(0.0)


def long_only_sig_to_weight(sig_series, sig, weight):
    long_count = (sig_series == sig).sum()

    if long_count != 0:
        sig_series.loc[(sig_series != sig)] = np.NaN
        sig_series.loc[sig_series == sig] = weight / long_count
    else:
        sig_series.loc[:] = np.NaN

    return sig_series.fillna(0.0)


def get_ic(ret_1m, scores):
    ret_ = ret_1m.reindex(scores.index, columns=scores.columns).loc[scores.index].values
    numbers = scores.count(1).values
    scor = scores.values

    dates = scores.index

    scor = np.expand_dims(scor, -1)
    ret_ = np.expand_dims(ret_, -1)

    cal_ic = np.concatenate([scor, ret_], axis=-1)

    cov = np.nansum(np.prod(cal_ic - np.nanmean(cal_ic, 1, keepdims=True), 2), 1)

    ic = cov / np.nanstd(cal_ic, 1).prod(-1) / numbers
    return pd.DataFrame(ic, columns=["IC"], index=dates)


def build_rank_port(scores):
    """
    결과값 scores가 들어가면 rank_port 생성
    percent_rank를 기준으로 abosolute deviation 기준으로 normalize하고
    2를 곱해 LONG/SHORT이 각각 1이 되도록 조정한 롱숏 포트폴리오 생성

    params: scores : 모델 결과값
    """
    pct_rank = scores.rank(1, pct=True).T  # , method = 'max').T

    rank_port = pct_rank - pct_rank.mean()

    rank_port = (rank_port / rank_port.abs().sum()).T * 2
    return rank_port


def get_report(score):
    rtn = pd.read_csv("./data/ret_data.csv")

    rtn = rtn.set_index("tdate")
    rtn.index = pd.to_datetime(rtn.index)
    rtn = rtn.shift(-1)

    score.index = pd.to_datetime(score.index)
    sig_data = score.rank(1, "first").apply(
        lambda x: pd.qcut(
            x,
            5,
            labels=False,
        )
        if not x.isnull().all()
        else x,
        1,
    )

    test_cut = 5

    ress = []
    mdds = []
    turnovers = []
    cagrs = []
    sharpes = []

    for signal in range(test_cut + 2):
        if signal == test_cut:

            name = "L-S"
            weight_sig_data = sig_data.copy().apply(
                sig_to_weight, axis=1, args=(test_cut - 1, 0, 1.0)
            )
        elif signal == test_cut + 1:
            weight_sig_data = build_rank_port(score.loc[sig_data.index])  # RANK_L-S
            weight_sig_data.index.name = "tdate"
            weight_sig_data.columns.name = "code"
        else:

            name = f"quan_{signal}"
            weight_sig_data = sig_data.copy().apply(
                long_only_sig_to_weight, axis=1, args=(signal, 1)
            )
        port = weight_sig_data.fillna(0)

        ret_data_ = rtn.loc[port.index[0] :]  # 해당 기간 맵핑
        ret_data_ = ret_data_.reindex(columns=port.columns)  # 종목 일치

        port = port.reindex(ret_data_.index, method="ffill")

        port_returns = (ret_data_ * port).sum(1).shift(1)

        turnover = weight_sig_data.diff()  # turnover 계산
        turnover.iloc[0] = weight_sig_data.iloc[0]

        res = (1 + port_returns.fillna(0)).cumprod()

        TO = (abs(turnover).sum(1) / 2).sum().mean()
        MDD = (res / res.cummax() - 1).min()  # MDD
        CAGR_ = res.values[-1] ** (1 / 36 / 30 * 360) - 1
        vol = np.std(res.pct_change().dropna())
        sharpe = (
            np.mean(res.pct_change().dropna())
            / np.std(res.pct_change().dropna())
            * np.sqrt(12)
        )

        sharpes.append(sharpe)
        mdds.append(MDD)
        turnovers.append(TO)
        cagrs.append(CAGR_)
        ress.append(res)
    columns = ["QUAN_{}".format(test_cut - d) for d in range(test_cut)]
    columns += ["L-S", "RANK_L-S"]

    ress = pd.concat(ress, 1)
    ress.columns = columns

    summary = pd.DataFrame(
        [ress.iloc[-1].values, mdds, turnovers, cagrs, sharpes],
        index=["RETURN", "MDD", "TURNOVER", "CAGR", "SHARPE"],
        columns=columns,
    )
    return summary, ress
