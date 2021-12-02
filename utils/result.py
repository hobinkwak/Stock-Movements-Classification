import os
from utils.dataload import *


def get_score_table(dataset, final_models, now):
    pairs = list(dataset.keys())
    etf_close = load_etf()

    WICS대, WICS대_exposure = load_wics_data()
    _, features = features_from_wics(WICS대)

    predict_last_month = {pair: feature_row(features, pair, "2021-05-31") for pair in pairs}
    Result = pd.DataFrame(final_models.values())
    Result.index = [p for p in pairs]
    winner_pred = pd.concat([pd.Series(Result[1][i]) for i in range(len(Result))], axis=1)
    winner_pred.index = pd.date_range('2017-12', '2021-05', freq='M')
    winner_pred.loc[pd.to_datetime('2021-05-31')] = list(Result.apply(lambda x: x[3][x[0].predict(predict_last_month[x.name][x[2]])[0]], axis=1))
    score = winner_pred.apply(lambda x: x.value_counts(), axis=1)

    etf_score = {etf: WICS대_exposure[etf][score.columns].loc[score.index[0]:] * score for etf in
                 WICS대_exposure.columns.get_level_values(0).unique()}
    score_table = pd.concat([etf_score[etf].sum(axis=1) for etf in etf_score.keys()], axis=1)
    score_table.columns = etf_score.keys()

    score_table_c = score_table.copy()
    score_table_c = levinv_etf(score_table_c, score_table, 'A139260', lev='A243880', inv=False)
    score_table_c = levinv_etf(score_table_c, score_table, 'A139250', lev='A243890', inv=False)
    score_table_c = levinv_etf(score_table_c, score_table, 'A102110', lev='A267770', inv=False)
    score_table_c = levinv_etf(score_table_c, score_table, 'A102110', lev='A123320', inv=False)
    score_table_c = levinv_etf(score_table_c, score_table, 'A102110', lev=False, inv='A252710')
    score_table_c = levinv_etf(score_table_c, score_table, 'A232080', lev='A233160', inv=False)
    score_table_c = levinv_etf(score_table_c, score_table, 'A232080', lev=False, inv='A250780')

    score_table_final = score_table_c.loc['2018-12-31':]
    score_table_final.index = [
        etf_close.loc[score_table_final.index[0]:].index[0] if (i == 0) else score_table_final.index[i] for i in
        range(len(score_table_final))]

    submission = score_table_final.stack()
    submission = submission.reset_index()
    submission.columns = ['tdate', 'code', 'score']

    os.makedirs(f'result/{now}/score', exist_ok=True)
    submission.to_csv(f'result/{now}/score/2021 빅페_미래에셋자산운용_score.csv')
    return submission