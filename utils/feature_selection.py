import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import pickle


def feature_select_by_vif(data, threshold=100):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [
        variance_inflation_factor(data.values, i) for i in range(data.shape[1])
    ]
    vif["features"] = data.columns
    new_X_data = data.drop(
        vif.loc[vif["VIF Factor"] >= threshold].features.values, axis=1
    )
    
    return new_X_data


def get_permutation_importance(model, X, y, save_dir):
    saved_dir, pair = save_dir
    result = permutation_importance(model, X, y, n_repeats=100, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    plt.figure(figsize=(20,10))
    plt.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=X.columns[sorted_idx])
    plt.title("Permutation Importances")
    plt.savefig(saved_dir + f'/permutation_importances_{pair}.png')
    feature_importance = pd.DataFrame(result.importances[sorted_idx].T, columns=X.columns).mean()
    feature_importance = feature_importance.loc[feature_importance > 0]
    return feature_importance


def get_mutual_info_by_feature(X, y):
    selector = SelectKBest(mutual_info_classif, k=len(X.columns))
    selector.fit(X, y)
    result = pd.DataFrame(zip(X.columns, selector.scores_))
    result.columns = ['col', 'score']
    result = result.loc[result['score'] > 0]
    result = result.sort_values(by='score', ascending=False)
    return result


def post_hoc_feature_select(model, X, y, save_dir, mi, pi):
    features_by_permutation_importance = get_permutation_importance(model, X, y, save_dir)
    features_by_mutual_info = get_mutual_info_by_feature(X, y)
    if mi:
        features_mi = list(features_by_mutual_info.col)
    else:
        features_mi = []
    if pi:
        features_pi = list(features_by_permutation_importance.index)
    else:
        features_pi = []
    total = features_mi + features_pi
    if len(total) < 1:
        selected_features = X.columns.tolist()
    else:
        selected_features = list(set(features_mi + features_pi))
    return selected_features, features_mi, features_pi
