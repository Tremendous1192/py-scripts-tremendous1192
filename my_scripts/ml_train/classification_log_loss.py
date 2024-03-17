import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss

def train_test_split(
        clf,
        X: pd.core.frame.DataFrame,
        y: pd.core.series.Series,
        n_splits: int,
        test_size: float,
        radom_state: int):
    '''
    predict_proba()メソッドを持つ分類器の評価指標が高くなるように訓練データと検証データを分割する.
    '''
    # 初期化
    sss = StratifiedShuffleSplit(n_splits =  n_splits, test_size = test_size, random_state = radom_state)
    best_score = 1000000
    best_index = None

    # k-fold cross validation
    for i, (train_index, valid_index) in enumerate(sss.split(X, y)):
        # 訓練データの学習
        clf.fit(X.iloc[train_index, :], y.iloc[train_index])
        y_proba = clf.predict_proba(X.iloc[valid_index, :])
        score_train = log_loss(y.iloc[valid_index], y_proba)
        # 評価データの学習
        clf.fit(X.iloc[valid_index, :], y.iloc[valid_index])
        y_proba = clf.predict_proba(X.iloc[train_index, :])
        score_valid = log_loss(y.iloc[train_index], y_proba)
        # 訓練データと検証データの評価関数の積で評価する
        score = score_train * score_valid
        # 最良スコアのインデックスを残す
        if score < best_score:
            print(f"Fold {i}: {score:.3f} = {score_train:.3f} x {score_valid:.3f}")
            best_score = score
            best_index = train_index
    
    # 結果
    clf.fit(X.iloc[best_index, :], y.iloc[best_index])
    print(f"Best Score {best_score:.3f}")
    return best_index
