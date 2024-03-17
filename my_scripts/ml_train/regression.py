import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

def train_test_split(
        clf,
        X: pd.core.frame.DataFrame,
        y: pd.core.series.Series,
        n_splits: int,
        test_size: float,
        radom_state: int,
        metric):
    '''
    回帰推定器の評価指標が良くなるように訓練データと検証データを分割する.
    '''
    # 初期化
    sss = StratifiedShuffleSplit(n_splits =  n_splits, test_size = test_size, random_state = radom_state)
    best_score = 100000000
    best_index = None

    # k-fold cross validation
    for i, (train_index, valid_index) in enumerate(sss.split(X, y)):
        # 訓練データの学習
        clf.fit(X.iloc[train_index, :], y.iloc[train_index])
        y_pred = clf.predict(X.iloc[valid_index, :])
        score_train = metric(y.iloc[valid_index], y_pred, average = "macro")
        # 評価データの学習
        clf.fit(X.iloc[valid_index, :], y.iloc[valid_index])
        y_pred = clf.predict(X.iloc[train_index, :])
        score_valid = metric(y.iloc[train_index], y_pred, average = "macro")
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
