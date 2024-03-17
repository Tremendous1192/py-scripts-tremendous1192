import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

def train_roc_auc(X: pd.core.frame.DataFrame,
          y: pd.core.series.Series,
          n_splits: int,
          test_size: float):
    '''
    StratifiedShuffleSplitを用いたRandomForestClasiferの学習\
    (roc_auc_score固定)
    '''
    # 初期化
    sss = StratifiedShuffleSplit(n_splits =  n_splits, test_size = test_size, random_state = 1192)
    clf = RandomForestClassifier(class_weight = "balanced", random_state = 1192)
    best_score = -1000
    best_index = None

    # k-fold cross validation
    for i, (train_index, valid_index) in enumerate(sss.split(X, y)):
        # 訓練データの学習
        clf.fit(X.iloc[train_index, :], y.iloc[train_index])
        y_proba = clf.predict_proba(X.iloc[valid_index, :])[:, 1]
        score = roc_auc_score(y.iloc[valid_index], y_proba)
        print(f"Fold {i}: {score:.3f}")
        # 最良スコアのインデックスを残す
        if score > best_score:
            best_score = score
            best_index = train_index
    
    # 結果
    model = clf.fit(X.iloc[best_index, :], y.iloc[best_index])
    print(f"Best Score {best_score:.3f}")
    return model, best_index, best_score

def train_roc_auc_square(X: pd.core.frame.DataFrame,
          y: pd.core.series.Series,
          n_splits: int,
          test_size: float):
    '''
    StratifiedShuffleSplitを用いたRandomForestClasiferの学習\
    (roc_auc_score固定)\
    訓練データと検証データの2つの評価関数の積で評価する
    '''
    # 初期化
    sss = StratifiedShuffleSplit(n_splits =  n_splits, test_size = test_size, random_state = 1192)
    clf = RandomForestClassifier(class_weight = "balanced", random_state = 1192)
    best_score = -1000
    best_score_train = -1000
    best_score_valid = -1000
    best_index = None

    # k-fold cross validation
    for i, (train_index, valid_index) in enumerate(sss.split(X, y)):
        # 訓練データの学習
        clf.fit(X.iloc[train_index, :], y.iloc[train_index])
        y_proba = clf.predict_proba(X.iloc[valid_index, :])[:, 1]
        score_train = roc_auc_score(y.iloc[valid_index], y_proba)
        # 評価データの学習
        clf.fit(X.iloc[valid_index, :], y.iloc[valid_index])
        y_proba = clf.predict_proba(X.iloc[train_index, :])[:, 1]
        score_valid = roc_auc_score(y.iloc[train_index], y_proba)
        # 訓練データと検証データの評価関数の積で評価する
        score = score_train * score_valid
        print(f"Fold {i}: {score:.3f} = {score_train:.3f} x {score_valid:.3f}")
        # 最良スコアのインデックスを残す
        if score > best_score:
            best_score = score
            best_score_train = score_train
            best_score_valid = score_valid
            best_index = train_index
    
    # 結果
    model = clf.fit(X.iloc[best_index, :], y.iloc[best_index])
    print(f"Best Score {best_score:.3f} = {best_score_train:.3f} x {best_score_valid:.3f}")
    return model, best_index, best_score

