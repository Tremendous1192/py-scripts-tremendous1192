import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score

def train(X: pd.core.frame.DataFrame,
          y: pd.core.series.Series,
          n_splits: int,
          test_size: float,
          metric):
    '''
    ShuffleSplitによるLogisticRegression推定器の学習
    '''
    # 交差分割
    rs = ShuffleSplit(n_splits =  n_splits, test_size = test_size, random_state = 1192)
    rs.get_n_splits(X)

    # 初期化
    tsr = LogisticRegression(max_iter = 1000, class_weight = "balanced", random_state = 1192)
    scores = []
    best_score = -1000
    best_index = None

    # k-fold cross validation
    for i, (train_index, valid_index) in enumerate(rs.split(X)):
        tsr.fit(X.iloc[train_index, :], y.iloc[train_index])
        y_pred = tsr.predict(X)
        score = metric(y, y_pred)
        print(f"Fold {i}: {score}")
        scores.append(score)
        if score > best_score:
            best_score = score
            best_index = train_index
    
    # 結果
    model = tsr.fit(X.iloc[best_index, :], y.iloc[best_index])
    print(f"Best Score {best_score}")
    return model, best_index, best_score

def train_roc_auc(X: pd.core.frame.DataFrame,
          y: pd.core.series.Series,
          n_splits: int,
          test_size: float):
    '''
    ShuffleSplitによるLogisticRegression推定器の学習\
    (roc_auc_score固定)
    '''
    # 交差分割
    rs = ShuffleSplit(n_splits =  n_splits, test_size = test_size, random_state = 1192)
    rs.get_n_splits(X)

    # 初期化
    tsr = LogisticRegression(max_iter = 1000, class_weight = "balanced", random_state = 1192)
    scores = []
    best_score = -1000
    best_index = None

    # k-fold cross validation
    for i, (train_index, valid_index) in enumerate(rs.split(X)):
        tsr.fit(X.iloc[train_index, :], y.iloc[train_index])
        y_proba = tsr.predict_proba(X)
        score = roc_auc_score(y, y_proba)
        print(f"Fold {i}: {score}")
        scores.append(score)
        if score > best_score:
            best_score = score
            best_index = train_index
    
    # 結果
    model = tsr.fit(X.iloc[best_index, :], y.iloc[best_index])
    print(f"Best Score {best_score}")
    return model, best_index, best_score