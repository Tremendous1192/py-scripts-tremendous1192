import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

def train_roc_auc(X: pd.core.frame.DataFrame,
          y: pd.core.series.Series,
          n_splits: int,
          test_size: float):
    '''
    StratifiedShuffleSplitを用いたLogisticRegression推定器の学習\
    (roc_auc_score固定)
    '''
    # 交差分割
    sss = StratifiedShuffleSplit(n_splits =  n_splits, test_size = test_size, random_state = 1192)

    # 初期化
    tsr = RandomForestClassifier(class_weight = "balanced", random_state = 1192)
    scores = []
    best_score = -1000
    best_index = None

    # k-fold cross validation
    for i, (train_index, valid_index) in enumerate(sss.split(X, y)):
        tsr.fit(X.iloc[train_index, :], y.iloc[train_index])
        y_proba = tsr.predict_proba(X.iloc[valid_index, :])[:, 1]
        score = roc_auc_score(y.iloc[valid_index], y_proba)
        print(f"Fold {i}: {score}")
        scores.append(score)
        if score > best_score:
            best_score = score
            best_index = train_index
    
    # 結果
    model = tsr.fit(X.iloc[best_index, :], y.iloc[best_index])
    print(f"Best Score {best_score}")
    return model, best_index, best_score