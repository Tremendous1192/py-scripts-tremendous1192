import pandas as pd
from sklearn.linear_model import TheilSenRegressor
from sklearn.model_selection import ShuffleSplit

def train(X: pd.core.frame.DataFrame,
          y: pd.core.series.Series,
          n_splits: int,
          test_size: float,
          metric):
    '''
    ShuffleSplitによるTheil-Sen推定器の学習
    '''
    # 初期化
    rs = ShuffleSplit(n_splits =  n_splits, test_size = test_size, random_state = 1192)
    rs.get_n_splits(X)
    tsr = TheilSenRegressor(max_iter = 1000)
    best_score = 1000000
    best_index = None

    # k-fold cross validation
    for i, (train_index, valid_index) in enumerate(rs.split(X)):
        # 訓練データの学習
        tsr.fit(X.iloc[train_index, :], y.iloc[train_index])
        y_pred = tsr.predict(X.iloc[valid_index, :])
        score = metric(y.iloc[valid_index], y_pred)
        print(f"Fold {i}: {score}")
        # 最良スコアのインデックスを残す
        if score < best_score:
            best_score = score
            best_index = train_index
    
    # 結果
    model = tsr.fit(X.iloc[best_index, :], y.iloc[best_index])
    print(f"Best Score {best_score}")
    return model, best_index, best_score
