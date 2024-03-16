import pandas as pd
from sklearn.ensemble import IsolationForest

def score_samples(X: pd.core.frame.DataFrame,
                  y: pd.core.series.Series,
                  X_test: pd.core.frame.DataFrame,
                  normal_value):
    '''
    IsolationForestによる異常値スコア列を追加する関数
    '''
    # 学習
    clf = IsolationForest(random_state = 1192)
    clf.fit(X[y == normal_value])

    # 異常値スコアの追加
    score_train = clf.score_samples(X)
    score_test = clf.score_samples(X_test)

    # return X, X_test
    return X.assign(score_IsolationForest = score_train), X_test.assign(score_IsolationForest = score_test)
