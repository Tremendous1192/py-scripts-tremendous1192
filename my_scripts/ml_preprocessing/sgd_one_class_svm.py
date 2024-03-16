import pandas as pd
from sklearn.linear_model import SGDOneClassSVM

def score_samples(X: pd.core.frame.DataFrame,
                  y: pd.core.series.Series,
                  X_test: pd.core.frame.DataFrame,
                  normal_value):
    '''
    One Class SVMによる異常値スコア列を追加する関数
    '''
    # 学習
    clf = SGDOneClassSVM(random_state = 1192)
    clf.fit(X[y == normal_value])

    # 異常値スコアの追加
    score_train = clf.score_samples(X)
    score_test = clf.score_samples(X_test)

    return X.assign(score_OneClassSVM = score_train), X_test.assign(score_OneClassSVM = score_test)
    #temp_X = X.insert(X.shape[1] - 1, "score_OneClassSVM", clf.score_samples(X)).copy()
    #temp_X_test = X_test.insert(X_test.shape[1] - 1, "score_OneClassSVM", clf.score_samples(X_test)).copy()
    #return temp_X, temp_X_test
