import pandas as pd
import polars as pl

def from_pandas_to_polars(X: pd.core.frame.DataFrame,
                          y: pd.core.series.Series,
                          X_test: pd.core.frame.DataFrame):
    '''
    訓練データとテストデータをpolarsデータセットに変換する
    '''
    train = pl.from_pandas(data = pd.concat([X, y], axis = 1), include_index = True)
    test = pl.from_pandas(data = X_test, include_index = True)
    return train, test

def to_pandas_datasets(df_train: pl.dataframe.frame.DataFrame,
                       df_test: pl.dataframe.frame.DataFrame,
                       id_variable: str,
                       response_variable: str,
                       explanatory_variables: str):
    '''
    訓練データとテストデータをpandasデータセットに変換する
    '''
    X = df_train[[id_variable] + explanatory_variables].to_pandas().set_index(id_variable)
    y = df_train[[id_variable, response_variable]].to_pandas().set_index(id_variable)[response_variable]
    X_test = df_test[[id_variable] + explanatory_variables].to_pandas().set_index(id_variable)
    
    return X, y, X_test