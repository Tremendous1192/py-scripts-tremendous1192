import polars as pl

def log10(df_train: pl.dataframe.frame.DataFrame,
          df_test: pl.dataframe.frame.DataFrame,
          column: str):
    '''
    訓練データとテストデータを常用対数変換する関数
    '''
    temp_train = df_train.with_columns(pl.col(column).log10())
    temp_test = df_test.with_columns(pl.col(column).log10())
    return temp_train, temp_test

def standardize(df_train: pl.dataframe.frame.DataFrame,
                df_test: pl.dataframe.frame.DataFrame,
                column: str):
    '''
    訓練データの平均値と標準偏差で標準化する関数
    '''
    mean = df_train[column].mean()
    std = df_train[column].std()
    # 標準化
    temp_train = df_train.with_columns((pl.col(column) - mean) / std)
    temp_test = df_test.with_columns((pl.col(column) - mean) / std)
    return temp_train, temp_test