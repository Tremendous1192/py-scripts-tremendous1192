import polars as pl

def describe(df_train: pl.dataframe.frame.DataFrame,
          df_test: pl.dataframe.frame.DataFrame,
          column: str):
    '''
    訓練データとテストデータのSeriesの基本統計量を比較する関数
    '''
    describe_train = df_train[column].describe().rename({"value": "train"})
    describe_test = df_test[column].describe().rename({"value": "test"})
    return describe_train.join(other = describe_test, on = "statistic", how = "outer_coalesce")
