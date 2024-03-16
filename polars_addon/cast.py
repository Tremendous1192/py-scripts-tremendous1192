import polars as pl

def cast_float_median_encoding(df_train: pl.dataframe.frame.DataFrame,
                               df_test: pl.dataframe.frame.DataFrame,
                               column: str):
    '''
    訓練データとテストデータのデータ型をpl.Float64に変換して、
    訓練データの中央値で欠損値を補完する関数
    '''
    # pl.Float64型への変換
    temp_train = df_train.with_columns(pl.col(column).cast(pl.Float64, strict = False))
    temp_test = df_test.with_columns(pl.col(column).cast(pl.Float64, strict = False))
    # 訓練データの中央値による欠損値補完
    median = temp_train[column].median()
    temp_train = temp_train.with_columns(pl.col(column).fill_null(value = median))
    temp_test = temp_test.with_columns(pl.col(column).fill_null(value = median))
    
    return temp_train, temp_test