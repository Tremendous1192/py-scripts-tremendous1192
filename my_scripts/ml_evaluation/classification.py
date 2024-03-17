import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import japanize_matplotlib
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
import polars as pl

def evaluate_roc_auc(y_true: pd.core.series.Series, y_proba: np.ndarray):
    '''
    横軸に儀陽性率、縦軸に真陽性率のROC曲線を描く関数
    '''
    # AUCスコアの算出
    auc_score = roc_auc_score(y_true = y_true, y_score = y_proba)
    print("AUC score", auc_score)

    # ROC曲線の要素（偽陽性率、真陽性率、閾値）の算出
    fpr, tpr, thresholds = roc_curve(y_true = y_true, y_score = y_proba)

    # ROC曲線の描画
    plt.plot(fpr, tpr, label = 'roc curve (area = %0.3f)' % auc_score)
    plt.plot([0, 1], [0, 1], linestyle = ':', label = 'random')
    plt.plot([0, 0, 1], [0, 1, 1], linestyle = ':', label = 'ideal')
    plt.legend()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.show()

def check_clasiffication(y_true: pd.core.series.Series, y_pred: np.ndarray):
    '''
    分類問題の混合行列を計算する
    '''
    positive = "true" if pl.Series("true", y_true.to_numpy()).dtype == pl.Boolean else "1"
    negative = "false" if pl.Series("predict", y_true.to_numpy()).dtype == pl.Boolean else "0"
    return (
        pl.DataFrame({
            "Valid": y_true.to_numpy(),
            "Predicted": y_pred
        })
        .pivot(
            index = "Valid", columns = "Predicted", values = "Predicted",
            aggregate_function = "len", sort_columns = True
        )
        .with_columns( (pl.col(negative) + pl.col(positive)).alias("All") )
        .with_columns([
            (pl.col(negative) / pl.col("All") * 100).round(decimals = 1).alias(negative + "_rate[%]"),
            (pl.col(positive) / pl.col("All") * 100).round(decimals = 1).alias(positive + "_rate[%]")
        ])
        .select([
            pl.col(positive),
            pl.col(negative),
            pl.col("All"),
            pl.col(positive + "_rate[%]"),
            pl.col(negative + "_rate[%]")
        ])
        .sort("Valid", descending = True)
    )
