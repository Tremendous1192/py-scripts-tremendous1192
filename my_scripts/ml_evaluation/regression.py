import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import japanize_matplotlib
import seaborn as sns


def evaluate(y_true: pd.core.series.Series, y_pred: np.ndarray, metric):
    '''
    回帰問題の精度を確認する関数
    '''
    score = str(np.round(metric(y_true, y_pred), decimals = 3))
    sns.scatterplot(x = y_true, y = y_pred, label = f"{score}")
    plt.legend()

    # y = x の基準線の描画
    min_value = np.min([y_true.min(), np.min(y_pred)])
    max_value = np.max([y_true.max(), np.max(y_pred)])
    plt.plot([min_value, max_value], [min_value, max_value], color = "red")

    # 軸ラベルの設定
    plt.xlabel("Actual response value")
    plt.ylabel("predicted response value")
    plt.show()
    pass