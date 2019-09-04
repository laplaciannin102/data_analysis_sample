#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: kosuke.asada
可視化用関数
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydotplus as pdp
import seaborn as sns

import japanize_matplotlib

# 設定のimport
from .mod_config import *
# 自作関数のimport
from .analysis import *
from .calculation import *
from .scraping import *
from .useful import *


def graph(x, y, title='graph', xlabel='x', ylabel='y'):
    """
    簡単なグラフを描く

    Args:
        x: sequence of scalar
            横軸(x軸)に使用する値。

        y: sequence of scalar
            縦軸(y軸)に使用する値。
        
        title: str, optional(default='graph')
            グラフのタイトルに使用するテキスト。

        xlabel: str, optional(default='x')
            横軸(x軸)のラベルに使用するテキスト。

        ylabel: str, optional(default='y')
            横軸(x軸)のラベルに使用するテキスト。
    
    Remarks:
        https://matplotlib.org/api/pyplot_api.html
    """
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def ratio_graph(x, y1, y2, xlabel='x', ylabel='y', y1label='y1', y2label='y2', ratio_label='ratio', figsize=(12, 10), fontsize=20, bar_width=0.75):
    """
    y1の割合を示すグラフを描く。使用例は下記Examples参照。

    Args:
        x: sequence of scalar
            横軸(x軸)に使用する値。

        y1: sequence of scalar
            縦軸(y軸)に使用する値y1。このy1の割合が折れ線として表示される。
        
        y2: sequence of scalar
            縦軸(y軸)に使用する値y2。
        
        xlabel: str, optional(default='x')
            横軸(x軸)のラベルに使用するテキスト。

        ylabel: str, optional(default='y')
            横軸(y軸)のラベルに使用するテキスト。

        y1label: str, optional(default='y1')
            横軸(y軸)の値y1のラベルに使用するテキスト。
        
        y2label: str, optional(default='y2')
            横軸(y軸)の値y2のラベルに使用するテキスト。
        
        ratio_label: str, optional(default='ratio')
            y1の割合を表す値のラベルに使用するテキスト。
        
        figsize: tuple of int, optional(default=(12, 10))
            グラフのサイズ。(横, 縦)で指定。
        
        fontsize: int, optional(default=20)
            フォントのサイズ。
        
        bar_width: float, optional(default=0.75)
            棒グラフの幅。
    
    Examples:
    
    names = ['太郎', '次郎', '三郎', '四郎', '五郎', '六郎']
    n = len(names)
    np.random.seed(seed=57)
    df = pd.DataFrame({'name': names})
    df['パン'] = np.random.randint(0, 300, n)
    df['お米'] = np.random.randint(0, 300, n)
    ratio_graph(df['name'], df['パン'], df['お米'], xlabel='名前', ylabel='食事回数', y1label='お米', y2label='パン', ratio_label='パンの割合')
    """
    x = list(x)
    xstr_li = [str(ii) for ii in x]
    y1 = np.array(y1)
    y2 = np.array(y2)
    y3 = y1 + y2                            # 合計
    ratio_list = list( y1 / (y1 + y2) )     # y1の割合

    df = pd.DataFrame({'x': xstr_li, 'y1': y1, 'y2': y2, 'r': ratio_list})
    df = df.reset_index(drop=False).rename(columns={'index': 'alt_id'})
    df['alt_id'] = df['alt_id'].astype(int)
    alt_id = df['alt_id'].values.tolist()

    ymax = y3.max()
    ymin = y3.min()
    ydiff = ymax - ymin     # y軸の幅

    if ymax > 0:
        ymax += ydiff * 0.1
    else:
        ymax = 0

    if ymin < 0:
        ymin -= ydiff * 0.1
    else:
        ymin = 0

    xmax = max(alt_id)
    xmin = min(alt_id)
    xdiff = xmax - xmin     # x軸の幅

    fig, ax1 = plt.subplots(figsize=figsize)

    w = bar_width # 棒グラフの幅
    # ind = np.arange(len(x)) # x方向の描画位置を決定するための配列

    ax1.bar(alt_id, y1, width=w, color='r', label=str(y1label))
    ax1.bar(alt_id, y2, width=w, bottom=y1, color='b', label=str(y2label))

    ax1.set_xlabel(str(xlabel), fontsize=fontsize)
    ax1.set_xticks(alt_id)
    ax1.set_xticklabels(xstr_li)
    ax1.set_xlim(0 - 1, xmax + 1)

    ax1.set_ylabel(str(ylabel), fontsize=fontsize)
    # ax1.set_ylim(0 - (ydiff * 0.1), ymax + (ydiff * 0.1))
    ax1.set_ylim(ymin, ymax)

    # Which axis to apply the parameters to.
    # ax1.tick_params('y', labelsize=15)
    ax1.tick_params('both', labelsize=15)

    ax2 = ax1.twinx()

    ax2.plot(alt_id, ratio_list, 'g', lw=2, linestyle=':', marker='o', ms=15, mec='black', mfc='yellow')
    ax2.set_ylabel(str(ratio_label), color='g', fontsize=fontsize)
    ax2.tick_params('y', colors='g', labelsize=15, grid_alpha=0.5)
    ax2.set_ylim(-0.05, 1)

    ax2.grid(axis='y', alpha=0.5, linestyle='-')

    ax1.legend(fontsize=15)
    # fig.tight_layout()
    plt.show()

