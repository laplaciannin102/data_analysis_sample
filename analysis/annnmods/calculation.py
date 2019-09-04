#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: kosuke.asada
計算用関数
"""


import gc
import numpy as np

# 設定のimport
from .mod_config import *
# 自作関数のimport
from .analysis import *
from .scraping import *
from .useful import *
from .visualization import *


def pro_round(num, ndigits=0):
    """
    数字を四捨五入で丸める。

    Args:
        num: int or float
            丸めたい数字。

        ndigits: int, optional(default=0)
            丸めた後の小数部分の桁数。

    Returns:
        rounded: int or float
            丸めた後の数字。
    """
    num *= 10 ** ndigits
    rounded = ( 2* num + 1 ) // 2
    rounded /= 10 ** ndigits

    if ndigits == 0:
        rounded = int(rounded)

    return rounded


def sturges_rule(num):
    """
    スタージェスの公式を用いて、
    サンプルサイズから階級(カテゴリ、ビン(bins))の数を計算する。
    公式のTeX表記: \[bins = 1 + \log_2{N} \nonumber \]

    Args:
        num: int
            サンプルサイズ。原則1以上の整数を想定。
    
    Returns:
        n_bins: int
            スタージェスの公式から導かれた適切な階級の数。
    """
    # numが0以下の時は1を返す
    if num <= 0:
        num = 1
        return 1
    
    # スタージェスの公式
    n_bins = int(pro_round(1 + np.log2(num), 0))
    
    return n_bins