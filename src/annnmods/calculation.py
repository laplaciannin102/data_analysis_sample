#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: kosuke.asada
数値計算用module
"""


import gc
import numpy as np

# 設定のimport
from .mod_config import *
# 自作moduleのimport
# from .useful import *
# from .scraping import *
# from .visualization import *
# from .preprocessing import *
# from .analysis import *


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


def in_section(x, section, bounds=[False, True]):
    """
    xがsectionに含まれているかどうかを返す。
    boundsがTrueだとClosed、FalseだとOpenになる。

    Args:
        x: int or float
            number
        
        section: list of float or list of str
            下端と上端。
        
        bounds: list of bool
            左端と右端を含むかどうか。
            [左端を含むか, 右端を含むか]
    
    Returns:
        ret: bool
            xがsectionに含まれているかどうか。
            含まれている時Trueを返す。
    """
    x = float(x)
    ret = False

    # left and right
    left = bounds[0]
    right = bounds[1]

    # Lower and Upper
    low = float(section[0])
    up = float(section[1])

    if left:
        left_condition = (low <= x)
    else:
        left_condition = (low < x)
    
    if right:
        right_condition = (x <= up)
    else:
        right_condition = (x < up)
    
    if left_condition and right_condition:
        ret = True
    
    return ret




