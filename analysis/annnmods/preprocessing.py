#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: kosuke.asada
前処理用module
"""


import sys, os
import gc
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 設定のimport
from .mod_config import *
# 自作moduleのimport
from .calculation import *
from .useful import *
from .scraping import *
from .visualization import *
# from .analysis import *


class AnnnPreprocessing():
    """
    前処理をまとめて行うクラス。ppとはPreprocessingのこと。
    exe_pp()を実行する前に、set_df()を実行しDataFrameをセットすること。

    Attributes:
        first_df: DataFrame, default=None
            最初のDataFrame。
        
        first_df_cols: list of str, default=[]
            最初のDataFrameのカラムのリスト。
        
        latest_df: DataFrame, default=None
            前処理後のDataFrame。
        
        latest_df_cols: list of str, default=[]
            前処理後のDataFrameのカラムのリスト。
        
        target: str, default=None
            目的変数となるカラムの名前。
        
        features: list of str, default=[]
            特徴量として使用するカラムのリスト。
        
        cols_dict: dict of list
            種々のカラムのリストをまとめたdictionary。
            各keyに対応するvalueは以下。
            key一覧: ignore, number, class, category, dummy, dtime

            'ignore': list of str, default=[]
                前処理の対象外となるカラムのリスト。
            'number': list of str, default=[]
                数値型のカラムのリスト。
            'class': list of str, default=[]
                階級で分ける対象のカラムのリスト。
            'bins': dict
                binsを入れたdict。
            'category': list of str, default=[]
                ダミー変数化前のカテゴリー変数のカラムのリスト。
            'dummy': list of str, default=[]
                ダミー変数のカラムのリスト。
            'dtime': list of str, default=[]
                時系列(datetime型)カラムのリスト。
                
        pp_dict: dict, default={}
            一連の前処理を並べた辞書。
            {1: ['func1', func1], 0: ['func0', func0], 2: ['func2', func2]}
            の様な形で指定し、keyの値が若い順に前処理は実行される。
            この場合、func0、func1、func2の順となる。
        
        n_pp: int, default=0
            前処理の数。pp_dictの要素数と考えて良い。
    
    Examples:
    
    df = sns.load_dataset('titanic')
    ap = AnnnPreprocessing()
    ap.set_df(df)

    # 各カラムリストをセット
    ap.set_ignore_cols(ignore_cols)
    ap.set_cls_cols(cls_cols)
    ap.set_cate_cols(cate_cols)

    def cls_pp(df, cols_dict):
        df0 = df
        cols_dict0 = cols_dict
        cls_cols = cols_dict0['class']
        bins_dict = cols_dict0['bins']
        cate_cols = cols_dict0['category']

        cls_cols0 = cls_cols
        bins_dict0 = None
        cate_cols0 = cate_cols

        # 階級に分ける
        if len(cls_cols) >= 1:

            if bins_dict is None:
                bins_dict0 = {}
                # スタージェスの公式を用いて階級に分ける
                df0, bins_dict0 = get_classes(
                    in_df = df0,
                    columns = cls_cols,
                    n_classes = sturges_rule(len(df)),
                    drop = True
                )
            
            # すでにbins_dictに値がある場合
            else:
                for col in cls_cols:
                    col_band = str(col) + '_band'
                    df0[col_band] = df0[col].astype(float).apply(lambda x: get_section_text(x=x, bins=bins_dict[col_band], bounds=[False, True]))
                    df0.drop(col, axis=1) # カラム削除
                bins_dict0 = bins_dict # そのまま
            
            # カテゴリ変数に追加
            cate_cols += list(bins_dict0.keys())
            print('get_classes is done!')
        
        cols_dict0['class'] = cls_cols0
        cols_dict0['bins'] = bins_dict0
        cols_dict0['category'] = cate_cols0
        return df0, cols_dict0

    def dmm_pp(df, cols_dict):
        df0 = df
        cols_dict0 = cols_dict
        cate_cols = cols_dict0['category']
        dummy_cols = cols_dict0['dummy']

        cate_cols0 = cate_cols
        dummy_cols0 = []

        # カテゴリ変数をダミー変数化
        if len(cate_cols) >= 1:
            # ダミー変数化
            df0, dummy_cols0 = get_dummies(
                in_df = df0,
                dummy_na=True,
                columns=cate_cols,
                drop_first=False
            )
            cate_cols0 = []
            print('get_dummies is done!')

            # すでにダミー変数カラムが設定されている場合はそれに従う。
            if not (dummy_cols is None):
                cols0 = sorted((set(dummy_cols0)-set(dummy_cols)), key=dummy_cols.index) # difference
                cols1 = sorted((set(dummy_cols)-set(dummy_cols0)), key=dummy_cols.index) # intersection
                df0 = df0.drop(labels=cols0, axis=1)
                for ii in cols1:
                    df0[ii] = 0
            
            # ダミー変数カラムが設定されていない場合は新たに設定する。
            else:
                cols_dict0['dummy'] = dummy_cols0
        
        cols_dict0['category'] = cate_cols0
        return df0, cols_dict0

    def gen_pp(df, cols_dict):
        df0 = df
        cols_dict0 = cols_dict
        ignore_cols = cols_dict0['ignore']
        cls_cols = cols_dict0['class']
        bins_dict = cols_dict0['bins'] # dict
        cate_cols = cols_dict0['category']
        num_cols = cols_dict0['number']
        dummy_cols = cols_dict0['dummy']

        cls_cols0 = []
        bins_dict0 = None
        cate_cols0 = []
        dummy_cols0 = []

        # 前処理をまとめて行う
        df0, bins_dict0, dummy_cols0 = general_preprocessing(
            df = df0,
            ignore_cols = ignore_cols,
            cls_cols = cls_cols,
            bins_dict = bins_dict,
            cate_cols = cate_cols,
            num_cols = num_cols,
            scaling = False
        )

        # すでにダミー変数カラムが設定されている場合はそれに従う。
        if not (dummy_cols is None):
            cols0 = sorted((set(dummy_cols0)-set(dummy_cols)), key=dummy_cols.index) # difference
            cols1 = sorted((set(dummy_cols)-set(dummy_cols0)), key=dummy_cols.index) # intersection
            df0 = df0.drop(labels=cols0, axis=1)
            for ii in cols1:
                df0[ii] = 0
        
        # ダミー変数カラムが設定されていない場合は新たに設定する。
        else:
            cols_dict0['dummy'] = dummy_cols0

        cols_dict0['class'] = cls_cols0
        cols_dict0['bins'] = bins_dict0
        cols_dict0['category'] = cate_cols0
        return df0, cols_dict0

    # 前処理の追加
    # ap.add_pp(preprocessing=cls_pp)
    # ap.add_pp(preprocessing=dmm_pp)
    ap.add_pp(preprocessing=gen_pp)

    # 前処理内容の確認
    ap.check_pp_name()

    # 前処理の実行
    ap.exe_pp()
    df0 = ap.latest_df
    """
    # 最初のDataFrame
    first_df = None
    first_df_cols = []

    # 最新のDataFrame
    latest_df = None
    latest_df_cols = []

    # 目的変数
    target = None

    # 説明変数
    features = []

    # 種々のカラムのdict
    cols_dict = {}
    cols_dict['ignore'] = []
    cols_dict['number'] = []
    cols_dict['class'] = []
    cols_dict['bins'] = None
    cols_dict['category'] = []
    cols_dict['dummy'] = None
    cols_dict['dtime'] = []

    # 前処理をまとめた辞書
    pp_dict = {}

    # 前処理の数
    n_pp = 0


    def __init__(self):
        """
        前処理クラスのコンストラクタ。
        """
        self.first_df = None
        self.latest_df = None

        # 種々のカラムのリストをdictに格納
        self.cols_dict = {}
        self.cols_dict['ignore'] = []
        self.cols_dict['number'] = []
        self.cols_dict['class'] = []
        self.cols_dict['bins'] = None
        self.cols_dict['category'] = []
        self.cols_dict['dummy'] = None
        self.cols_dict['dtime'] = []

        # 前処理
        self.pp_dict = {}
        self.n_pp = len(self.pp_dict)


    def init_df(self):
        """
        前処理後のDataFrameの初期化を行う。
        つまり前処理前に戻す。
        """
        self.latest_df = self.first_df
        self.latest_df_cols = self.first_df_cols

        # 数値型
        num_df = self.latest_df.select_dtypes(include='number')
        self.cols_dict['number'] = num_df.columns.tolist()

        # 時系列(datetime型)
        num_df = self.latest_df.select_dtypes(include='datetime')
        self.cols_dict['datetime'] = num_df.columns.tolist()
        gc.collect()


    def set_df(self, df):
        """
        前処理の対象となるDataFrameをセットする。

        Args:
            df: DataFrame
                前処理の対象となるDataFrame
        """
        self.first_df = df
        self.first_df_cols = self.first_df.columns.tolist()
        
        # latest_dfを初期化する。
        self.init_df()


    def set_ignore_cols(self, cols):
        """
        前処理の対象外とするカラムを設定。
        目的変数やidなどを設定。

        Args:
            cols: list of str
                前処理の対象外とするカラムのリスト
        """
        if type(cols) == list:
            self.cols_dict['ignore'] = [ii for ii in cols if ii in self.latest_df_cols]
    

    def set_num_cols(self, cols):
        """
        数値型のカラムを設定。

        Args:
            cols: list of str
                数値型のカラムのリスト
        """
        if type(cols) == list:
            self.cols_dict['number'] = [ii for ii in cols if ii in self.latest_df_cols]
    

    def set_cls_cols(self, cols):
        """
        階級に分ける対象のカラムを設定。

        Args:
            cols: list of str
                階級に分ける対象のカラムのリスト
        """
        if type(cols) == list:
            self.cols_dict['class'] = [ii for ii in cols if ii in self.latest_df_cols]
    

    def set_bins_dict(self, bins_dict):
        """
        階級に分ける対象のカラムの階級の境界値リストを格納したdict。
        基本的に使わない。

        Args:
            bins_dict: dict
                binsを入れたdict。
        """
        if type(bins_dict) == dict:
            bins_dict0 = {}
            for key in bins_dict.keys():
                if key in self.latest_df_cols:
                    bins_dict0[key] = bins_dict[key]
            self.cols_dict['bins'] = bins_dict0


    def set_cate_cols(self, cols):
        """
        カテゴリー変数のカラムを設定。
        ダミー変数化などで利用。

        Args:
            cols: list of str
                カテゴリー変数のカラムのリスト
        """
        if type(cols) == list:
            self.cols_dict['category'] = [ii for ii in cols if ii in self.latest_df_cols]
    

    def set_dummy_cols(self, cols):
        """
        ダミー変数のカラムを設定。

        Args:
            cols: list of str
                ダミー変数のカラムのリスト
        """
        if type(cols) == list:
            self.cols_dict['dummy'] = [ii for ii in cols if ii in self.latest_df_cols]
    

    def set_dtime_cols(self, cols):
        """
        時系列(datetime型)のカラムを設定。

        Args:
            cols: list of str
                時系列(datetime型)のカラムのリスト
        """
        if type(cols) == list:
            self.cols_dict['dummy'] = [ii for ii in cols if ii in self.latest_df_cols]
    

    def add_pp(self, preprocessing=(lambda x: (x[0], x[1])), pp_num=None):
        """
        前処理を追加する。
        preprocessingは(前処理後DataFrame, 前処理cols_dict(種々のカラム))を返す
        関数でなければならない。

        Args:
            preprocessing: function, optional(default=(lambda x: x[0], x[1])
                DataFrameに前処理を施す関数。デフォルトでは恒等写像。
                次の形式でなければならない。

                Args:
                    df: DataFrame
                        元のDataFrame。
                    
                    cols_dict: dict of list
                        前処理実行前の種々のカラムのリストをまとめたdictionary。
                        各keyに対応するvalueは以下。
                        key一覧: ignore, number, class, bins, category, dummy, dtime

                        'ignore': list of str, default=[]
                            前処理の対象外となるカラムのリスト。
                        'number': list of str, default=[]
                            数値型のカラムのリスト。
                        'class': list of str, default=[]
                            階級で分ける対象のカラムのリスト。
                        'bins': dict
                            binsを入れたdict。
                        'category': list of str, default=[]
                            ダミー変数化前のカテゴリー変数のカラムのリスト。
                        'dummy': list of str, default=[]
                            ダミー変数のカラムのリスト。
                        'dtime': list of str, default=[]
                            時系列(datetime型)カラムのリスト。
                
                Returns:
                    df: DataFrame
                        前処理実行後のDataFrame。
                    
                    cols_dict: dict of list
                        前処理実行後の種々のカラムのリストをまとめたdictionary。
                        各keyに対応するvalueは以下。
                        key一覧: ignore, number, class, bins, category, dummy, dtime

                        'ignore': list of str, default=[]
                            前処理の対象外となるカラムのリスト。
                        'number': list of str, default=[]
                            数値型のカラムのリスト。
                        'class': list of str, default=[]
                            階級で分ける対象のカラムのリスト。
                        'bins': dict
                            binsを入れたdict。
                        'category': list of str, default=[]
                            ダミー変数化前のカテゴリー変数のカラムのリスト。
                        'dummy': list of str, default=[]
                            ダミー変数のカラムのリスト。
                        'dtime': list of str, default=[]
                            時系列(datetime型)カラムのリスト。
            
            pp_num: int, optional(default=None)
                preprocessingを前処理の何番目に追加するか。
                デフォルトでは最後に追加される。0始まりで考える。
        
        Returns:
            : bool
                追加の成否。Falseは追加失敗。
        
        Examples:
        
        df = sns.load_dataset('titanic')
        ap = AnnnPreprocessing()
        ap.set_df(df)

        # 前処理の追加
        # ap.add_pp(preprocessing=cls_pp)
        # ap.add_pp(preprocessing=dmm_pp)
        ap.add_pp(preprocessing=gen_pp)

        # 前処理内容の確認
        ap.check_pp_name()

        # 前処理の実行
        ap.exe_pp()
        df0 = ap.latest_df
        """
        if pp_num is None:
            pp_num = self.n_pp
        pp_keys = list(self.pp_dict.keys())
        
        # pp_dictのkeyを昇順に並べる
        pp_keys.sort()

        # pp_numがint型か調査
        if type(pp_num) != int:
            print('error: pp_num should be integer')
            return False
        
        pp_num = int(pp_num)
        if pp_num == self.n_pp:
            self.pp_dict[pp_num] = [str(preprocessing.__name__), preprocessing]
        elif pp_num > self.n_pp:
            pp_num = self.n_pp
            self.pp_dict[pp_num] = [str(preprocessing.__name__), preprocessing]
        elif pp_num < self.n_pp:
            tmp_dict = {}
            for ii in pp_keys:
                if ii < pp_num:
                    tmp_dict[ii] = self.pp_dict[ii]
                else:
                    tmp_dict[ii+1] = self.pp_dict[ii]
            tmp_dict[pp_num] = [str(preprocessing.__name__), preprocessing]
            tmp_keys = list(tmp_dict.keys())
            tmp_keys.sort()
            self.pp_dict = {}
            for ii in tmp_keys:
                self.pp_dict[ii] = tmp_dict[ii]
        
        elif pp_num < 0:
            print('error: pp_num < 0')
            return False
        self.n_pp = int(len(self.pp_dict))
        print('pp number:', pp_num)
        print('pp name:', self.pp_dict[pp_num][0])
        print('is added.')
        print()
        return True
    

    def delete_pp(self, pp_num):
        """
        前処理を前処理dictから削除する。

        Args:
            削除する前処理の番号
        """
        # pp_numがint型か調査
        if type(pp_num) != int:
            print('error: pp_num should be integer')
            return False
        
        pp_num = int(pp_num)
        if (0 <= pp_num)and(pp_num < self.n_pp):
            tmp_dict = self.pp_dict
            tmp_keys = list(tmp_dict.keys())
            tmp_keys.sort()
            self.pp_dict = {}
            for ii in tmp_keys:
                if ii < pp_num:
                    self.pp_dict[ii] = tmp_dict[ii]
                elif ii > pp_num:
                    self.pp_dict[ii-1] = tmp_dict[ii]
            print('pp number:', pp_num)
            print('pp name:', tmp_dict[pp_num][0])
            print('is deleted.')
            print()
        else:
            return False
        return True


    def check_pp_dict(self):
        """
        セットされた前処理の内容を確認する。
        前処理後は、
        ap = AnnnPreprocessing()
        ap.latest_df でDataFrameを取得。

        Returns:
            pp_name_dict: dict
                keys: int
                    pp number
                values: str
                    pp name
        """
        pp_keys = list(self.pp_dict.keys())
        
        # pp_dictのkeyを昇順に並べる
        pp_keys.sort()
        
        pp_name_dict = {}
        for key in pp_keys:
            pp_name = self.pp_dict[key][0]
            print('pp number:', key)
            print('pp name:', pp_name)
            print()
            pp_name_dict[key] = pp_name
        return pp_name_dict


    def exe_pp(self):
        """
        execute preprocessingのこと。前処理を全て実行する。
        DataFrameがセットされていないと実行されない。

        Returns:
            : bool
            前処理実行の成否
        """
        # Noneの場合
        if self.latest_df is None:
            print('error: latest_df is None')
            return False
        
        # latest_dfがDataFrameでない場合
        elif type(self.latest_df) != pd.DataFrame:
            print('error: latest_df is not DataFrame')
            return False

        pp_keys = list(self.pp_dict.keys())
        
        # pp_dictのkeyを昇順に並べる
        pp_keys.sort()

        # 一時的に前処理用のDataFrameを作成
        df = self.latest_df.copy()
        print('■■ execute all preprocessing from now!! ■■')
        print('before shape:', self.latest_df.shape)
        print()

        # 前処理対象外のカラムを隔離
        ignore_df = df[self.cols_dict['ignore']]
        df = df.drop(labels=self.cols_dict['ignore'], axis=1)

        # 種々のカラムのdict
        # ignore, number, class, bins, category, dummy, dtime
        cols_dict = self.cols_dict

        # pp_dictのkeyの数字が若い順に前処理を適用する
        for key in pp_keys:
            pp_name = self.pp_dict[key][0]
            pp_func = self.pp_dict[key][1]

            print('preprocessing start!:', pp_name)

            # DataFrameに前処理を適用する
            df, cols_dict = pp_func(df, cols_dict)

            print('preprocessing end!:', pp_name)
            print()
        
        # 無視していたカラムと結合
        df = pd.merge(df, ignore_df, left_index=True, right_index=True, how='outer')
        self.latest_df = df

        self.cols_dict = cols_dict

        print('■■ all preprocessing is done!! ■■')
        print('after shape:', self.latest_df.shape)
        print('______________________________')
        return True


def get_classes(in_df, columns=None, n_classes=1, drop=False, suffix='_band'):
    """
    DataFrameを階級に分ける。
    スタージェスの公式を用いた使用例は下記Examples参照。

    Args:
        in_df: DataFrame
            階級に分けたいDataFrame。
        
        columns: list of str, optional(default=None)
            階級に分けたいカラム名のリスト。
            数値型以外は無視される。
            Noneの時はin_dfの数値型の全てのカラムが指定される。
        
        n_classes: int, optional(default=10)
            階級の数。
        
        drop: bool, optional(default=True)
            元のカラムを削除するかどうか。
        
        suffix: str, optional(default='_band')
            バンドカラムの接尾語。

    Returns:
        out_df, bins_dict: tuple
            out_df: DataFrame
                階級に分けたDataFrame。
            
            bins_dict: dict
                binsを入れたdict。

    Examples:
    
    # タイタニック生存予測
    df = sns.load_dataset('titanic')
    # スタージェスの公式を用いる
    df0, bins_dict = get_classes(
        in_df = df,
        columns = ['age', 'fare'],
        n_classes = sturges_rule(len(df)),
        drop = True
    )
    # 階級に分けたものをダミー変数化する
    df1 = pd.get_dummies(df0, columns=list(bins_dict.keys()))
    """
    out_df = in_df.copy()

    # 階級値のlistを入れるdict
    bins_dict = {}

    # 数値型のカラムのみを指定
    num_df = out_df.select_dtypes(include='number')

    if columns is None:
        # Noneの時は数値型の全カラムを指定
        columns = num_df.columns.tolist()
    
    # out_dfのカラムにある
    columns = [ii for ii in columns if ii in out_df.columns.tolist()]
    for col in columns:
        col_band = col + '_band'
        out_df[col_band], bins = pd.cut(out_df[col], n_classes, retbins=True)
        
        bins_dict[str(col_band)] = [str(ii) for ii in list(bins)]

        if drop:
            # 元のカラムを削除
            out_df = out_df.drop(col, axis=1)
    
    return out_df, bins_dict


def get_dummies(in_df, columns=[], dummy_na=True, drop_first=False):
    """
    DataFrameを階級に分ける。
    スタージェスの公式を用いた使用例は下記Examples参照。

    Args:
        in_df: DataFrame
            階級に分けたいDataFrame。
        
        columns: list of str, optional(default=[])
            ダミー変数化したいカラム名のリスト。
            in_dfのカラムに含まれるもの以外無視される。
        
        dummy_na: bool, optional(default=True)]
            NA(not available)の値用のカラムを作成するかどうか。
            Trueで作成する。
        
        drop_first: bool, optional(default=False)
            ダミー変数化した後カラムを1つ削除するかどうか。
            Trueで削除する。

    Returns:
        ret: tuple
            out_df, dummy_cols
            out_df: DataFrame
                ダミー変数化した後のDataFrame。
            
            dummy_cols: list of str
                ダミー変数化によって生成されたカラムのリスト。
    """
    out_df = in_df.copy()
    df_cols = out_df.columns.tolist()
    cate_cols = [ii for ii in columns if ii in df_cols]
    not_cate_cols = sorted((set(df_cols)-set(cate_cols)), key=df_cols.index)

    # カテゴリ変数のカラムのみ
    cate_df = out_df[cate_cols]
    not_cate_df = out_df[not_cate_cols]

    dummied_cate_df = pd.get_dummies(cate_df, prefix_sep='_', dummy_na=dummy_na, columns=cate_cols, drop_first=drop_first)
    dummy_cols = dummied_cate_df.columns.tolist()

    out_df = pd.merge(not_cate_df, dummied_cate_df, left_index=True, right_index=True, how='inner')
    ret = out_df, dummy_cols
    return ret


# スケール変換
def make_scaling(in_df, columns=[], scaling=False):
    """
    スケール変換を行う。

    Args:
        in_df: DataFrame
            スケール変換前のDataFrame。
        
        columns: list of str, optional(default=[])
            スケーリング処理をする対象の数値カラムのリスト。
            何も指定しない場合は数値カラムのリスト全てが選択される。

        scaling: bool or str, optional(default=False)
            スケーリング処理をするかどうか。標準化と正規化。
            'standard': Standard scaling
                平均0、分散1にする。
            'minmax': MinMax scaling
                最小値0、最大値1にする。
    
    Returns:
        out_df: DataFrame
            スケール変換後のDatFrame。
    """
    out_df = in_df.copy()
    df_cols = out_df.columns.tolist()

    # 数値カラム
    num_cols = [ii for ii in columns if ii in df_cols]

    # スケール変換
    # 標準化
    if scaling == 'standard':
        if len(num_cols) >= 1:
            num_df = out_df[num_cols]
        else:
            num_df = out_df.select_dtypes(include='number')
            num_cols = num_df.columns.tolist()
        sscaler = StandardScaler() # インスタンスの作成
        sscaler.fit(num_df)
        sscaled = sscaler.transform(num_df)
        sscaled_df = pd.DataFrame(sscaled, columns=num_cols)
        out_df.drop(labels=num_cols, axis=1, inplace=True)
        out_df = pd.merge(out_df, sscaled_df, left_index=True, right_index=True, on='outer')
        print('standard scaling is done!')

    # 正規化
    elif scaling == 'minmax':
        if len(num_cols) >= 1:
            num_df = out_df[num_cols]
        else:
            num_df = out_df.select_dtypes(include='number')
            num_cols = num_df.columns.tolist()
        mmscaler = MinMaxScaler(feature_range=(0, 1)) # インスタンスの作成
        mmscaler.fit(num_df)
        mmscaled = mmscaler.transform(num_df)
        mmscaled_df = pd.DataFrame(mmscaled, columns=num_cols)
        out_df.drop(labels=num_cols, axis=1, inplace=True)
        out_df = pd.merge(out_df, mmscaled_df, left_index=True, right_index=True, on='outer')
        print('minmax scaling is done!')
    
    return out_df


def general_preprocessing(df, ignore_cols=[], cls_cols=[], bins_dict=None, cate_cols=[], num_cols=[], scaling=False):
    """
    一般的な前処理全般を行う関数。

    Args:
        df: DataFrame
            前処理を行う対象のDataFrame。
        
        ignore_cols: list of str, optional(default=[])
            前処理の対象外とするカラムのリスト。
            目的変数やidなどを設定。
        
        cls_cols: list of str, optional(default=[])
            階級に分ける対象のカラムのリスト。
        
        bins_dict: dict, optional(default=None)
            binsを入れたdict。
        
        cate_cols: list of str, optional(default=[])
            ダミー変数化する対象のカテゴリのカラムのリスト。
        
        num_cols: list of str, optional(default=[])
            スケーリング処理をする対象の数値カラムのリスト。

        scaling: bool or str, optional(default=False)
            スケーリング処理をするかどうか。
            'standard': Standard scaling
            'minmax': MinMax scaling
    
    Returns:
        ret: tuple
            df0, bins_dict0, dummy_cols0
            df0: DataFrame
                前処理済のDataFrame
            
            bins_dict0: dict
                binsを入れたdict
            
            dummy_cols0: list of str
                ダミー変数カラムのリスト。
    """
    df0 = df.copy()
    df_cols = df.columns.tolist()
    print('before shape:', df0.shape)

    # dfのcolumnsに含まれるもの以外無視する
    ignore_cols = [ii for ii in ignore_cols if ii in df_cols]
    cls_cols = [ii for ii in cls_cols if ii in df_cols]
    cate_cols = [ii for ii in cate_cols if ii in df_cols]
    num_cols = [ii for ii in num_cols if ii in df_cols]

    # 前処理対象外のカラムをチェック
    ignore_df = df0[ignore_cols]
    df0 = df0.drop(labels=ignore_cols, axis=1)

    # binsを入れたdict
    bins_dict0 = None

    # 階級に分ける
    if len(cls_cols) >= 1:

        if bins_dict is None:
            bins_dict0 = {}
            # スタージェスの公式を用いて階級に分ける
            df0, bins_dict0 = get_classes(
                in_df = df0,
                columns = cls_cols,
                n_classes = sturges_rule(len(df)),
                drop = True
            )
        
        # すでにbins_dictに値がある場合
        else:
            for col in cls_cols:
                col_band = str(col) + '_band'
                df0[col_band] = df0[col].astype(float).apply(lambda x: get_section_text(x=x, bins=bins_dict[col_band], bounds=[False, True]))
                df0 = df0.drop(col, axis=1) # カラム削除
            bins_dict0 = bins_dict # そのまま
        
        # カテゴリ変数に追加
        cate_cols += list(bins_dict0.keys())
        print('get_classes is done!')
    
    # ダミー変数化で生成されたカラム
    dummy_cols0 = []

    # カテゴリ変数をダミー変数化
    if len(cate_cols) >= 1:
        # ダミー変数化
        df0, dummy_cols0 = get_dummies(
            in_df = df0,
            dummy_na=True,
            columns=cate_cols,
            drop_first=False
        )
        print('get_dummies is done!')

    # スケール変換
    df0 = make_scaling(df0, columns=num_cols, scaling=scaling)

    # 無視していたカラムと結合
    df0 = pd.merge(df0, ignore_df, left_index=True, right_index=True, how='outer')
    print('after shape:', df0.shape)

    ret = df0, bins_dict0, dummy_cols0
    return ret


"""
AnnnPreprocessingのadd_pp()で追加するための関数群
hoge_pp(df, cols_dict)の形式で書く。
"""

# 階級に分ける
def cls_pp(df, cols_dict):
    """
    階級に分ける前処理。

    Args:
        df: DataFrame
            前処理前のDataFrame。

        cols_dict: dict
            前処理前の種々のカラムのdict。
    
    Returns:
        df0: DataFrame
            前処理後のDataFrame。

        cols_dict0: dict
            前処理後の種々のカラムのdict。
    """
    df0 = df
    cols_dict0 = cols_dict
    cls_cols = cols_dict0['class']
    bins_dict = cols_dict0['bins']
    cate_cols = cols_dict0['category']

    cls_cols0 = cls_cols
    bins_dict0 = None
    cate_cols0 = cate_cols

    # 階級に分ける
    if len(cls_cols) >= 1:

        if bins_dict is None:
            bins_dict0 = {}
            # スタージェスの公式を用いて階級に分ける
            df0, bins_dict0 = get_classes(
                in_df = df0,
                columns = cls_cols,
                n_classes = sturges_rule(len(df)),
                drop = True
            )
        
        # すでにbins_dictに値がある場合
        else:
            for col in cls_cols:
                col_band = str(col) + '_band'
                df0[col_band] = df0[col].astype(float).apply(lambda x: get_section_text(x=x, bins=bins_dict[col_band], bounds=[False, True]))
                df0.drop(col, axis=1) # カラム削除
            bins_dict0 = bins_dict # そのまま
        
        # カテゴリ変数に追加
        cate_cols += list(bins_dict0.keys())
        print('get_classes is done!')
    
    cols_dict0['class'] = cls_cols0
    cols_dict0['bins'] = bins_dict0
    cols_dict0['category'] = cate_cols0
    return df0, cols_dict0


# ダミー変数化
def dmm_pp(df, cols_dict):
    """
    ダミー変数化する前処理。

    Args:
        df: DataFrame
            前処理前のDataFrame。

        cols_dict: dict
            前処理前の種々のカラムのdict。
    
    Returns:
        df0: DataFrame
            前処理後のDataFrame。

        cols_dict0: dict
            前処理後の種々のカラムのdict。
    """
    df0 = df
    cols_dict0 = cols_dict
    cate_cols = cols_dict0['category']
    dummy_cols = cols_dict0['dummy']

    cate_cols0 = cate_cols
    dummy_cols0 = []

    # カテゴリ変数をダミー変数化
    if len(cate_cols) >= 1:
        # ダミー変数化
        df0, dummy_cols0 = get_dummies(
            in_df = df0,
            dummy_na=True,
            columns=cate_cols,
            drop_first=False
        )
        cate_cols0 = []
        print('get_dummies is done!')

        # すでにダミー変数カラムが設定されている場合はそれに従う。
        if not (dummy_cols is None):
            cols0 = sorted((set(dummy_cols0)-set(dummy_cols)), key=dummy_cols.index) # difference
            cols1 = sorted((set(dummy_cols)-set(dummy_cols0)), key=dummy_cols.index) # intersection
            df0 = df0.drop(labels=cols0, axis=1)
            for ii in cols1:
                df0[ii] = 0
        
        # ダミー変数カラムが設定されていない場合は新たに設定する。
        else:
            cols_dict0['dummy'] = dummy_cols0
    
    cols_dict0['category'] = cate_cols0
    return df0, cols_dict0


# 一般的前処理
def gen_pp(df, cols_dict):
    """
    一般的前処理。

    Args:
        df: DataFrame
            前処理前のDataFrame。

        cols_dict: dict
            前処理前の種々のカラムのdict。
    
    Returns:
        df0: DataFrame
            前処理後のDataFrame。

        cols_dict0: dict
            前処理後の種々のカラムのdict。
    """
    df0 = df
    cols_dict0 = cols_dict
    ignore_cols = cols_dict0['ignore']
    cls_cols = cols_dict0['class']
    bins_dict = cols_dict0['bins'] # dict
    cate_cols = cols_dict0['category']
    num_cols = cols_dict0['number']
    dummy_cols = cols_dict0['dummy']

    cls_cols0 = []
    bins_dict0 = None
    cate_cols0 = []
    dummy_cols0 = []

    # 前処理をまとめて行う
    df0, bins_dict0, dummy_cols0 = general_preprocessing(
        df = df0,
        ignore_cols = ignore_cols,
        cls_cols = cls_cols,
        bins_dict = bins_dict,
        cate_cols = cate_cols,
        num_cols = num_cols,
        scaling = False
    )

    # すでにダミー変数カラムが設定されている場合はそれに従う。
    if not (dummy_cols is None):
        cols0 = sorted((set(dummy_cols0)-set(dummy_cols)), key=dummy_cols.index) # difference
        cols1 = sorted((set(dummy_cols)-set(dummy_cols0)), key=dummy_cols.index) # intersection
        df0 = df0.drop(labels=cols0, axis=1)
        for ii in cols1:
            df0[ii] = 0
    
    # ダミー変数カラムが設定されていない場合は新たに設定する。
    else:
        cols_dict0['dummy'] = dummy_cols0

    cols_dict0['class'] = cls_cols0
    cols_dict0['bins'] = bins_dict0
    cols_dict0['category'] = cate_cols0
    return df0, cols_dict0



