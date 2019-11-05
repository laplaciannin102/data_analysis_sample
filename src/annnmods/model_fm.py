#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: kosuke.asada
モデル作成用module
"""


import sys, os
import gc
import time
import numpy as np
import pandas as pd


class ClassifierModel():
    """
    分類モデルの諸々を管理する。

    Attributes:
        clf: instance of Classifier
            分類モデルのインスタンス。

        target: str
            目的変数。

        features: list of str
            特徴量。
        
        is_trained: bool
            学習済みフラグ
    """
    clf = None
    clf_attributes = None

    # 目的変数
    target = 'target'

    # 特徴量
    features = []
    """
    # 保持しないことにする
    # 教師データ
    train_df = None
    train_x = None
    train_y = None
    """
    train_shape = (0, 0)

    # 寄与率
    importances = None

    # 学習済みフラグ
    is_trained = False


    def __init__(self, clf=None, args=None, target='target', features=[]):
        """
        コンストラクタ。

        Args:
            args: dict
                分類フォレストの引数。

            target: str
                目的変数。

            features: list of str
                特徴量。
        """
        self.clf = clf
        self.args = args
        self.target = target
        self.features = features
        self.importances = None
        self.is_trained = False
    

    def set_clf_model(self, clf):
        """
        分類モデルの属性を設定する。

        Args:
            args: dict
                分類モデルの引数。
        """
        self.clf = clf


    def set_clf_args(self, args):
        """
        分類モデルの属性を設定する。

        Args:
            args: dict
                分類モデルの引数。
        """
        self.args = args


    def train(self, train_df):
        """
        学習する。

        Args:
            train_df: DataFrame
                教師データ。
        """
        train_cols = train_df.columns.tolist()
        if (type(self.target)==str)and(len(self.features)>=1):
            if (self.target in train_cols)and(set(self.features) <= set(train_cols)):
                # self.train_df = train_df
                # self.train_x = self.train_df[self.features]
                # self.train_y = self.train_df[self.target]
                # self.rfc.fit(self.train_x, self.train_y)
                train_x = train_df[self.features]
                train_y = train_df[self.target]

                # train_xyで学習
                self.train_xy(train_x, train_y)
            else:
                print('train error: there is an error in columns')
        else:
            print('train error: there is an error in columns')

    
    def train_xy(self, train_x, train_y):
        """
        学習する。

        Args:
            train_x: DataFrame
                教師データの特徴量部分。

            train_y: 1d array-like, or label indicator array / sparse matrix
                教師データの目的変数部分。
        """
        if len(train_x) == len(train_y):
            # self.train_x = train_x
            # self.train_y = train_y
            # self.train_df = pd.concat([self.train_x, self.train_y], axis=1, join='outer')
            self.features = train_x.columns.tolist()

            # self.clf.fit(train_x, train_y)

            # 学習済みフラグ
            self.is_trained = True
            print('model is trained')
        else:
            print('train error: length of x and length of y is different')


    def predict(self, data_x):
        """
        data_xからyを予測する。
        pred_y, prob_y

        Args:
            data_x: DataFrame
                説明変数。
        
        Returns:
            ret: tuple of DataFrame
                pred_y, prob_y
                pred_y:
                    目的変数の予測値。
                prob_y:
                    目的変数が各クラスに分類される確率。
        """
        pred_y = self.clf.predict(data_x)
        prob_y = self.clf.predict_proba(data_x)
        ret = pred_y, prob_y
        gc.collect()
        return ret
    

    def validate(self, data_x, data_y, class_num=1):
        """
        検証。
        result_df, unique_df, score_df, cm_df

        Args:
            data_x: DataFrame
                説明変数。

            data_y: 1d array-like, or label indicator array / sparse matrix
                目的変数。
        
        Returns:
            ret: tuple of DataFrame
                result_df, unique_df, score_df, cm_df
        """
        pred_y = self.clf.predict(data_x)
        prob_y = self.clf.predict_proba(data_x)

        rm = ResultManager(model_type='clf')
        rm.set_clf_result(data_x, data_y, pred_y, prob_y, class_num=class_num, target=self.target)
        result_df, unique_df, score_df, cm_df = rm.get_result()
        ret = result_df, unique_df, score_df, cm_df
        gc.collect()
        return ret


