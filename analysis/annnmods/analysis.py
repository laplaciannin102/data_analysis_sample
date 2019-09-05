#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: kosuke.asada
分析用関数
"""


import gc
import time
import pandas as pd
import codecs
import seaborn as sns

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 評価関数
# 分類
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

# 回帰
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 設定のimport
from .mod_config import *
# 自作関数のimport
from .calculation import *
from .scraping import *
from .useful import *
from .visualization import *


def pro_read_csv(path, encoding='utf-8', usecols=None):
    """
    pd.read_csv()で読めないcsvファイルを読み込む。

    Args:
        path: str
            読み込むファイルのパス。

        encoding: str, optional(default='utf-8)
            エンコード。

        usecols: list of str, optional(default=None)
            指定したカラムのみを読み込む場合に使用。

    Returns:
        df: DataFrame
            読み込んだDataFrame。
    """
    if usecols is None:
        print('usecols: all columns')
        with codecs.open(path, 'r', encoding, 'ignore') as file:
            df = pd.read_table(file, delimiter=',')
    else:
        # 指定したカラムのみ読み込む場合
        print('usecols:', usecols)
        with codecs.open(path, 'r', encoding, 'ignore') as file:
            df = pd.read_table(file, delimiter=',', usecols=usecols)

    return df


def check_var_info(df, target=None, file_name='', filecol_is=False, transcol_is=True):
    """
    DataFrameの各カラムの示す変数の概要をまとめる。

    Args:
        df: DataFrame
            概要を調べるDataFrame。
        
        target: str, optional(default=None)
            目的変数となるカラム名。Noneの時はtarget関連のカラムは作成しない。
        
        file_name: str, optional(default='')
            DataFrameが保存されていたファイルの名前。一応パスではない想定。
        
        filecol_is: bool, optional(default=False)
            file_nameカラムを作成するかどうか。Trueで作成する。
        
        transcol_is: bool, optional(default=True)
            カラム名の日本語訳を行うかどうか。Trueで行う。

    Returns:
        var_info_df: DataFrame
            各カラムの示す変数の概要をまとめたDataFrame。
            columns:
                file_name:          ファイル名
                var_name:           変数名
                var_name_ja:        変数名の日本語訳
                dtype:              データ型
                n_unique:           値の種類数
                major_vals:         最も多数派の値
                count_of_major:     多数派が占めるレコード数
                missing_rate:       欠損率
                n_exist:            非欠損数
                n_missing:          欠損数
                n_rows:             行数。レコード数。
                mean:               平均値
                std:                標準偏差
                min:                最小値
                med:                中央値
                max:                最大値
                corr_with_target:   目的変数との相関
                abs_corr_with_target: 目的変数との相関の絶対値
    
    Examples:
    
    # タイタニック生存予測
    titanic_df = sns.load_dataset('titanic')
    var_info_df = check_var_info(titanic_df)
    # var_info_dfのカラム名を日本語にしたい場合
    enja_dict = {
        'file_name': 'ファイル名',
        'var_name': '変数名',
        'var_name_ja': '変数名日本語訳',
        'dtype': 'データ型',
        'n_unique': '値の種類数',
        'major_vals': '最多数値',
        'count_of_major': '多数値レコード数',
        'missing_rate': '欠損率',
        'n_exist': '非欠損数',
        'n_missing': '欠損数',
        'n_rows': 'レコード数',
        'mean': '平均値',
        'std': '標準偏差',
        'min': '最小値',
        'med': '中央値',
        'max': '最大値',
        'corr_with_target': '目的変数との相関',
        'abs_corr_with_target': '目的変数との相関_絶対値'
    }
    var_info_df = var_info_df.rename(columns=enja_dict)
    """
    # 各変数についてまとめたい事柄(今後増やしていきたい)
    var_info_cols = [
        'file_name',        # ファイル名
        'var_name',         # 変数名
        'var_name_ja',      # 変数名の日本語訳
        'dtype',            # データ型
        'n_unique',         # 値の種類数
        'major_vals',       # 最も多数派の値
        'count_of_major',   # 多数派が占めるレコード数
        'missing_rate',     # 欠損率
        'n_exist',          # 非欠損数
        'n_missing',        # 欠損数
        'n_rows'            # 行数。レコード数。
    ]

    key = 'var_name'
    var_info_df = pd.DataFrame(columns=[key])
    df_cols = df.columns.tolist()
    var_info_df[key] = df_cols

    # 基礎統計料
    basic_stat_cols = ['mean', 'std', 'min', 'med','max']
    var_info_cols += basic_stat_cols

    # targetとの相関
    # targetがNoneでなく、df_colsの中にある時
    if (not target is None) and (target in df_cols):
        corr_cols = ['corr_with_target', 'abs_corr_with_target']
        var_info_cols += corr_cols

    if not filecol_is:
        var_info_cols.remove('file_name')
    
    if not transcol_is:
        var_info_cols.remove('var_name_ja')
    
    # ファイル名
    if filecol_is:
        var_info_df['file_name'] = file_name

    # 日本語訳
    if transcol_is:
        var_info_df['var_name_ja'] = var_info_df[key].apply(translate_to_ja)
        time.sleep(0.4)

    # データ型
    dtype_df = pd.DataFrame(df.dtypes).reset_index()
    dtype_df.columns = [key, 'dtype']
    var_info_df = pd.merge(var_info_df, dtype_df, on=key, how='inner')

    # 値の種類数
    nunique_df = pd.DataFrame(df.nunique()).reset_index()
    nunique_df.columns = [key, 'n_unique']
    var_info_df = pd.merge(var_info_df, nunique_df, on=key, how='inner')

    # 最も多数派の値
    major_list = []
    for var_name in df_cols:
        vc_df = pd.DataFrame(df[var_name].value_counts())
        cmx = vc_df[var_name].max()
        if cmx >= 2:
            vc_df = vc_df[vc_df[var_name]==cmx]
            major_vals = vc_df.index.tolist()
            major_vals_text = ''
            for val in major_vals:
                major_vals_text += str(val) + '/'
            major_vals_text = major_vals_text[:-1]
        else:
            # 1の時は多数派の概念を考えない
            major_vals_text = ''
        major_list += [[var_name, major_vals_text, cmx]]
    major_df = pd.DataFrame(major_list, columns=[key, 'major_vals', 'count_of_major'])
    var_info_df = pd.merge(var_info_df, major_df, on=key, how='inner')
    

    # 非欠損数
    n_exist_df = pd.DataFrame(df.notnull().sum()).reset_index()
    n_exist_df.columns = [key, 'n_exist']
    var_info_df = pd.merge(var_info_df, n_exist_df, on=key, how='inner')

    # 欠損数
    n_missing_df = pd.DataFrame(df.isnull().sum()).reset_index()
    n_missing_df.columns = [key, 'n_missing']
    var_info_df = pd.merge(var_info_df, n_missing_df, on=key, how='inner')

    # 行数
    var_info_df['n_rows'] = len(df)

    # 欠損率
    e = 'n_exist'
    m = 'n_missing'
    var_info_df['missing_rate'] = var_info_df.apply(lambda row: row[m]/(row[e]+row[m]) if ((row[e]+row[m])!=0) else 0, axis=1)

    # 基礎統計量
    bst_df = df.describe()
    bst_df = bst_df.T.rename(columns={'50%': 'med'})[basic_stat_cols].reset_index(drop=False)
    bst_df.columns = [key] + basic_stat_cols
    var_info_df = pd.merge(var_info_df, bst_df, on=key, how='outer')

    # targetとの相関
    # targetがNoneでなく、df_colsの中にある時
    if (not target is None) and (target in df_cols):
        corr_df = df.corr()
        target_corr_df = corr_df[[target]]
        target_corr_df[corr_cols[1]] = target_corr_df[target].apply(abs)
        target_corr_df = target_corr_df.reset_index()
        target_corr_df.columns = [key, corr_cols[0], corr_cols[1]]
        var_info_df = pd.merge(var_info_df, target_corr_df, on=key, how='outer')
        del corr_df
        gc.collect()


    # 整理
    var_info_df = var_info_df.reset_index(drop=True)[var_info_cols]

    return var_info_df


def check_data_list(path_list, encoding='utf-8'):
    """
    各ファイルの概要をまとめたDataFrameを返す。
    csvとxlsxに対応。

    Args:
        path_list: list of str
            概要を知りたいファイルパスのリスト。

        encoding: str, optional(default='utf-8)
            各ファイルを読み込むときのエンコード。

    Returns:
        data_info_df, var_info_df: tuple
            data_info_df: DataFrame
                データの概要をまとめたDataFrame。

            var_info_df: DataFrame
                カラムの概要をまとめたDataFrame。
                check_var_info()参照。
    """
    n_files = len(path_list)    # file数
    print('num of all files:', n_files)

    # dataの概要をまとめたDataFrame
    info_cols = ['path', 'file_name', 'n_rows', 'n_cols', 'columns']
    data_info_df = pd.DataFrame(columns=info_cols)

    # 各カラムの概要をまとめたDataFrame
    var_info_cols = [
        'file_name',        # ファイル名
        'var_name',         # 変数名
        'var_name_ja',      # 変数名の日本語訳
        'dtype',            # データ型
        'n_unique',         # 値の種類数
        'major_vals',       # 最も多数派の値
        'count_of_major',   # 多数派が占めるレコード数
        'missing_rate',     # 欠損率
        'n_exist',          # 非欠損数
        'n_missing',        # 欠損数
        'n_rows'            # 行数。レコード数。
    ]
    # 基礎統計料
    basic_stat_cols = ['mean', 'std', 'min', 'med','max']
    var_info_cols += basic_stat_cols
    var_info_df = pd.DataFrame(columns=var_info_cols)

    # 各ファイルについてまとめる
    for path in path_list:
        path = str(path)
        file_name = path.split('/')[-1].split('\\')[-1]     # ファイル名
        extension = file_name.split('.')[-1]     # 拡張子

        if extension == 'csv':
            df = pro_read_csv(path, encoding=encoding)
        elif extension == 'xlsx':
            df = pd.read_excel(path, index=False)
        else:
            # csvでもxlsxでも無ければ一旦無視する。
            continue

        shape = df.shape
        n_rows = int(shape[0])   # 行数。レコード数。
        n_cols = int(shape[1])   # 列数。カラム数。
        columns = df.columns.tolist()   # カラム一覧。

        tmp = pd.DataFrame([[path, file_name, n_rows, n_cols, columns]], columns=info_cols)
        data_info_df = data_info_df.append(tmp)

        var_tmp = check_var_info(df, target=None, file_name=file_name, filecol_is=True, transcol_is=True)
        var_info_df = var_info_df.append(var_tmp)

        # 掃除
        del df, tmp, var_tmp
        gc.collect()

    data_info_df = data_info_df.reset_index(drop=True)
    var_info_df = var_info_df.reset_index(drop=True)
    print('num of files of data:', len(data_info_df))

    # 戻り値
    ret = data_info_df, var_info_df
    return ret


def print_clf_score(true_y, pred_y):
    """
    分類モデルの評価関数の値を表示する。
    sklearn.metrics.classification_report も参考に。

    Args:
        true_y: 1d array-like, or label indicator array / sparse matrix
            実測値。

        pred_y: 1d array-like, or label indicator array / sparse matrix
            予測値。
    
    Returns:
        score_df, cm_df: tuple
            score_df: DataFrame
                分類問題の評価関数の値をまとめたDataFrame。
                columns: Accuracy, F-measure, Precision, Recall, Log_Loss, ROC_AUC
            
            cm_df: DataFrame
                混同行列のDataFrame。
    
    Remarks:
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """
    acc = accuracy_score(true_y, pred_y)
    f1 = f1_score(true_y, pred_y)
    precision = precision_score(true_y, pred_y)
    recall = recall_score(true_y, pred_y)
    l_loss = log_loss(true_y, pred_y)
    roc_auc = roc_auc_score(true_y, pred_y)

    score_df = pd.DataFrame(
        [[acc, f1, precision, recall, l_loss, roc_auc]],
        columns=['Accuracy', 'F-measure', 'Precision', 'Recall', 'Log_Loss', 'ROC_AUC']
        )

    # 混同行列
    cm_df = confusion_matrix(pred_y, true_y)

    print('正解率:', pro_round(acc, 2))
    print('f1:', pro_round(f1, 2))
    print('適合率(precision):', pro_round(precision,2))
    print('再現率(recall):', pro_round(recall, 2))
    print('log loss:', pro_round(l_loss, 2))
    print('ROC AUC:', pro_round(roc_auc, 2))
    
    return score_df, cm_df


def print_reg_score(true_y, pred_y):
    """
    回帰モデルの評価関数の値を表示する。

    Args:
        true_y: 1d array-like, or label indicator array / sparse matrix
            実測値。

        pred_y: 1d array-like, or label indicator array / sparse matrix
            予測値。
    
    Returns:
        score_df: DataFrame
            回帰問題の評価関数の値をまとめたDataFrame。
            columns: MSE, R2_score
    
    Remarks:
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    """
    mse = mean_squared_error(true_y, pred_y)
    r2 = r2_score(true_y, pred_y)

    score_df = pd.DataFrame(
        [[mse, r2]],
        columns=['MSE', 'R2_score']
        )

    print('平均二乗誤差(mean squared error):', pro_round(mse, 2))
    print('決定係数(R^2):', pro_round(r2, 2))
    
    return score_df


def extract_isone(row, cols=None, sep='/'):
    """
    辞書型のrowの指定された各カラムのうち1かTrueであるものを抜き出す。
    ダミー変数化した後のDataFrameなどに使用。
    DataFrameに対してapply()の中で使用する想定。
    使用例は下記Examples参照。

    Args:
        row: dict
            DataFrameの各行を表すdict。
        
        cols: list of str or tuple of str, optional(default=None)
            判定したいカラム名のリスト。
            初期設定のNoneが選ばれた場合、rowのkey全てが選ばれる。
        
        sep: str, optional(default='/')
            出力で使用する文字列の区切り文字。
    
    Returns:
        isone_list, isone_text: tuple
            isone_list: list of str
                値が1かTrueであるカラム名のリスト。
            
            isone_text: str
                値が1かTrueであるカラム名を繋げて一つの文字列にしたもの。

    Examples:

    df['isone'] = df.apply(lambda row: extract_isone(row)[1], axis=1)
    """
    # 宣言
    isone_list = []
    isone_text = ''

    if cols is None:
        # colsがNoneの時はrowのkey全てを対象にする
        cols = list(row.keys())
    else:
        cols = list(cols)
    
    for ii in cols:
        if (row[ii] == 1) or (row[ii] == '1') or (row[ii] == True):
            isone_list += [str(ii)]
            isone_text += str(ii) + sep
    
    # isone_textが2文字以上の時
    if len(isone_text) >= 2:
        # 最後の1文字を消す
        isone_text = isone_text[:-1]
    
    return isone_list, isone_text


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
                binsを入れたdict

    Examples:
    
    # タイタニック生存予測
    titanic_df = sns.load_dataset('titanic')
    # スタージェスの公式を用いる
    df1, bins = get_classes(
        in_df = titanic_df,
        columns = ['age', 'fare'],
        n_classes = sturges_rule(len(titanic_df)),
        drop = True
        )
    # 階級に分けたものをダミー変数化する
    df2 = pd.get_dummies(df1, columns=bins.keys())
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
        
        bins_dict[str(col_band)] = list(bins)

        if drop:
            # 元のカラムを削除
            out_df = out_df.drop(col, axis=1)
    
    return out_df, bins_dict


def get_cross(df, target=None, cols=[], normalize=False):
    """
    クロス集計をする。pd.crosstabを使用。

    Args:
        df: DataFrame
            クロス集計をしたいDataFrame。
        
        target: str, optional(default=None)
            カウント対象の変数名。目的変数などを想定。
            Noneの時は該当するレコード数をカウントするのみ。
        
        cols: list of str, optional(default=[])
            クロス集計対象となるカラム一覧。dfに存在するカラムのみを想定。
        
        normalize: bool, optional(default=False)
            カウントだけでなく、割合も計算するかどうか。
            Trueの時計算する。targetを1つ指定している時のみ使用可能。
    
    Returns:
        cross_df: DataFrame
            クロス集計を行ったDataFrame。
    """
    tmp = df.copy()
    
    # カテゴリカルな説明変数などをクロス集計のindexに使用するためセット
    cross_idxes = []
    # dfに存在するカラムのみにする
    cols = [str(ii) for ii in cols if str(ii) in tmp.columns.tolist()]
    for col in cols:
        cross_idxes.append(tmp[col])
    
    # 目的変数などをクロス集計のcolumnsに使用するためセット
    cross_cols = []
    if (type(target) == str)and(str(target) in tmp.columns.tolist()):
        cross_cols = [tmp[target]]
    
    # クロス集計を算出
    cross_df = pd.crosstab(index=cross_idxes, columns=cross_cols, margins=True, normalize=False).reset_index(drop=False)
    cross_df = cross_df.rename(columns={'__dummy__': 'all_count'})
    
    if len(cross_cols) == 1:
        cross_df = cross_df.rename(columns=(lambda x: x if not(x in tmp[target].unique().tolist()) else target + '_' + str(x) + '_count'))
    
    # 割合のクロス集計を算出
    if (normalize)and(len(cross_cols) == 1):
        norm_cross_df = pd.crosstab(index=cross_idxes, columns=cross_cols, margins=False, normalize='index').reset_index(drop=False)
        norm_cross_df.drop(labels=cols, inplace=True, axis=1)
        norm_cross_df = norm_cross_df.rename(columns=(lambda x: x if not(x in tmp[target].unique().tolist()) else target + '_' + str(x) + '_ratio'))
        cross_df = pd.merge(cross_df, norm_cross_df, left_index=True, right_index=True, how='outer')
    del cross_df.columns.name
    gc.collect()

    return cross_df


# score_func: 評価関数
def rfc_gridsearch(data_x, data_y, param_grid, evaluation='accuracy', n_cv=10):
    """
    sklearn.ensemble.RandomForestClassifier()の
    GridSearchを行う。

    Args:
        data_x: array-like or sparse matrix of shape = [n_samples, n_features]
            説明変数。
        
        data_y: array-like, shape = [n_samples] or [n_samples, n_outputs]
            目的変数。
        
        param_grid: dict
            変化させるパラメータの値。
        
        evaluation: str, optional(default='accuracy')
            評価関数。
            評価関数evaluationで指定できるもの: 
                ‘accuracy’	metrics.accuracy_score
                ‘balanced_accuracy’	metrics.balanced_accuracy_score	for binary targets
                ‘average_precision’	metrics.average_precision_score
                ‘brier_score_loss’	metrics.brier_score_loss
                ‘f1’	metrics.f1_score	for binary targets
                ‘f1_micro’	metrics.f1_score	micro-averaged
                ‘f1_macro’	metrics.f1_score	macro-averaged
                ‘f1_weighted’	metrics.f1_score	weighted average
                ‘f1_samples’	metrics.f1_score	by multilabel sample
                ‘neg_log_loss’	metrics.log_loss	requires predict_proba support
                ‘precision’ etc.	metrics.precision_score	suffixes apply as with ‘f1’
                ‘recall’ etc.	metrics.recall_score	suffixes apply as with ‘f1’
                ‘roc_auc’
        
        n_cv: int, optional(default=10)
            grid searchでcross validationをする際の分割数。
    
    Returns:
        gs: instance of GridSearchCV
            grid searchのインスタンス。
    
    Examples:
    
    evaluations_list = ['accuracy', 'f1', 'precision', 'recall']
    for eval_func in evaluations_list:
        gs = rfc_gridsearch(data_X, data_y, evaluation=eval_func)
        print('__________________________________________________________________  ')
    
    Remarks:
        RandomForestClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        GridSearchCV:
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """
    # ランダムフォレストのインスタンスを作成
    rfc_gs = RandomForestClassifier(random_state=57)

    # ハイパーパラメータ値のリスト
    if type(param_grid) == dict:
        param_grid = [param_grid]
    else:
        param_grid = [{
            'n_estimators':[i for i in range(10,50,5)],
            'criterion':['gini','entropy'],
            'max_depth':[i for i in range(1,21,1)],
            'min_samples_leaf':[i for i in range(1, 10, 1)],
            'bootstrap':[False],
            'random_state':[57]
        }]

    # ハイパーパラメータ値のリストparam_gridを指定し、
    # グリッドサーチを行うGridSearchCVクラスをインスタンス化
    gs = GridSearchCV(
        estimator = rfc_gs,
        param_grid = param_grid,
        scoring = evaluation,
        cv = n_cv,
        n_jobs = 5
    )

    # fitさせる
    gs = gs.fit(data_x, data_y)

    # 評価関数
    print('evaluation:', evaluation, '  ')
    
    # モデルの最良スコアを出力
    print('best score:', gs.best_score_, '  ')

    # 最良スコアとなるパラメータ値を出力
    print('best params:', gs.best_params_, '  ')

    return gs


def general_preprocessing(df, ignore_cols=[], cls_cols=[], cate_cols=[], scaling=False, num_cols=[]):
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
        
        cate_cols: list of str, optional(default=[])
            ダミー変数化する対象のカテゴリのカラムのリスト。
        
        scaling: bool or str, optional(default=False)
            スケーリング処理をするかどうか。
            'standard': Standard scaling
            'minmax': MinMax scaling
        
        num_cols: list of str, optional(default=[])
            スケーリング処理をする対象の数値カラムのリスト。
    
    Returns:
        df0: DataFrame
            前処理済のDataFrame
    """
    df0 = df.copy()
    print('before shape:', df0.shape)

    # 前処理対象外のカラムをチェック
    ignore_df = df0[ignore_cols]
    df0 = df0.drop(labels=ignore_cols, axis=1)

    # 階級に分ける
    if len(cls_cols) >= 1:
        # スタージェスの公式を用いて階級に分ける
        df0, bins = get_classes(
            in_df = df,
            columns = cls_cols,
            n_classes = sturges_rule(len(df)),
            drop = True
            )
        # カテゴリ変数に追加
        cate_cols += list(bins.keys())
        print('get_classes is done!')

    # カテゴリ変数をダミー変数化
    if len(cate_cols) >= 1:
        # ダミー変数化
        df0 = pd.get_dummies(df0, dummy_na=True, columns=cate_cols)
        print('get_dummies is done!')

    # スケール変換
    if scaling == 'standard':
        if len(num_cols) >= 1:
            num_df = df0[num_cols]
        else:
            num_df = df0.select_dtypes(include='number')
            num_cols = num_df.columns.tolist()
        sscaler = StandardScaler() # インスタンスの作成
        sscaler.fit(num_df)
        sscaled = sscaler.transform(num_df)
        sscaled_df = pd.DataFrame(sscaled, columns=num_cols)
        df0.drop(labels=num_cols, axis=1, inplace=True)
        df0 = pd.merge(df0, sscaled_df, left_index=True, right_index=True, on='outer')
        print('standard scaling done!')

    elif scaling == 'minmax':
        if len(num_cols) >= 1:
            num_df = df0[num_cols]
        else:
            num_df = df0.select_dtypes(include='number')
            num_cols = num_df.columns.tolist()
        mmscaler = MinMaxScaler(feature_range=(0, 1)) # インスタンスの作成
        mmscaler.fit(num_df)
        mmscaled = mmscaler.transform(num_df)
        mmscaled_df = pd.DataFrame(mmscaled, columns=num_cols)
        df0.drop(labels=num_cols, axis=1, inplace=True)
        df0 = pd.merge(df0, mmscaled_df, left_index=True, right_index=True, on='outer')
        print('minmax scaling done!')
    
    # 無視していたカラムと結合
    df0 = pd.merge(df0, ignore_df, left_index=True, right_index=True, how='outer')
    print('after shape:', df0.shape)

    return df0
    

