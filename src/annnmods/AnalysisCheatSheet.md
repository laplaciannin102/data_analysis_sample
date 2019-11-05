# Load modules

```python
import sys, os
import gc
import time
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# 自作関数
# from annnmods import *
import annnmods as am
```

```python
am.print_usage()
```

# Configuration

## 左寄せにするマジックコマンド

```python
%%html
<style>
    table{float:left}
    .MathJax{float: left;}
</style>
```

## データフレームの表示設定

```python
# データフレームの表示行数、表示列数
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)

# カラム内の文字数。デフォルトは50。
pd.set_option("display.max_colwidth", 100)
```

## パスの設定

```python
# ファイルのパス
path = os.getcwd()
# 一応チェンジディレクトリしておく
os.chdir(path)
```

```python
gc.collect()
```

# Constants

## paths

```python
input_path = '../input/'
output_path = '../output/'
```

# Analysis

## python code memo

```python
# 欠損があるレコードを削除
df = df.dropna(subset=necessary)

# リストをソートする
li = [2, 1, 3]
li.sort()
print(li)      # [1, 2, 3]

# 順序を保ったままリストの重複を削除する
li = [1, 2, 3, 1, 3]
li = sorted(set(li), key=li.index)
print(li)       # [1, 2, 3]
```

## DataFrameの各カラムの情報を取得

```python
# タイタニック生存予測
df = sns.load_dataset('titanic')
var_info_df = am.check_var_info(df)
# var_info_dfのカラム名を日本語にしたい場合
enja_dict = {
    'file_name': 'ファイル名',
    'var_name': '変数名',
    'var_name_ja': '変数名日本語訳',
    'dtype': 'データ型',
    'n_unique': '値の種類数',
    'mode': '最頻値',
    'count_of_mode': '最頻値レコード数',
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
```

### output

# output of check_var_info
|   |  変数名   |変数名日本語訳|データ型|値の種類数|  最頻値   |最頻値レコード数| 欠損率 |非欠損数|欠損数|レコード数|平均値 |標準偏差|最小値|中央値|最大値|目的変数との相関|目的変数との相関_絶対値|
|--:|-----------|--------------|--------|---------:|-----------|---------------:|-------:|-------:|-----:|---------:|------:|-------:|-----:|-----:|-----:|---------------:|----------------------:|
|  0|survived   |生き残った    |int64   |         2|          0|             549|   0.000|     891|     0|       891| 0.3838|  0.4866|  0.00|  0.00|   1.0|         1.00000|                1.00000|
|  1|pclass     |pclass        |int64   |         3|          3|             491|   0.000|     891|     0|       891| 2.3086|  0.8361|  1.00|  3.00|   3.0|        -0.33848|                0.33848|
|  2|sex        |性別          |object  |         2|male       |             577|   0.000|     891|     0|       891|    NaN|     NaN|   NaN|   NaN|   NaN|             NaN|                    NaN|
|  3|age        |上げ          |float64 |        88|         24|              30|   0.199|     714|   177|       891|29.6991| 14.5265|  0.42| 28.00|  80.0|        -0.07722|                0.07722|
|  4|sibsp      |同胞          |int64   |         7|          0|             608|   0.000|     891|     0|       891| 0.5230|  1.1027|  0.00|  0.00|   8.0|        -0.03532|                0.03532|
|  5|parch      |尊敬する      |int64   |         7|          0|             678|   0.000|     891|     0|       891| 0.3816|  0.8061|  0.00|  0.00|   6.0|         0.08163|                0.08163|
|  6|fare       |やります      |float64 |       248|       8.05|              43|   0.000|     891|     0|       891|32.2042| 49.6934|  0.00| 14.45| 512.3|         0.25731|                0.25731|
|  7|embarked   |乗船した      |object  |         3|S          |             644|   0.002|     889|     2|       891|    NaN|     NaN|   NaN|   NaN|   NaN|             NaN|                    NaN|
|  8|class      |クラス        |category|         3|Third      |             491|   0.000|     891|     0|       891|    NaN|     NaN|   NaN|   NaN|   NaN|             NaN|                    NaN|
|  9|who        |誰            |object  |         3|man        |             537|   0.000|     891|     0|       891|    NaN|     NaN|   NaN|   NaN|   NaN|             NaN|                    NaN|
| 10|adult_male |成人男性      |bool    |         2|True       |             537|   0.000|     891|     0|       891|    NaN|     NaN|   NaN|   NaN|   NaN|        -0.55708|                0.55708|
| 11|deck       |デッキ        |category|         7|C          |              59|   0.772|     203|   688|       891|    NaN|     NaN|   NaN|   NaN|   NaN|             NaN|                    NaN|
| 12|embark_town|embark_town   |object  |         3|Southampton|             644|   0.002|     889|     2|       891|    NaN|     NaN|   NaN|   NaN|   NaN|             NaN|                    NaN|
| 13|alive      |生きている    |object  |         2|no         |             549|   0.000|     891|     0|       891|    NaN|     NaN|   NaN|   NaN|   NaN|             NaN|                    NaN|
| 14|alone      |一人で        |bool    |         2|True       |             537|   0.000|     891|     0|       891|    NaN|     NaN|   NaN|   NaN|   NaN|        -0.20337|                0.20337|


## Preprocessing

```python
# titanic生存予測
df = sns.load_dataset('titanic')

# 前処理クラスのインスタンス化
ap = AnnnPreprocessing()

# 目的変数
target = 'survived'
df = df.drop('alive', axis=1)

# 無視するカラム
ignore_cols = [target]

# 階級に分けるカラム
cls_cols = ['age', 'fare']

# カテゴリ変数のカラム
cate_cols = [
    'pclass',
    'sex',
    'sibsp',
    'parch',
    'embarked',
    'class',
    'who',
    'adult_male',
    'deck',
    'embark_town',
    'alone'
]

# DataFrameのセット
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
```

## グラフの描画を行う

```python
# グラフの描画
names = ['太郎', '次郎', '三郎', '四郎', '五郎', '六郎']
n = len(names)
np.random.seed(seed=57)
df = pd.DataFrame({'name': names})
df['パン'] = np.random.randint(0, 300, n)
df['お米'] = np.random.randint(0, 300, n)
am.ratio_graph(df['name'], df['パン'], df['お米'], xlabel='名前', ylabel='食事回数', y1label='お米', y2label='パン', ratio_label='パンの割合')
```

