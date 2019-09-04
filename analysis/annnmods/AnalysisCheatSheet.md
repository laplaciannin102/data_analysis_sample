```python
# 欠損があるレコードを削除
df = df.dropna(subset=necessary)
```

```python
# DataFrameの各カラムの情報をまとめて取得する

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
    'max': '最大値'
}
var_info_df = var_info_df.rename(columns=enja_dict)
```

```python
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
```

```python
# グラフの描画
names = ['太郎', '次郎', '三郎', '四郎', '五郎', '六郎']
n = len(names)
np.random.seed(seed=57)
df = pd.DataFrame({'name': names})
df['パン'] = np.random.randint(0, 300, n)
df['お米'] = np.random.randint(0, 300, n)
ratio_graph(df['name'], df['パン'], df['お米'], xlabel='名前', ylabel='食事回数', y1label='お米', y2label='パン', ratio_label='パンの割合')
```
