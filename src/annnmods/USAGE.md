# 自作moduleのディレクトリ構成
以下を想定

analysis/
  ├ NoteBook.ipynb
  ├ Script.py
  │
  └ annnmods/
    ├ __init__.py             # 初期化. 全moduleをimport.
    ├ analysis.py             # 分析用module
    ├ calculation.py          # 数値計算用module
    ├ preprocessing.py        # 前処理用module
    ├ scraping.py             # スクレイピング用module
    ├ useful.py               # 便利module
    ├ visualization.py        # 可視化用module
    ├ mod_config.py           # 設定ファイル
    ・
    ・
    └ AnalysisCheatSheet.md   # 分析用チートシート

# 使用方法
## moduleのimport
from annnmods import *
または
import annnmods as am

## moduleの使用方法を確認
print_usage()
または
am.print_usage()

[AnalysisCheatSheet.md](./AnalysisCheatSheet.md)も参照

## 各関数(functionとする)のdocstringは以下で確認できる
print(function.__doc__)