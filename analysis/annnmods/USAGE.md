# 自作関数のディレクトリ構成
以下を想定

analysis/
  ├ NoteBook.ipynb
  ├ Script.py
  │
  └ annnmods/
    ├ analysis.py       # 分析用関数
    ├ calculation.py    # 数値計算用関数
    ├ scraping.py       # スクレイピング用関数
    ├ useful.py         # 便利関数
    ├ visualization.py  # 可視化用関数
    ・
    ・
    └ __init__.py       # 初期化. 全関数をimport.

# 使用方法
## 関数のimport
from annnmods import *
または
import annnmods as am

## 関数の使用方法
print_usage()
または
am.print_usage()