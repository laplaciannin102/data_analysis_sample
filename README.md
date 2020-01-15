# data_analysis_sample
sample of data analysis

## Contents menu

### python
- simple_python_analysis_001
  - simple is the best.

- simple_python_analysis_002
  - simple_python_analysis_001に自作関数を追加するなど。おすすめ。

- simple_python_analysis_all
  - simple_python_analysisを全て格納。

### R
- no contents

### Others
- start_jupyter.bat

## directory structure
- **input**
  - 入力データ格納ディレクトリ。
- **output**
  - 出力データ格納ディレクトリ。
- **intermediate**
  - 中間データ格納ディレクトリ。
- **docs**
  - ドキュメント格納ディレクトリ。
- **src**
  - 分析用program格納ディレクトリ。Source Code。
  - ここにsample programを置いています。

## start_jupyter.bat
- 好きなディレクトリでJupyter Notebookを起動するbatファイルです
- Remarks:
    - [[Qiita]WindowsのDドライブでJupyter Notebookを起動するバッチファイル](https://qiita.com/AnnnPsinan414/items/7764723ed5183ea4b3e4)


```bat
cd /d %~dp0

jupyter notebook
```
