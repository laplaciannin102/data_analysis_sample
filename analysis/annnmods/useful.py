#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: kosuke.asada
便利関数
"""


import sys, os
import gc
import configparser
from glob import glob
from pathlib import Path
from googletrans import Translator


# 設定のimport
from .mod_config import *
# 自作関数のimport
from .analysis import *
from .calculation import *
from .scraping import *
from .visualization import *


def print_usage(modules_dir=mod_dir, extension='py'):
    """
    関数の使用方法を表示する。

    Args:
        dir_path: str, optional(default='./annnmods/')
            自作関数ファイルを置いたディレクトリのパス。
        
        extension: str, optional(default='py')
            自作関数ファイルの拡張子。デフォルトではpy。
    """
    usage_path = modules_dir + 'USAGE.md'
    with open(usage_path, mode='r', encoding='utf-8') as f:
        text = f.read()
    print(text)
    print()
    # 読み込み済ライブラリ表示
    print_imports(modules_dir=modules_dir, extension=extension)
    print()
    print()
    # 自作関数一覧表示
    print_funcs(modules_dir=modules_dir, extension=extension)
    print()
    gc.collect()


def print_funcs(modules_dir=mod_dir, extension='py'):
    """
    全ての自作関数名を表示する。

    Args:
        modules_dir: str, optional(default='./annnmods/')
            自作関数ファイルを置いたディレクトリのパス。
        
        extension: str, optional(default='py')
            自作関数ファイルの拡張子。デフォルトではpy。

    Returns:
        all_funcs: list of str
            全ての自作関数名のリスト。
    """
    print('■■■■■■■■■■■■■■■■■■■■')
    print('■■■■■■■自作関数一覧■■■■■■■')
    print('■■■■■■■■■■■■■■■■■■■■')
    module_paths = list_segments(dir_path=modules_dir, extension=extension)
    all_funcs = []

    for path in module_paths:
        with open(path, mode='r', encoding='utf-8') as f:
            rl = f.readlines()
            funcs = [str(ii).replace(':\n', '') for ii in rl if ('def ' in ii) and not('# not_read') in ii] # not_read
            for ii in range(10):
                funcs = [str(ii).replace(' def', 'def') for ii in funcs]
            funcs = [str(ii).replace('def ', '') for ii in funcs if not ('__init__(' in ii)]
            all_funcs += funcs
    
    # 重複の削除
    all_funcs = sorted(set(all_funcs), key=all_funcs.index)
    # 昇順に並び替える
    all_funcs.sort()
    # 標準出力
    [print(ii) for ii in all_funcs]
    print('■■■■■■■■■■■■■■■■■■■■')
    return all_funcs


def print_imports(modules_dir=mod_dir, extension='py'):
    """
    全てのimportした外部モジュール名を表示する。

    Args:
        modules_dir: str, optional(default='./annnmods/')
            自作関数ファイルを置いたディレクトリのパス。
        
        extension: str, optional(default='py')
            自作関数ファイルの拡張子。デフォルトではpy。

    Returns:
        all_imports: list of str
            全ての外部モジュール名のリスト。
    """
    print('■■■■■■■■■■■■■■■■■■■■')
    print('■■■読み込み済みライブラリ一覧■■■■')
    print('■■■■■■■■■■■■■■■■■■■■')
    module_paths = list_segments(dir_path=modules_dir, extension=extension)
    all_imports = []

    for path in module_paths:
        with open(path, mode='r', encoding='utf-8') as f:
            rl = f.readlines()
            ext_mods = [str(ii).replace('\n', '') for ii in rl if ('import ' in ii) and not('# not_read') in ii and not('print_imports(') in ii] # not_read
            all_imports += ext_mods
    
    # 重複の削除
    all_imports = sorted(set(all_imports), key=all_imports.index)
    # 昇順に並び替える
    all_imports.sort()
    # 標準出力
    [print(ii) for ii in all_imports]
    print('■■■■■■■■■■■■■■■■■■■■')
    return all_imports


def list_segments(dir_path='./', rescursive=False, extension=None):
    """
    dir_path以下にあるファイルの相対パスのリストを返す

    Args:
        dir_path: str, optional(default='./')
            検索したいディレクトリのパス。

        rescursive: bool, optional(default=False)
            再帰的に検索するかどうか。

        extension: str, list of str or tuple of str, optional(default=None)
            拡張子。'csv'とか['csv', 'xlsx']とかみたいな形で指定。

    Returns:
        path_list: list of str
            ファイルの相対パスのリスト。
    """
    # ディレクトリ
    dir_p = Path(dir_path)

    # 再帰的に検索するかどうか
    resc_path = './*'
    if rescursive:
        resc_path = '**/*'

    # 拡張子
    if extension is None:
        ext_list = ['']
    elif (type(extension) == tuple) or (type(extension) == list):
        extension = list(extension)
        ext_list = ['.' + str(ii) for ii in extension]
    else:
        ext_list = ['.' + str(extension)]

    # それぞれの拡張子について検索
    path_list = []
    for ext in ext_list:
        path_list += list(dir_p.glob(resc_path + ext))

    # strに直す
    path_list = [str(ii) for ii in path_list]
    # 重複の削除
    path_list = sorted(set(path_list), key=path_list.index) # 3.6以降ではl=list(dict.fromkeys(l))でも

    return path_list


def translate_to_ja(text):
    """
    Google翻訳で日本語に翻訳する。

    Args:
        text: str
            翻訳したいテキスト。

    Returns:
        ja_text: str
            日本語訳されたテキスト。
    """
    text = str(text)
    translator = Translator()
    # srcはデフォルトでautoになっており自動で元の言語を判定してくれる
    ja_text = translator.translate(text, dest='ja').text
    ja_text = str(ja_text)
    return ja_text


def text_to_empty(text, delete=[]):
    """
    text内の特定の文字列を全て空文字にする。

    Args:
        text: str
            空文字にしたい文字列を含むテキスト。
        
        delete: list of str, optional(default=[])
            空文字にしたい文字列のリスト。
    """
    text = str(text)
    delete = list(delete)

    for ii in delete:
        text = text.replace(str(ii), '')
    
    return text


def pro_makedirs(dir_path):
    """
    ディレクトリを作成する。指定のディレクトリが存在しない場合のみ作成する。

    Arg:
        path: str
            ディレクトリのパス。
    """
    dir_path = str(dir_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    else:
        pass

