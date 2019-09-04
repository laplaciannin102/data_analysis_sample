#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: kosuke.asada
スクレイピング用関数
"""


import sys, os
import gc
import time
import json
from urllib import parse
import requests
from bs4 import BeautifulSoup

# 設定のimport
from .mod_config import *
# 自作関数のimport
from .analysis import *
from .calculation import *
from .useful import *
from .visualization import *


"""
こちらの記事を参考にさせていただきました。
Qiita: https://qiita.com/derodero24/items/949ac666b18d567e9b61
Github: https://github.com/derodero24/Deropy/blob/master/google.py


「ドラえもん」で200件検索する場合
google = Google()
# テキスト検索
result = google.Search('ドラえもん', type='text', maximum=200)
# 画像検索
result = google.Search('ドラえもん', type='image', maximum=200)
"""
class Google:
    def __init__(self):
        self.GOOGLE_SEARCH_URL = 'https://www.google.co.jp/search'
        self.session = requests.session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0'})

    def Search(self, keyword, type='text', maximum=100):
        """
        Google検索を行う。
        
        Args:
            keyword: str
                検索ワード。
            
            type: str, optional(default='text')
                検索のタイプ。textならテキスト検索、imageなら画像検索。
            
            maximum: int, optional(default=200)
                取得する検索結果の最大数。初期は200件取得。

        Returns:
            result: list of str
                検索結果として得られたURLのリスト。
        """
        print('Google', type.capitalize(), 'Search :', keyword)
        result, total = [], 0
        query = self.query_gen(keyword, type)
        while True:
            # 検索
            html = self.session.get(next(query)).text
            links = self.get_links(html, type)

            # 検索結果の追加
            if not len(links):
                print('-> No more links')
                break
            elif len(links) > maximum - total:
                result += links[:maximum - total]
                break
            else:
                result += links
                total += len(links)

        print('-> Finally got', str(len(result)), 'links')
        return result

    def query_gen(self, keyword, type):
        '''検索クエリジェネレータ'''
        page = 0
        while True:
            if type == 'text':
                params = parse.urlencode({
                    'q': keyword,
                    'num': '100',
                    'filter': '0',
                    'start': str(page * 100)})
            elif type == 'image':
                params = parse.urlencode({
                    'q': keyword,
                    'tbm': 'isch',
                    'filter': '0',
                    'ijn': str(page)})

            yield self.GOOGLE_SEARCH_URL + '?' + params
            page += 1

    def get_links(self, html, type):
        '''リンク取得'''
        soup = BeautifulSoup(html, 'lxml')
        if type == 'text':
            elements = soup.select('.rc > .r > a')
            links = [e['href'] for e in elements]
        elif type == 'image':
            elements = soup.select('.rg_meta.notranslate')
            jsons = [json.loads(e.get_text()) for e in elements]
            links = [js['ou'] for js in jsons]
        return links


def get_soup(url):
    """
    URLからBeautifulSoupのインスタンスを作成する。requests.get使用。

    Args:
        url: str
            soupインスタンスを作成したいURL
    
    Returns:
        soup: bs4.BeautifulSoup instance
            soupインスタンス。
    """
    # htmlを取得
    html = requests.get(url)

    # soupを作成
    soup = BeautifulSoup(html.content, 'html.parser')

    return soup


def download_img(url, file_name='./img.jpg', force=False):
    """
    画像をダウンロードする。requests.get使用。

    Args:
        url: str
            ダウンロードする画像ファイルのURL。
        
        file_name: str, optional(default='./img.jpg')
            ダウンロードした後の画像ファイルの名前。パスを含む。
        
        force: bool, optional(default=False)
            force to overwrite. 強制的に上書きするかどうか。
            True: 強制的に上書きする。
            False: すでにファイルが存在していれば保存しない。
    """
    file_name = str(file_name)

    if not force:
        # すでにファイルが存在していれば保存しない
        if os.path.exists(file_name):
            print(file_name + ' is already exists.')
            return

    # 保存する
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(r.content)


def get_domain(url):
    """
    URLからドメイン名を取得する

    Args:
        url: str
            ドメイン名を取得したいURL
    
    Returns:
        domain_url: str
            ドメイン名。
    """
    url = str(url)

    try:
        # URLをパースする
        parsed_url = parse.urlparse(url)
        
        # URLスキーマ
        #print(parsed_url.scheme) # http
        
        # ネットワーク上の位置(≒ドメイン)を取得する
        #print(parsed_url.netloc) # www.python.ambitious-engineer.com
        
        # 階層パス
        #print(parsed_url.path) # /archives/
        
        # クエリ要素 
        #print(parsed_url.query) # s=hoge&x=0&y=0
        
        # フォーマットする 
        domain_url = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_url)
    except:
        print('get domain error:', url)
        domain_url = ''

    domain_url = str(domain_url)
    return domain_url


def dl_all_imgs(url, dir_path='./', extension=['jpg', 'png'], make_dir=True):
    """
    特定のページに存在する画像ファイルを全てダウンロードする。

    Args:
        url: str, list of str or tuple of str
            ページのURL。リストで複数指定すると複数ページを見に行く。
        
        dir_path: str, optional(default='./')
            ファイルの保存先となるディレクトリのパス。
        
        extension: str, list of str or tuple of str, optional(default=['jpg', 'png'])
            拡張子。'jpg'とか['jpg', 'png']とかみたいな形で指定。
        
        make_dir: bool, optional(default=True)
            保存先の下に更にURLごとのディレクトリを作成するかどうか。
            True: 作成する。
            False: 作成しない。
    """
    np.random.seed(57)
    # url
    url_list = []
    if (type(url) == tuple) or (type(url) == list):
        url_list = list(url)
    else:
        url_list = [str(url)]

    # 拡張子
    ext_list = []
    if (type(extension) == tuple) or (type(extension) == list):
        extension = list(extension)
        ext_list = ['.' + str(ii) for ii in extension]
    else:
        ext_list = ['.' + str(extension)]

    url_num = 0
    for url in url_list:
        
        img_num = 1

        # ドメイン名
        dom = get_domain(url)

        try:
            soup = get_soup(url)
            title = str(soup.title.string)   # ページタイトル
            if title=='':
                title = text_to_empty(str(url), ['/', '.', ':', '?', '*', 'http', 'www'])
                title = title[:50]
            
            img_url_list = []
            [img_url_list.extend([img.get('data-src'), img.get('src')]) for img in soup.find_all('img')]
            img_url_list = [ii for ii in img_url_list if not ii is None]

            if len(img_url_list)==0:
                continue

            tmp = []
            for ext in ext_list:
                tmp += [ii for ii in img_url_list if ext in ii]
            
            # 重複の削除
            img_url_list = sorted(set(tmp), key=img_url_list.index)
            
            # 各URLごとのディレクトリを作成する
            title = text_to_empty(title, ['/', '.', ':', '?', '*', 'http', 'www'])
            url_dir_path = dir_path + title
            pro_makedirs(url_dir_path)
            print(url_dir_path, '作成')

            for img_url in img_url_list:

                img_url = (dom + img_url).replace(dom * 2, dom)

                # 同ページ内で2枚目以降は間隔をあけてダウンロードする
                if img_num > 1:
                    t = np.random.randint(2, 5) # 1秒から5秒の間
                    # 待つ
                    time.sleep(t)

                # ファイル名
                file = url_dir_path + '/' + str(img_url).split('/')[-1]
                file = text_to_empty(file, [':', '?', '*'])

                # ダウンロードする
                download_img(img_url, file)
                img_num += 1
            
            url_num += 1
        
        except:
            print('error:', url)
            continue
