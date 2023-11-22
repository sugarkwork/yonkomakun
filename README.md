# 四コマくん

それっぽい四コマ漫画を AI で全自動生成するためのコードです。


# 使い方

.env ファイルを作成します。
.env ファイルの中に OpenAI の API キーを記載します。

    API_KEY="sk-dummykeydummykeydummykeydummykeydummykey"


uvicorn コマンドを実行します。ポート番号は適宜好みの番号にしてください。

    uvicorn main:app --reload --port 7861


とりあえず生成ボタンを押すと、生成される……はず。


# 組み込み済みフォントのライセンス表記

本プログラムには以下のフォントを組み込んでいます。
ライセンスの詳細は fonts 配下の フォント名ディレクトリの下にあるライセンスファイルを参照してください。


* 源暎フォント

https://okoneya.jp/font/genei-antique.html

    Copyright (c) 2017-2018, おたもん (http://okoneya.jp/font/), with Reserved Font Name '源暎'.

        Copyright (c) 2014, 2015, Adobe Systems Incorporated (http://www.adobe.com/), with Reserved Font Name 'Source'.
        Copyright (c) 2014-2015, 自家製フォント工房.
        Copyright (c) FONT 910.
        Copyright (c) 2003-2012, Philipp H. Poll.
        Copyright (c) 2008-2014, Classical Letter Project.
        Copyright (c) 2013, Google Inc.
        Copyright (c) Font Awesome.


    This Font Software is licensed under the SIL Open Font License,
    Version 1.1.

    This license is copied below, and is also available with a FAQ at:
    http://scripts.sil.org/OFL


