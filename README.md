# CNN-Pretrained-Model
既存の学習済みモデルをKerasで使用する。

# ファイル構成
・pretrained_model.py
  -学習済みモデルそのまま利用して画像認識
  
・transfer_learning.py
  -転移学習を行い、自作モデルを保存する
  
・load_mymodel.py
  -transfer_learning.pyで作成したモデルを読み込んで利用する
  
# プログラム実行の準備
## pretrained_model.py

imagesフォルダを作成してcat.jpgをフォルダ内に入れる。

勿論猫の画像以外でも試すことができる。

ファイル名やフォルダを変える場合にはプログラム内のfilenameを書き換える。

## transfer_learning.py

clean-datasetフォルダを作成してそのフォルダ内にtrain、validationフォルダを作る。

上記フォルダ(train,validation)内にクラス名フォルダを作成してクラスごとにデータを入れる。

nTrain,nValをデータの数に書き換える。(初期値は学習データ数が600、評価データ数が150)

## load_mymodel.py

transfer_learning.pyを実行した後、saveフォルダが作成されていることを確認して実行する。

# 実行に必要なライブラリ
・numpy

・matplotlib

・pillow

・opencv

・Keras (Tensorflow backend)

・Tensorflow (各自バックエンドで利用したいライブラリをインストール)

# 実行確認済み環境
・python 3.7.3
