# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:40:44 2019

@author: Yuta
"""

from keras import models
from keras.applications import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

import matplotlib.pyplot as plt
import numpy as np

#保存しておいたモデル構造をjsonファイルから読み込み
json_str = open('save/mymodel_struct.json','r').read()
mymodel = models.model_from_json(json_str)
#保存しておいたモデルの重みをh5ファイルから読み込み
mymodel.load_weights('save/mymodel_weights.h5')

#vgg16ネットワークを読み込み
vgg_conv = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(224,224,3))

#予測したい画像のファイル名
filename = 'images/watermelon.jpg'
original = load_img(filename,target_size=(224,224))
plt.imshow(original)
plt.axis('off')
plt.show()
img = img_to_array(original)
img = np.expand_dims(img,axis=0)
input_tensor = preprocess_input(img.copy())

#vgg16に画像を入力して特徴量を取得する
features = vgg_conv.predict(input_tensor)

input_features = np.reshape(features,(1,7*7*512))
#自作モデルに特徴量を入力して予測
prediction = mymodel.predict(input_features)

#テキストファイルからインデックスとクラスラベルを読み込む
label_str = open('save/idx2label.txt','r').readlines()

idx2label = {}
#インデックスからラベルを取得できる辞書を作成
for v_k in label_str:
    v,k = v_k.split(' ')
    idx2label[int(v)] = k

answer = np.argmax(prediction)
#結果を表示
print('予測確率:%.3f'%prediction[0][answer])
print(idx2label[answer])