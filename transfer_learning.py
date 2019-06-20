# -*- coding: utf-8 -*-
"""
Created on Thu May 30 07:08:31 2019

@author: Yuta
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

from keras import models
from keras import layers
from keras import optimizers
from keras.callbacks import EarlyStopping

import os

#VGG16モデルを全結合層を除いて読み込む
vgg_conv = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(224,224,3))

#モデルの構成を表示
vgg_conv.summary()

train_dir = './clean-dataset/train'
validation_dir = './clean-dataset/validation'

#学習データの合計枚数と評価データの合計枚数
nTrain = 600
nVal = 150

#preprocess_inputを用いて前処理
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

batch_size = 20
#学習データ分の配列を宣言&ゼロで初期化
train_features = np.zeros(shape=(nTrain,7,7,512))
#学習データのラベル分の配列を宣言&ゼロで初期化
train_labels = np.zeros(shape=(nTrain,3))

train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

i = 0
for inputs_batch,labels_batch in train_generator:
    #モデルに画像を入力してbatchサイズ分の特徴量を取得
    features_batch = vgg_conv.predict(inputs_batch)
    #予め宣言しておいたtrain_featuresに特徴量を代入
    train_features[i * batch_size:(i + 1) * batch_size] = features_batch
    train_labels[i * batch_size:(i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTrain:
        break

#7×7×512のテンソルを7*7*512のベクトルに変換
train_features = np.reshape(train_features,(nTrain,7*7*512))

validation_features = np.zeros(shape=(nVal,7,7,512))
validation_labels = np.zeros(shape=(nVal,3))

validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

i = 0
for inputs_batch,labels_batch in validation_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    validation_features[i * batch_size:(i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size:(i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break

validation_features = np.reshape(validation_features,(nVal,7*7*512))


#自作モデルの構築
model = models.Sequential()
model.add(layers.Dense(256,activation='relu',input_dim=7*7*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3,activation='softmax'))

#最適化アルゴリズム、損失関数の設定
model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

#自作モデルを学習
history = model.fit(train_features,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(validation_features,validation_labels))

os.makedirs('save',exist_ok=True)
#モデルを保存
model_json_str = model.to_json()
open('save/mymodel_struct.json','w').write(model_json_str)
model.save_weights('save/mymodel_weights.h5')

#損失曲線
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training loss','Validation loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss.png')
plt.show()

#精度曲線
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Training accuracy','Validation accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('acc.png')
plt.show()

#fnamesは画像のファイル名
fnames = validation_generator.filenames
#ground_truthは正解ラベル
ground_truth = validation_generator.classes

label2index = validation_generator.class_indices
#indexからlabel対応の辞書を作成
idx2label = dict((v,k) for k,v in label2index.items())
#辞書をテキストファイルに保存
string = ''
for v,k in idx2label.items():
    string += str(v)+' '+k+'\n'
open('save/idx2label.txt','w').write(string)

#自作モデルを使って予測
predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)

#各クラス5個分の予測結果を出力
i = 0
for prediction in predictions[::10]:
    original = load_img('{}/{}'.format(validation_dir,fnames[i]))
    plt.imshow(original)
    plt.show()
    print('Original label:{},Prediction:{},confidence:{:.3f}'.format(
            idx2label[ground_truth[i]],
            idx2label[prediction],
            prob[i][prediction]))
    i += 10

#予測結果と正解ラベルが異なっているものをerrorsとする
errors = np.where(predictions != ground_truth)[0]
print('No of errors = {}/{}'.format(len(errors),nVal))

#誤認識したデータをすべて表示
for i in range(len(errors)):
    pred_class = np.argmax(prob[errors[i]])
    pred_label = idx2label[pred_class]
    #元画像を読み込んで表示
    original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
    plt.imshow(original)
    plt.show()
    #予測結果を出力
    print('Original label:{},Prediction:{},confidence:{:.3f}'.format(
            fnames[errors[i]].split('\\')[0],
            pred_label,
            prob[errors[i]][pred_class]))
