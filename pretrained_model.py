# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:45:53 2019

@author: Yuta
"""

import numpy as np
import matplotlib.pyplot as plt

from keras.applications import vgg16,inception_v3,resnet50,mobilenet
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.imagenet_utils import decode_predictions

import cv2

#各種モデルの読み込み
vgg_model = vgg16.VGG16(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
resnet_model = resnet50.ResNet50(weights='imagenet')
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

filename = 'images/cat.jpg'
#画像を学習サイズ(224,224)で読み込み
original = load_img(filename,target_size=(224,224))
plt.imshow(original)
plt.axis('off')
plt.show()
print('PIL image size',original.size)
#PIL形式をnumpy配列に変換
numpy_image = img_to_array(original)
print('numpy array size',numpy_image.shape)
#次元拡張関数で(1,224,224,3)の形に変換
img_tensor = np.expand_dims(numpy_image,axis=0)
print('image batch size',img_tensor.shape)

#VGG16を使用
#画像の前処理
processed_image = vgg16.preprocess_input(img_tensor.copy())
#画像を入力して予測
predictions = vgg_model.predict(processed_image)
#予測結果上位3つのクラスラベルを取得
label_vgg = decode_predictions(predictions,top=3)
print(label_vgg)

#ResNet50を使用
processed_image = resnet50.preprocess_input(img_tensor.copy())
predictions = resnet_model.predict(processed_image)
label_resnet = decode_predictions(predictions,top=3)
print(label_resnet)

#MobileNetを使用
processed_image = mobilenet.preprocess_input(img_tensor.copy())
predictions = mobilenet_model.predict(processed_image)
label_mobilenet = decode_predictions(predictions,top=3)
print(label_mobilenet)

#Inception_V3を使用
#Inception_V3の学習画像サイズ(299,299)で画像を読み込む
original = load_img(filename,target_size=(299,299))
numpy_image = img_to_array(original)
img_tensor = np.expand_dims(numpy_image,axis=0)
processed_image = inception_v3.preprocess_input(img_tensor.copy())
predictions = inception_model.predict(processed_image)
label_inception = decode_predictions(predictions,top=3)
print(label_inception)

numpy_image = np.uint8(img_to_array(original)).copy()
numpy_image = cv2.resize(numpy_image,(900,900))

cv2.putText(numpy_image, "VGG16: {}, {:.2f}".format(label_vgg[0][0][1], label_vgg[0][0][2]), (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
cv2.putText(numpy_image, "MobileNet: {}, {:.2f}".format(label_mobilenet[0][0][1], label_mobilenet[0][0][2]), (350, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
cv2.putText(numpy_image, "Inception: {}, {:.2f}".format(label_inception[0][0][1],  label_inception[0][0][2]), (350, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
cv2.putText(numpy_image, "ResNet50: {}, {:.2f}".format(label_resnet[0][0][1], label_resnet[0][0][2]), (350, 145), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
#結果を画像として保存する
cv2.imwrite("images/{}_output.jpg".format(filename.split('/')[-1].split('.')[0]),  cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR))

#結果を表示
plt.imshow(numpy_image)
plt.axis('off')
plt.show()
