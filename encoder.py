import glob
import numpy as np
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from tqdm import tqdm
import pickle

imagesPath = 'Flickr8k_Dataset/Flicker8k_Dataset/'
images = glob.glob(imagesPath+'*.jpg')

# 处理数据集


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def split_data(dict):
    temp = []
    for i in images:
        if i[len(imagesPath):] in dict:
            temp.append(i)
    return temp


def preprocess_one(imageName):
    img = image.load_img(imageName, target_size=(299, 299))
    res = image.img_to_array(img)
    res = np.expand_dims(res, axis=0)
    res = preprocess_input(res)
    return res


trainImagesIndexFile = 'Flickr8k_text/Flickr_8k.trainImages.txt'
trainImagesIndex = set(open(trainImagesIndexFile, 'r').read().strip().split('\n'))
trainImages = split_data(trainImagesIndex)

validationImagesIndexFile = 'Flickr8k_text/Flickr_8k.devImages.txt'
validationImagesIndex = set(open(validationImagesIndexFile, 'r').read().strip().split('\n'))
validationImages = split_data(validationImagesIndex)

testImagesIndexFile = 'Flickr8k_text/Flickr_8k.testImages.txt'
testImagesIndex = set(open(testImagesIndexFile, 'r').read().strip().split('\n'))
testImages = split_data(testImagesIndex)

# 编码器构造
model = InceptionV3(weights='imagenet')

myInput = model.input
hidden_layer = model.layers[-2].output
myModel = Model(myInput, hidden_layer)


def encode(image):
    image = preprocess_one(image)
    res = myModel.predict(image)
    res = np.reshape(res, res.shape[1])
    return res


def encode_total():
    encodingTrain = {}
    for img in tqdm(trainImages):
        encodingTrain[img[len(imagesPath):]] = encode(img)

    with open("encoded_tra.p", "wb") as encodedPickle:
        pickle.dump(encodingTrain, encodedPickle)

    encodingTest = {}
    for img in tqdm(testImages):
        encodingTest[img[len(imagesPath):]] = encode(img)

    with open("encoded_tes.p", "wb") as encodedPickle:
        pickle.dump(encodingTest, encodedPickle)

    return encodingTrain, encodingTest
