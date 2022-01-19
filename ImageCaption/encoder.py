import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
import pickle
import dataset_process
import my_model

# 处理数据集


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def split_data(dict):
    temp = []
    for i in dataset_process.images:
        if i[len(dataset_process.imagesPath):] in dict:
            temp.append(i)
    return temp


def preprocess_one(imageName):
    img = image.load_img(imageName, target_size=(299, 299))
    res = image.img_to_array(img)
    res = np.expand_dims(res, axis=0)
    res = preprocess_input(res)
    return res


def encode(image):
    image = preprocess_one(image)
    res = my_model.encoder.predict(image)
    res = np.reshape(res, res.shape[1])
    return res


def encode_total():
    encodingTrain = {}
    for img in tqdm(dataset_process.trainImages):
        encodingTrain[img[len(dataset_process.imagesPath):]] = encode(img)

    with open("saved_model/encoded_tra.p", "wb") as encodedPickle:
        pickle.dump(encodingTrain, encodedPickle)

    encodingTest = {}
    for img in tqdm(dataset_process.testImages):
        encodingTest[img[len(dataset_process.imagesPath):]] = encode(img)

    with open("saved_model/ncoded_tes.p", "wb") as encodedPickle:
        pickle.dump(encodingTest, encodedPickle)

    return encodingTrain, encodingTest