import os
import numpy as np
import pickle
from keras.preprocessing import sequence
import my_model
import pandas as pd
from keras.preprocessing import image

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

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

def predict_captions(image):
    word2idx = pickle.load(open('saved_model/wti_onehot.p', 'rb'))
    idx2word = pickle.load(open('saved_model/itw_onehot.p', 'rb'))
    my_model.languageModel.load_weights('saved_model/LSTM_loss.h5')
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=40, padding='post')
        e = encode(image)
        preds = my_model.languageModel.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)

        if word_pred == "<end>" or len(start_word) > 40:
            break

    return ' '.join(start_word[1:-1])

def get_caption(image):
    print("-----Python: getting caption of "+ image + "-----")
    if os.path.exists(image):
        return predict_captions(image)
    else:
        print("-----Python: Error. File is not exist: " + image + "-----")
        return "<Error>"