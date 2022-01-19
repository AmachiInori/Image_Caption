import numpy as np
import pickle
from keras.preprocessing import sequence
import encoder
import dataset_process
import my_model
import pandas as pd

encodingTrain = pickle.load(open('saved_model/encoded_tra.p', 'rb'))
encodingTest = pickle.load(open('saved_model/encoded_tes.p', 'rb'))

word2idx = {val: index for index, val in enumerate(dataset_process.unique)}
idx2word = {index: val for index, val in enumerate(dataset_process.unique)}


samples_per_epoch = 383454

def data_generator(batch_size=32):
    partial_caps = []
    next_words = []
    images = []

    df = pd.read_csv('saved_model/training_captions.txt', delimiter='\t')
    df = df.sample(frac=1) # 乱序
    iter = df.iterrows()
    c = []
    imgs = []
    for i in range(df.shape[0]):
        x = next(iter)
        c.append(x[1][1])
        imgs.append(x[1][0])

    count = 0
    while True:
        for j, text in enumerate(c):
            current_image = encodingTrain[imgs[j]]
            for i in range(len(text.split()) - 1):
                count += 1

                partial = [word2idx[txt] for txt in text.split()[:i + 1]]
                partial_caps.append(partial)
                n = np.zeros(dataset_process.vocabularyAmount)
                n[word2idx[text.split()[i + 1]]] = 1
                next_words.append(n)

                images.append(current_image)

                if count >= batch_size:
                    next_words = np.asarray(next_words)
                    images = np.asarray(images)
                    partial_caps = sequence.pad_sequences(partial_caps, maxlen=40, padding='post')
                    yield [[images, partial_caps], next_words]
                    partial_caps = []
                    next_words = []
                    images = []
                    count = 0


def train(epochs, times):
    my_model.languageModel.summary()
    my_model.languageModel.fit_generator(data_generator(batch_size=128), samples_per_epoch=int(samples_per_epoch / times), nb_epoch=epochs,
                                         verbose=1)
    with open("saved_model/wti_onehot.p", "wb") as encodedPickle:
        pickle.dump(word2idx, encodedPickle)
    with open("saved_model/itw_onehot.p", "wb") as encodedPickle:
        pickle.dump(idx2word, encodedPickle)
    my_model.languageModel.save_weights('saved_model/LSTM_loss.h5')


def continue_train(epochs, times):
    global word2idx
    word2idx = pickle.load(open('saved_model/wti_onehot.p', 'rb'))
    global idx2word
    idx2word = pickle.load(open('saved_model/itw_onehot.p', 'rb'))
    my_model.languageModel.load_weights('saved_model/LSTM_loss.h5')
    my_model.languageModel.fit_generator(data_generator(batch_size=128), samples_per_epoch=int(samples_per_epoch / times), nb_epoch=epochs,
                                         verbose=1)
    my_model.languageModel.save_weights('saved_model/LSTM_loss.h5')


def predict_captions(image):
    word2idx = pickle.load(open('saved_model/wti_onehot.p', 'rb'))
    idx2word = pickle.load(open('saved_model/itw_onehot.p', 'rb'))
    my_model.languageModel.load_weights('saved_model/LSTM_loss.h5')
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=40, padding='post')
        e = encoder.encode(image)
        preds = my_model.languageModel.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)

        if word_pred == "<end>" or len(start_word) > 40:
            break

    return ' '.join(start_word[1:-1])