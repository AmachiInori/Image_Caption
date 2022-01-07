import numpy as np
from numpy import array
import os
from PIL import Image
import pickle
import string
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from keras.preprocessing import sequence
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, concatenate, Flatten, Merge
import encoder
import pandas as pd
import glob


captionFile = 'Flickr8k_text/Flickr8k.token.txt'
captions = open(captionFile, 'r').read().strip().split('\n')

imagesPath = 'Flickr8k_Dataset/Flicker8k_Dataset/'
images = glob.glob(imagesPath+'*.jpg')


def split_data(dict):
    temp = []
    for i in images:
        if i[len(imagesPath):] in dict:
            temp.append(i)
    return temp


trainImagesIndexFile = 'Flickr8k_text/Flickr_8k.trainImages.txt'
trainImagesIndex = set(open(trainImagesIndexFile, 'r').read().strip().split('\n'))
trainImages = split_data(trainImagesIndex)

validationImagesIndexFile = 'Flickr8k_text/Flickr_8k.devImages.txt'
validationImagesIndex = set(open(validationImagesIndexFile, 'r').read().strip().split('\n'))
validationImages = split_data(validationImagesIndex)

testImagesIndexFile = 'Flickr8k_text/Flickr_8k.testImages.txt'
testImagesIndex = set(open(testImagesIndexFile, 'r').read().strip().split('\n'))
testImages = split_data(testImagesIndex)


encoding_train = pickle.load(open('encoded_tra.p', 'rb'))
encoding_test = pickle.load(open('encoded_tes.p', 'rb'))

dictionary = {}
for i, caption in enumerate(captions):
    caption = caption.split('\t')
    caption[0] = caption[0][:len(caption[0])-2]
    if caption[0] in dictionary:
        dictionary[caption[0]].append(caption[1])
    else:
        dictionary[caption[0]] = [caption[1]]


trainDictionary = {}
for i in trainImages:
    if i[len(imagesPath):] in dictionary:
        trainDictionary[i] = dictionary[i[len(imagesPath):]]

validationDictionary = {}
for i in validationImages:
    if i[len(imagesPath):] in dictionary:
        validationDictionary[i] = dictionary[i[len(imagesPath):]]

testDictionary = {}
for i in testImages:
    if i[len(imagesPath):] in dictionary:
        testDictionary[i] = dictionary[i[len(imagesPath):]]


Sent = []
for key, val in trainDictionary.items():
    for i in val:
        Sent.append('<start> ' + i + ' <end>')
words = [i.split() for i in Sent]
unique = []
for i in words:
    unique.extend(i)
unique = list(set(unique))

word2idx = {val: index for index, val in enumerate(unique)}
idx2word = {index: val for index, val in enumerate(unique)}

maxSenLen = 0
for s in Sent:
    s = s.split()
    if len(s) > maxSenLen:
        maxSenLen = len(s)


f = open('training_captions.txt', 'w')

f.write("image_id\tcaptions\n")
for key, val in trainDictionary.items():
    for i in val:
        f.write(key[len(imagesPath):] + "\t" + "<start> " + i +" <end>" + "\n")

f.close()

df = pd.read_csv('training_captions.txt', delimiter='\t')
c = [i for i in df['captions']]
imgs = [i for i in df['image_id']]

vocab_size = len(unique)
samples_per_epoch = 0
for s in Sent:
    samples_per_epoch += len(s.split())-1


def data_generator(batch_size=32):
    partial_caps = []
    next_words = []
    images = []

    df = pd.read_csv('training_captions.txt', delimiter='\t')
    df = df.sample(frac=1)
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
            current_image = encoding_train[imgs[j]]
            for i in range(len(text.split()) - 1):
                count += 1

                partial = [word2idx[txt] for txt in text.split()[:i + 1]]
                partial_caps.append(partial)

                # Initializing with zeros to create a one-hot encoding matrix
                # This is what we have to predict
                # Hence initializing it with vocab_size length
                n = np.zeros(vocab_size)
                # Setting the next word to 1 in the one-hot encoded matrix
                n[word2idx[text.split()[i + 1]]] = 1
                next_words.append(n)

                images.append(current_image)

                if count >= batch_size:
                    next_words = np.asarray(next_words)
                    images = np.asarray(images)
                    partial_caps = sequence.pad_sequences(partial_caps, maxlen=maxSenLen, padding='post')
                    yield [[images, partial_caps], next_words]
                    partial_caps = []
                    next_words = []
                    images = []
                    count = 0


embedding_size = 300

image_model = Sequential([
        Dense(embedding_size, input_shape=(2048,), activation='relu'),
        RepeatVector(maxSenLen)
    ])

caption_model = Sequential([
        Embedding(vocab_size, embedding_size, input_length=maxSenLen),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(300))
    ])

final_model = Sequential([
        Merge([image_model, caption_model], mode='concat', concat_axis=1),
        Bidirectional(LSTM(256, return_sequences=False)),
        Dense(int(vocab_size)),
        Activation('softmax')
    ])

final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

def train(epochs, times):
    final_model.summary()
    with open("wti_onehot.p", "wb") as encodedPickle:
        pickle.dump(word2idx, encodedPickle)
    with open("itw_onehot.p", "wb") as encodedPickle:
        pickle.dump(idx2word, encodedPickle)
    final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=int(samples_per_epoch/times), nb_epoch=epochs,
                          verbose=1)
    final_model.save_weights('LSTM_loss.h5')


def continue_train(epochs, times):
    global word2idx
    word2idx = pickle.load(open('wti_onehot.p', 'rb'))
    global idx2word
    idx2word = pickle.load(open('itw_onehot.p', 'rb'))
    final_model.load_weights('LSTM_loss.h5')
    final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=int(samples_per_epoch/times), nb_epoch=epochs,
                              verbose=1)
    final_model.save_weights('LSTM_loss.h5')


def predict_captions(image):
    word2idx = pickle.load(open('wti_onehot.p', 'rb'))
    idx2word = pickle.load(open('itw_onehot.p', 'rb'))
    final_model.load_weights('LSTM_loss.h5')
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=maxSenLen, padding='post')
        e = encoder.encode(image)
        preds = final_model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)

        if word_pred == "<end>" or len(start_word) > maxSenLen:
            break

    return ' '.join(start_word[1:-1])


continue_train(1,2)