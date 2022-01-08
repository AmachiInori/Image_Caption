from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
import dataset_process
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation,  Merge

incv3 = InceptionV3(weights='imagenet')
myInput = incv3.input
hiddenLayer = incv3.layers[-2].output
encoder = Model(myInput, hiddenLayer)


embedding_size = 300

imageModel = Sequential([
        Dense(embedding_size, input_shape=(2048,), activation='relu'),
        RepeatVector(dataset_process.maxSenLen)
    ])

captionModel = Sequential([
        Embedding(dataset_process.vocabularyAmount, embedding_size, input_length=dataset_process.maxSenLen),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(300))
    ])

languageModel = Sequential([
        Merge([imageModel, captionModel], mode='concat', concat_axis=1),
        Bidirectional(LSTM(256, return_sequences=False)),
        Dense(int(dataset_process.vocabularyAmount)),
        Activation('softmax')
    ])

languageModel.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])



