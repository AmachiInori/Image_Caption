import glob

imagesPath = 'Flickr8k_Dataset/Flicker8k_Dataset/'
images = glob.glob(imagesPath+'*.jpg')
captionFile = 'Flickr8k_text/Flickr8k.token.txt'
captions = open(captionFile, 'r').read().strip().split('\n')


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

maxSenLen = 0
for s in Sent:
    s = s.split()
    if len(s) > maxSenLen:
        maxSenLen = len(s)

vocabularyAmount = len(unique)