import pickle
import predict
import os

ctpdictionary = {}
ptcdictionary = {}

try:
    ctpdictionary, ptcdictionary = pickle.load(open('ipdict.p', 'rb'))
except FileNotFoundError:
    with open('ipdict.p', 'wb') as dictDump:
        pickle.dump((ctpdictionary, ptcdictionary), dictDump)


def add_images(listOfImages):
    for image in listOfImages:
        if not ptcdictionary.get(image, -1) == -1:
            continue
        caption = predict.get_caption(image)
        ptcdictionary[image] = caption
        words = caption.split()
        for word in words:
            if ctpdictionary.get(word, -1) == -1:
                ctpdictionary[word] = []
            ctpdictionary[word].append(image)

    with open('ipdict.p', 'wb') as dictDump:
        pickle.dump((ctpdictionary, ptcdictionary), dictDump)


def delete_images(listOfImages):
    for image in listOfImages:
        if ptcdictionary.get(image, -1) == -1:
            continue
        caption = predict.get_caption(image)
        ptcdictionary.pop(image)
        words = caption.split()
        for word in words:
            ctpdictionary[word].remove(image)

    with open('ipdict.p', 'wb') as dictDump:
        pickle.dump((ctpdictionary, ptcdictionary), dictDump)


def get_list(keyWordsList):
    res = {}
    if len(keyWordsList) == 0:
        for image in ptcdictionary:
            res[image] = ptcdictionary[image]
    elif len(keyWordsList) == 1:
        for image in ctpdictionary[keyWordsList[0]]:
            res[image] = ptcdictionary[image]
    else:
        for image in ctpdictionary[keyWordsList[0]]:
            res[image] = ptcdictionary[image]
        for keyword in keyWordsList:
            if ctpdictionary.get(keyword, -1) == -1:
                return {}
            badimage = []
            for image, str in res.items():
                if not image in ctpdictionary[keyword]:
                    badimage.append(image)
            for bima in badimage:
                res.pop(bima)
    return res


def open_image(image):
    os.startfile(image)

