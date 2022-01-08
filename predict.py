import decoder

def get_caption(image):
    return decoder.predict_captions(image)