import cv2 as cv

def get_image(input_path,filter=None):
    img = cv.imread(input_path)
    if filter!=None:
        cv.cvtColor(img,filter)
    return img

def data_prep(image):
    modified_image = cv.resize(image,(600,400),interpolation = cv.INTER_AREA)
    modified_image = modified_image.reshape((modified_image.shape[0]*modified_image.shape[1],3))
    return modified_image

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))