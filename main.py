import math
import time

import tensorflow.keras
import numpy as np
from cv2 import cv2

lineWidth = 30


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

''' Create the array of the right shape to feed into the keras model
The 'length' or number of images you can put into the array is
determined by the first position in the shape tuple, in this case 1. '''
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def predictPicture(image):

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #turn the image into a numpy array
    image_array = np.asarray(image)


    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return(prediction[0])



image = cv2.imread('skymap.png')

h, w, _ = image.shape

ycount = math.floor(h / lineWidth)
xcount = math.floor(w / lineWidth)

total = ycount * xcount

for i in range(ycount):

    for j in range(xcount):

        print(total - (i+1)*(j+1))
        start = time.time()

        y1 = i * lineWidth
        y2 = (i + 1) * lineWidth
        x1 = j * lineWidth
        x2 = (j + 1) * lineWidth
        crop_img = image[y1:y2, x1:x2]
        crop_img = cv2.resize(crop_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        prediction = predictPicture(crop_img)

        if prediction[0] * 100 > 80:
            return_image = cv2.putText(image, 'c', org=(round((y1 + y2) / 2), round((x1 + x2) / 2)), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        if prediction[1] * 100 > 80:
            return_image = cv2.putText(image, 's', org=(round((y1 + y2) / 2), round((x1 + x2) / 2)), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        print("time :", time.time() - start)
cv2.imshow('aa', return_image)
cv2.waitKey(0)