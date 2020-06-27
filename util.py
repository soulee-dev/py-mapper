#Mapping tool to teach CNN model
import uuid
import numpy as np
import cv2
import math
import json

isTraninMode = True
isCrossWalk = True

windowName = 'skymap'
lineWidth = 30
lineColor = (255, 255 ,255)
boxColor = (100, 100, 100)

def drawLines(image, width):
    h, w, _ = image.shape

    #TODO Rectangle로 대체 가능?
    #TODO round말고 버림해야할수도
    for i in range(round(w / width)):
        x, y = width * i, width * i
        image = cv2.line(image, (x, 0), (x, h), lineColor)
        image = cv2.line(image, (0, y), (w, y), lineColor)

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xround = math.ceil(x / lineWidth)
        yround = math.ceil(y / lineWidth)

        y1 = (yround - 1) * lineWidth
        y2 = yround * lineWidth
        x1 = (xround - 1) * lineWidth
        x2 = xround * lineWidth

        crop_img = param[y1:y2, x1:x2]
        crop_img = cv2.resize(crop_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        if isTraninMode:
            if isCrossWalk:
                filename = 'datasets/train/crosswalk/' + str(uuid.uuid4()) + '.png'
            else:
                filename = 'datasets/train/sidewalk/' + str(uuid.uuid4()) + '.png'

        if not isTraninMode:
            if isCrossWalk:
                filename = 'datasets/test/crosswalk/' + str(uuid.uuid4()) + '.png'
            else:
                filename = 'datasets/test/sidewalk/' + str(uuid.uuid4()) + '.png'


        cv2.imwrite(filename, crop_img)


        cv2.imshow('preview', crop_img)

        cv2.rectangle(param, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), boxColor, -1)

def loadImage(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    n = np.fromfile(filename, dtype)
    image = cv2.imdecode(n, flags)

    drawLines(image, lineWidth)

    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, onMouse, param=image)

    while True:
        cv2.imshow(windowName, image)

        key = cv2.waitKey(1)

        if key == 27:
            break

    cv2.destroyAllWindows()

with open('train_sets.json', encoding='UTF8') as json_file:
    train_sets = json.load(json_file)

absolute_directory = train_sets['absolute_directory']

for _, data_sets in enumerate(train_sets['data_sets']):
    region_name = train_sets['data_sets'][data_sets]['region']
    for currentNumber, directories in enumerate(train_sets['data_sets'][data_sets]['directories']):
        file_name = train_sets['data_sets'][data_sets]['filenames'][currentNumber]
        file_extention = train_sets['data_sets'][data_sets]['extentions']
        for image_index in range(100):
            print(f'{currentNumber} - {image_index}')
            image_index += 1
            if len(str(image_index)) < 2:
                image_index = '0' + str(image_index)
            loadImage(f'{absolute_directory}/{region_name}/{directories}/{file_name}{image_index}.{file_extention}')