#Mapping tool to teach CNN model
import uuid

import cv2
import math

isTraninMode = True
isCrossWalk = True

windowName = 'skymap'
lineWidth = 15
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

image = cv2.imread('skymap.png')
drawLines(image, lineWidth)

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xround = math.ceil(x / lineWidth)
        yround = math.ceil(y / lineWidth)

        y1 = (yround - 1) * lineWidth
        y2 = yround * lineWidth
        x1 = (xround - 1) * lineWidth
        x2 = xround * lineWidth

        crop_img = image[y1:y2, x1:x2]
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

        cv2.rectangle(image, (x1, y1), (x2, y2), boxColor, -1)


cv2.namedWindow(windowName)
cv2.setMouseCallback(windowName, onMouse)

while True:
    cv2.imshow(windowName, image)

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()