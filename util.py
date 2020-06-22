#Mapping tool to teach CNN model

import cv2
import math

windowName = 'skymap'
lineWidth = 20
color = (255, 255 ,255)


#TODO line들의 좌표값또한 return 필요 (button이라는 형식으로 좌표 range가 좋을듯)
def drawLines(image, width):
    h, w, _ = image.shape

    #TODO Rectangle로 대체 가능?
    for i in range(round(w / width)):
        x, y = width * i, width * i
        image = cv2.line(image, (x, 0), (x, h), color)
        image = cv2.line(image, (0, y), (w, y), color)

image = cv2.imread('skymap.png')
drawLines(image, lineWidth)

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xround = math.ceil(x / lineWidth)
        yround = math.ceil(y / lineWidth)

        cv2.rectangle(image, ((xround - 1) * lineWidth, (yround - 1 )* lineWidth), (xround * lineWidth, yround * lineWidth), color, -1)

cv2.namedWindow(windowName)
cv2.setMouseCallback(windowName, onMouse)

while True:
    cv2.imshow(windowName, image)

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()