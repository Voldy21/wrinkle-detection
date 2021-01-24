import cv2
import numpy as np
from matplotlib import pyplot as plt
# from IPython.display import display, Image
import math


def resize(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


lefteye_cascade = cv2.CascadeClassifier(
    "Resources\haarcascade_lefteye_2splits.xml")
righteye_cascade = cv2.CascadeClassifier(
    "Resources\haarcascade_righteye_2splits.xml")
face_cascade = cv2.CascadeClassifier(
    "Resources\haarcascade_frontalface_default.xml")
img = cv2.imread("Resources/face5.jpg")
while img.shape[0] > 400:
    print(img.shape)
    img = resize(img, 90)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray_img, scaleFactor=1.05, minNeighbors=10)
edges = cv2.Canny(gray_img, 100, 100)
cv2.imshow("edges", edges)
cv2.waitKey(0)

# for x, y, w, h in faces:
#     cropped_img = img[y:y+h, x:x+w]

#     lefteye = lefteye_cascade.detectMultiScale(
#         cropped_img, scaleFactor=1.05, minNeighbors=10)
#     righteye = righteye_cascade.detectMultiScale(
#         cropped_img, scaleFactor=1.05, minNeighbors=10)
#     for x, y, w, h in lefteye:
#         cv2.rectangle(cropped_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
#     for x, y, w, h in righteye:
#         cv2.rectangle(cropped_img, (x, y), (x+w, y+h), (0, 255, 0), 1)

#     cv2.imshow("newface", cropped_img)
#     cv2.waitKey(0)
