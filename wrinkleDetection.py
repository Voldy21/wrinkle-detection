from skimage.data import camera
from skimage.filters import frangi, hessian

import cv2
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display, Image

face_cascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
img = cv2.imread("Resources/face3.jpg")

gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img,scaleFactor = 1.05,minNeighbors=10)

for x,y,w,h in faces :
    cropped_img = img[y:y+h,x:x+w]
    edges = cv2.Canny(np.mean(cropped_img, axis=2).astype(np.uint8), 50, 200)
    out = np.bitwise_or(cropped_img, edges[:,:,np.newaxis])
    # print(edges[0])
    cv2.imshow("out", out)
    cv2.waitKey(0)
    number_of_edges = np.count_nonzero(edges)

# it seems canny can only deal with 1 channel 8-bit images
# --> let's take the mean and convert to 8bit (could also use cvtColor instead)
canny = cv2.Canny(np.mean(img, axis=2).astype(np.uint8), 50, 200)
print(np.count_nonzero(canny))
# cv2.imwrite('canny.png', canny)
# for bitwise_or the shape needs to be the same, so we need to add an axis,
# since our input image has 3 axis while the canny output image has only one 2 axis
# out = np.bitwise_or(img, canny[:,:,np.newaxis])
# cv2.imwrite('out.png', out)
# cv2.imshow("out", out)