import cv2 
import numpy as np
from math import atan, pi,ceil
cap = cv2.VideoCapture('videos/videos-object-7.mp4')
def nothing(x):
    pass
cv2.namedWindow('Track')
cv2.createTrackbar('T1','Track',0,255,nothing)
cv2.setTrackbarPos('T1','Track',24)
cv2.createTrackbar('T2','Track',0,255,nothing)
cv2.setTrackbarPos('T2','Track',33)
kernel = np.ones((4,4))
while True:
    succes,img = cap.read()
    img = cv2.resize(img,(800,600))
    img = cv2.GaussianBlur(img,(9,9),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh1 = cv2.getTrackbarPos('T1','Track')
    thresh2 = cv2.getTrackbarPos('T2','Track')
    canny = cv2.Canny(gray,thresh1,thresh2)
    dill = cv2.dilate(canny,kernel,iterations=2)
    new_img = img.copy()
    contours,h = cv2.findContours(dill,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10000 :
                rect = cv2.minAreaRect(cnt) 
                box = cv2.boxPoints(rect) 
                box = np.intp(box) 
    cv2.drawContours(img,[box],0,(0,255,0),5)
    Hori = np.concatenate((dill, gray), axis=1)
    Hori_2 = np.concatenate((new_img,img), axis=1)
    cv2.imshow('One layer', Hori)
    cv2.imshow('More layers', Hori_2)
    if cv2.waitKey(50) & 0xFF == ord('q'): 
        break