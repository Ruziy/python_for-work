import cv2 
import numpy as np
cap = cv2.VideoCapture('videos/test-video.mp4')
faces = cv2.CascadeClassifier('faces.xml')
def nothing(x):
    pass
cv2.namedWindow('Track')
cv2.createTrackbar('T1','Track',0,255,nothing)
cv2.createTrackbar('T2','Track',0,255,nothing)
while True:
    succes,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    result = faces.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=4)
    thresh1 = cv2.getTrackbarPos('T1','Track')
    thresh2 = cv2.getTrackbarPos('T2','Track')
    canny = cv2.Canny(gray,thresh1,thresh2)
    cv2.imshow('canny',canny)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
    for (x,y,w,h) in result :
     cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),thickness=3)
     cv2.putText(img,'I am',(x+w//2,y+h+h//10),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),thickness=1)
    cv2.imshow('Result',img)