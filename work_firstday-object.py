import cv2 
import numpy as np
cap = cv2.VideoCapture('videos/videos-object-3.mkv')
def nothing(x):
    pass
cv2.namedWindow('Track')
cv2.createTrackbar('T1','Track',0,255,nothing)
cv2.createTrackbar('T2','Track',0,255,nothing)
kernel = np.ones((5,5))
while True:
    succes,img = cap.read()
    img = cv2.resize(img,(600,400))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh1 = cv2.getTrackbarPos('T1','Track')
    thresh2 = cv2.getTrackbarPos('T2','Track')
    canny = cv2.Canny(gray,thresh1,thresh2)
    dill = cv2.dilate(canny,kernel,iterations=1)
    
    contours,h = cv2.findContours(dill,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            cv2.drawContours(img,contour,-1,(200,200,0),3)
            p = cv2.arcLength(contour,True)
            num = cv2.approxPolyDP(contour,0.01*p,True)
            x,y,w,h = cv2.boundingRect(num)
            if len(num) == 3:
                cv2.rectangle(img,(x,y,x+w,y+h),(0,0,255),4)            
    cv2.imshow('canny',canny)
    cv2.imshow('dill',dill)
    cv2.imshow('img',img)
    if cv2.waitKey(50) & 0xFF == ord('q'): 
        break