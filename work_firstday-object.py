import cv2 
import numpy as np
from math import atan, pi,ceil
cap = cv2.VideoCapture('videos/videos-object-7.mp4')
choose = input('Choose num peaks')
choose = int(choose)
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
    contours,h = cv2.findContours(dill,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            cv2.drawContours(new_img,contour,-1,(200,200,0),3)
            p = cv2.arcLength(contour,True)
            # print(p)
            num = cv2.approxPolyDP(contour,0.3*p,True)
            cv2.line(img,(520,200),(526,193),(119,201,105),thickness=3)
            x,y,w,h = cv2.boundingRect(num)
            if len(num) == choose:
                print(x,y,w,h)
                cv2.rectangle(img,(x,y,x+w,y+h),(0,0,255),3)   
                parall = ceil(atan(abs((x-y)/1+x*y))*180/pi)
                if choose == 4 and parall == 90:
                    cv2.putText(img,'90 degree',(x+w//4,y+h+h//6),cv2.FONT_HERSHEY_TRIPLEX,0.6,(0,255,0),thickness=1)
    print(max(contour[0][0]))
    Hori = np.concatenate((dill, gray), axis=1)
    Hori_2 = np.concatenate((new_img,img), axis=1)
    cv2.imshow('One layer', Hori)
    cv2.imshow('More layers', Hori_2)
    if cv2.waitKey(50) & 0xFF == ord('q'): 
        break