import cv2
import numpy as np
img = cv2.imread('images/work-9.jpg')
img = cv2.resize(img,(600,400))
img=cv2.GaussianBlur(img,(3,3),0)
trained_face_data = cv2.CascadeClassifier('faces.xml')
trained_eye_data = cv2.CascadeClassifier('eye.xml')
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img ,scaleFactor=1.3,minNeighbors=3)
eye_coordinates = trained_eye_data.detectMultiScale(grayscaled_img ,scaleFactor=1.3,minNeighbors=4)
if len(face_coordinates) != 0:
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0, 255), 2)
        cv2.putText(img,'Person',(x+w//4,y+h+h//6),cv2.FONT_HERSHEY_TRIPLEX,0.6,(0,0,255),thickness=1)
if len(face_coordinates) == 0:
    kernel = np.ones((5,5))
    canny = cv2.Canny(grayscaled_img,200,30)
    dill = cv2.dilate(canny,kernel,iterations=1)
    contours,h = cv2.findContours(dill,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for (x, y, w, h) in eye_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.putText(img,'Eye',(x+w//4,y+h+h//3),cv2.FONT_HERSHEY_TRIPLEX,0.6,(0,255,0),thickness=1)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 7000:
            img_withContors = img.copy()
            img_withContors = cv2.drawContours(img_withContors,contour,-1,(200,200,0),3)
            num = cv2.approxPolyDP(contour,1,False)
            x,y,w,h = cv2.boundingRect(num)
            cv2.rectangle(img,(x, y), ((x + w), (y + h)),(255,0,0),3) 
            # if len(eye_coordinates) != 0:
            #     for (x_2, y_2, w_2, h_2) in eye_coordinates:  
            #         cv2.rectangle(img,(x, y), ((x + w), (y + h)//3),(0,0,255),3) 
        # elif area > 5000 and area < 7000 :
        #     img_withContors = img.copy()
        #     img_withContors = cv2.drawContours(img_withContors,contour,-1,(200,200,0),3)
        #     num = cv2.approxPolyDP(contour,1,False)
        #     x,y,w,h = cv2.boundingRect(num)
        #     cv2.rectangle(img,(x,y,x+w,y+h),(0,0,255),3)
        #     print(h)
    

cv2.imshow('Rezult', img)
cv2.waitKey(0)
