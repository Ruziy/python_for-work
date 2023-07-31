import cv2
import numpy as np

def auto_scaleFactor_forBody(turple,num_person):
        scaleFact = 1.0
        while(len(turple) != num_person):
                scaleFact = scaleFact+0.01
                if(int(scaleFact) == 4):
                    break
                turple = trained_body_data.detectMultiScale(grayscaled_img ,scaleFactor=scaleFact,minNeighbors=3)
                return turple


img = cv2.imread('images/test-upperbody.jpg')
img = cv2.resize(img,(600,400))
img=cv2.GaussianBlur(img,(3,3),0)

trained_face_data = cv2.CascadeClassifier('faces.xml')
trained_body_data = cv2.CascadeClassifier('upperbody.xml')
trained_eye_data = cv2.CascadeClassifier('eye.xml')

print("Choose numb of person\n")
num_of_person = int(input())
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
new_canny = cv2.Canny(grayscaled_img,200,50)

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img ,scaleFactor=1.3,minNeighbors=3)
eye_coordinates = trained_eye_data.detectMultiScale(grayscaled_img ,scaleFactor=1.05,minNeighbors=3)
body_coordinates = trained_body_data.detectMultiScale(grayscaled_img ,scaleFactor=1.03,minNeighbors=3)

if len(body_coordinates) != 0:
        scaleFact = 1.0
        while(len(body_coordinates) != num_of_person):
                scaleFact = scaleFact+0.01
                if(int(scaleFact) == 4):
                    break
                body_coordinates = trained_body_data.detectMultiScale(grayscaled_img ,scaleFactor=scaleFact,minNeighbors=3)      
        x,y,w,h = body_coordinates[0]   
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
        cv2.putText(img,'Body',(x+w//4,y+h+h//4),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,255),thickness=1)

if len(face_coordinates) != 0 :
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0, 255), 2)
        cv2.putText(img,'Person',(x+w//4,y+h+h//6),cv2.FONT_HERSHEY_TRIPLEX,0.6,(0,0,255),thickness=1)

if len(face_coordinates) == 0 and len(body_coordinates)== 0:
        scaleFact = 1.0
        while(len(eye_coordinates) != num_of_person):
                scaleFact = scaleFact+0.01
                if(int(scaleFact) == 4):
                    break
                eye_coordinates = trained_eye_data.detectMultiScale(grayscaled_img ,scaleFactor=scaleFact,minNeighbors=3)
        for (x, y, w, h) in eye_coordinates[0:num_of_person]:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
                            cv2.rectangle(img, (x-w*2, y-h*2), (x + w*2, y + h*4), (0,0,255), 2)
                            cv2.putText(img,'Eye',(x+w//4,y+h+h//3),cv2.FONT_HERSHEY_TRIPLEX,0.6,(0,255,0),thickness=1)


cv2.imshow('Rezult', img)
cv2.waitKey(0)
