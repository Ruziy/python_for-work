import cv2

trained_face_data = cv2.CascadeClassifier('faces.xml')
img = cv2.imread('images/work-7.jpg')
img = cv2.resize(img,(600,400))
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img ,scaleFactor=1.3,minNeighbors=2)
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0,0, 255), 2)
    cv2.putText(img,'Person',(x+w//4,y+h+h//6),cv2.FONT_HERSHEY_TRIPLEX,0.6,(0,0,255),thickness=1)
cv2.imshow('Rezult', img)
cv2.waitKey(0)
