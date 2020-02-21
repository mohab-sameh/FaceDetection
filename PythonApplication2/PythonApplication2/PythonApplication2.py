
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')


def detection(grayscale, img):
    face = face_cascade.detectMultiScale(grayscale, 1.3, 5)
    for (x_face, y_face, w_face, h_face) in face:
        ri_grayscale = grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
        ri_color = img[y_face:y_face+h_face, x_face:x_face+w_face]
        smile = smile_cascade.detectMultiScale(ri_grayscale, 1.8, 65)
        for (x_smile, y_smile, w_smile, h_smile) in smile: 
            cv2.rectangle(ri_color,(x_smile, y_smile),(x_smile+w_smile, y_smile+h_smile), (0, 100, 255), 2)
    return img 


cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(grayscale, 1.2, 4)
    eyes = eye_cascade.detectMultiScale(grayscale, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


    final = detection(grayscale, img)
    cv2.imshow('Video', final)



    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
cap.release()