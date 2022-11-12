import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# features = np.load("feature.npy")
# labels = np.load("labels.npy")

camera = cv.VideoCapture(0)


people = ['chetan', 'dinesh', 'nayan']
dir = r"E:\Coding\learn\opencv\one\train\chetan\WhatsApp Image 2022-11-12 at 11.45.54 AM (1).jpeg"
dir = r"E:\Coding\learn\opencv\one\train\chetan\WhatsApp Image 2022-11-12 at 11.45.58 AM.jpeg"
face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.read('face_trained.yml')

img = cv.imread(dir)
result, img = camera.read()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
face_rect = haar_cascade.detectMultiScale(gray, 1.1, 3)

for(x,y,w,h) in face_rect:
    face_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(face_roi)
    print(f"Label: {people[label]} with a condidence of {100-confidence}")