import cv2 as cv
import os
import numpy as np

people = []

features = []
labels = []

dir = r'E:\Coding\learn\opencv\one\train'
for i in os.listdir(dir):
    people.append(i)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

def create_train():
    for person in people:
        path = os.path.join(dir, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3) 

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print(f"length of features: {len(features)}")
print(f"length of labels: {len(labels)}")

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recogonizer = cv.face.LBPHFaceRecognizer_create()
face_recogonizer.train(features, labels)

face_recogonizer.save("face_trained.yml")
np.save("feature.npy", features)
np.save("labels.npy", labels)