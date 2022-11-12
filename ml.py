import os
import cv2 as cv
# camera = cv.VideoCapture(0)

people = []
feature = []
labels = []

for i in os.listdir(r'E:\Coding\learn\opencv\one\train'):
    people.append(i)
dir  = r"E:\Coding\learn\opencv\one\train"

haar_cascade = cv.CascadeClassifier('haar_face.xml')
def create_train():
    for person in people:
        path = os.path.join(dir, person)
        label = people.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in face_rect:
                faces_roi = gray[y:y+h, x:x+w]
                feature.append(faces_roi)
                labels.append(label)

def lets_train():
    path = os.listdir.join(dir, "chetan")
create_train()

print(f"Length of the features = {len(feature)}")
print(f"Length of the labels = {len(labels)}")
# while True:
#     result, frame = camera.read()
#     if result:
#         img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
#         print(f"Total numeber of faces found in the image: {len(faces_rect)}")
#         for (x, y, w, h) in faces_rect:
#             cv.rectangle(frame,(x, y), (x+w,y+h), (0, 255, 0), thickness=2)
#         cv.imshow("Detected faces", frame)
#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break
# cv.destroyAllWindows()
# camera.release()