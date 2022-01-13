
import os
import cv2 as cv
import numpy as np

p=[]
for i in os.listdir( r'C:\Users\dell\OneDrive\Codes\OpenCv\Photos\train'):
    p.append(i)

print(p)
DIR=r'C:\Users\dell\OneDrive\Codes\OpenCv\Photos\train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')
features=[]

labels=[]

def create_train():
    for person in p:
        path=os.path.join(DIR,person)
        label=p.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            img_array=cv.imread(img_path)
            gr=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            faces_rect= haar_cascade.detectMultiScale(gr,scaleFactor=1.1,minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi=gr[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()

features=np.array(features,dtype='object')
labels=np.array(labels)
print(len(features))

face_recog=cv.face.LBPHFaceRecognizer_create()

#Train face reccog  on features

face_recog.train(features,labels)

face_recog.save('face_trainedMe.yml')

np.save('features.npy',features)
np.save('labels.npy',labels)