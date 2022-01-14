
import numpy as np
import cv2 as cv

haar_cascade=cv.CascadeClassifier('haar_face.xml')
people=['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling','Me']
#label=np.load('labels.npy')
#features=np.load('features.npy')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trainedMe.yml')
cap=cv.VideoCapture(0,cv.CAP_DSHOW)

while cap.isOpened():

    success,img=cap.read()

    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cv.imshow('Person',gray)

    faces_rect=haar_cascade.detectMultiScale(gray,1.1,4)

    for(x,y,w,h) in faces_rect:
        faces_roi=gray[y:y+h,x:x+h]

        label,confidence=face_recognizer.predict(faces_roi)
        if people[label]=='Me':

            print(f'Label={people[label]} with a confidence of {confidence}')

            cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        else:
            x=False    

    cv.imshow("Detected Face",img) 

    cv.waitKey(1) 