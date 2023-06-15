import cv2
import sys
import numpy as np
import os
import ffmpeg
import imageio
import shutil


vid = cv2.VideoCapture('images\\vid1.mp4')
success, image = vid.read()
count = 0


print("running")

os.mkdir('tempFrames')
os.mkdir('edittedFrames')
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (1280, 720))

while success:
    filepath = "tempFrames\\frame%d.jpg" % count
    cv2.imwrite(filepath, image)
    success, image = vid.read()
    count += 1
    print('Read a new frame: ', success)
    
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    frontal_faces = classifier.detectMultiScale(gray, 1.1, 4)

    profile_classifier = cv2.CascadeClassifier('haarcascade_profileface.xml')
    profile_faces = profile_classifier.detectMultiScale(gray, 1.1, 4)

    faces = [frontal_faces, profile_faces]

    for f in faces:
        for (x, y, w, h) in f:
            face = img[y:y+h, x:x+w]
            face = cv2.GaussianBlur(face, (min(23, h), min(23,w)), 80)
            img[y:y+face.shape[0], x:x+face.shape[1]] = face
    
    cv2.imwrite("edittedFrames\\frame%d.jpg" % count, img)
    out.write(img)



out.release()

shutil.rmtree('tempFrames')
shutil.rmtree('edittedFrames')
