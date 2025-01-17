import cv2
import numpy as np

face_cascade_name = '../../haarcascade_frontalface_alt.xml'
eyes_cascade_name = '../../haarcascade_eye_tree_eyeglasses.xml'
file_name = 'video/face_01.mp4'

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:

        center = (x + w // 2, y + h // 2)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        faceROI = frame_gray[y:y + h, x:x + w]
        # detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)
    cv2.imshow("Capture - Face", frame)

face_cascade = cv2.CascadeClassifier("../../haarcascade_frontalface_alt.xml")
eyes_cascade = cv2.CascadeClassifier("../../haarcascade_eye_tree_eyeglasses.xml")

# # 1. Load the cascades
# if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
#     print('Error loading face cascade')
#     exit(0)
# if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
#     print('Error loading eyes cascade')
#     exit(0)

# 2. Read the video stream
# cap = cv2.VideoCapture(file_name)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("error opening video")
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print("No Frame")
        break
    detectAndDisplay(frame)

    # push the 'q' Key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break