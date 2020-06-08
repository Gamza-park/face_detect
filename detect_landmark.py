import numpy as np
import dlib
import cv2

EYEBROWS = list(range(17, 27))
JAWLINE = list(range(17, 27))
NOSE = list(range(27, 36))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
ALL = list(range(0,68))

predictor_file = 'shape_predictor_68_face_landmarks.dat'
image_file = 'image/marathon_02.jpg'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
image = cv2.resize(image,(1000,1000))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

for(i, rect) in enumerate(rects):
    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    show_parts = points[ALL]

    for(i, point) in enumerate(show_parts):
        x = point[0,0]
        y = point[0,1]
        cv2.circle(image, (x,y), 1, (0,255,255), -1)
        cv2.putText(image, "{}".format(i + 1), (x , y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,2), 1)

cv2.imshow("image", image)
cv2.waitKey(0)