import cv2
import face_recognition
import pickle
import time

file_name = 'video/son_02.mp4'
encoding_file = 'encodings.pickle'
unknown_name = 'Unknown'

model_method = 'cnn'

def detectAndDisplay(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect
    boxes = face_recognition.face_locations(rgb, model=model_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    # initialize
    names = []

    # loop
    for encoding in encodings:
        # match each face
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = unknown_name

        # check to see
        if True in matches:
            # find the index
            matchedIdxs = [i for (i,b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        # update list
        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0,255,0)
        line = 2
        if(name == unknown_name):
            color = (255, 0, 0)
            line = 1
            name = ''
        cv2.rectangle(image, (left,top), (right,bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image,name,(left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, line)

    # show image
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    cv2.imshow("Detect", image)

# load
data = pickle.loads(open(encoding_file, "rb").read())

# read video
cap = cv2.VideoCapture(file_name)
if not cap.isOpened():
    print("error check video")
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print(' No capture')
    detectAndDisplay(frame)

    # push 'q' button to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
