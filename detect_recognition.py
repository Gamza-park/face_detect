import cv2
import face_recognition
import pickle
import time

image_file = 'image/marathon_01.jpg'
encoding_file = 'encodings.pickle'
unknown_name = 'Unknown'

model_method = 'cnn'


def detectAndDisplay(image):
    start_time = time.time()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb,
                                            model=model_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    # initialize the list of names for each face detected
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = unknown_name

        # found a match
        if True in matches:
            # find the indexes of all matched faces then initialize
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # select first entry in the dictionary)
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0, 255, 0)
        line = 2
        if (name == unknown_name):
            color = (0, 0, 255)
            line = 1
            name = 'Unknown'

        cv2.rectangle(image, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, color, line)

    end_time = time.time()
    process_time = end_time - start_time
    print("frame took {:.3f} sec".format(process_time))
    # show the output image
    cv2.imshow("Recognition", image)


# load the known faces and embeddings
data = pickle.loads(open(encoding_file, "rb").read())

# load the input image
image = cv2.imread(image_file)
detectAndDisplay(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
