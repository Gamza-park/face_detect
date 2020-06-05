import cv2
import face_recognition
import pickle

dataset_paths = ['dataset/son/', 'dataset/tedy/']
names = ['Son', 'Tedy']
number_images = 10
image_type = '.jpg'
encoding_file = 'encodings.pickle'

model_method = 'cnn'

# initialize
knownEncodings = []
knownNames = []

# loop
for (i, dataset_path) in enumerate(dataset_paths):
    # extract the person name from names
    name = names[i]

    for idx in range(number_images):
        file_name = dataset_path + str(idx + 1) + image_type


        image = cv2.imread(file_name)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect
        boxes = face_recognition.face_locations(rgb,
                                                model=model_method)

        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            print(file_name, name, encoding)
            knownEncodings.append(encoding)
            knownNames.append(name)

# Save
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encoding_file, "wb")
f.write(pickle.dumps(data))
f.close()
