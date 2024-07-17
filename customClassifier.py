# custom classifier trainer

import numpy as np
import cv2
from PIL import Image # pillow
import os

# training function for faces
def train(dataset_dir):

    # use os to take everything from the dataset file and put it under the localFiles list
    localFiles = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []

    # process each image and put them and their ids into trainable lists
    for frame in localFiles:
        img = Image.open(frame).convert('L') # convert grayscale using pillow to open images
        imgNp = np.array(img, 'uint8') # need numpy data array of image for training

        # split the file directory into its location and filename, take filename and split by periods, take the first number after 'user'
        id = int(os.path.split(frame)[1].split('.')[1])

        # add faces and ids to their respective lists
        faces.append(imgNp)
        ids.append(id)

    ids = np.array(ids) # turn to numpy array for training with image map array (np)

    clf = cv2.face.LBPHFaceRecognizer_create() # classifier training init
    clf.train(faces, ids) # train
    clf.write("customClassifier.yml") # trained dataset written