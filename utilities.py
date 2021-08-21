import os
import time
import operator
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2 as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.backend import expand_dims


def list_paths(path):
    '''
    Lists all image paths and labels from a given directory.
    '''

    images = list()
    labels = list()
    for name in os.listdir(path):
        if name != ".DS_Store":
            labels.append(name)

    for label in labels:
        full_path = os.path.join(path, label)
        for img in os.listdir(full_path):
            images.append(os.path.join(full_path, img))
    return images


def create_dataset_and_labels(images_paths):
    ''' 
    Reads given paths and saves images/labels and puts them into
    separate lists
    '''

    images = list()
    labels = list()
    lower_skin = np.array([0, 58, 30])
    upper_skin = np.array([33, 255, 255])

    for path in images_paths:
        try:
            image = cv.imread(path)
            images.append(image)
            labels.append(path.split(os.path.sep)[-2])
        except:
            print(path + ' not found')
            continue

    images = np.array(images, dtype="float") / 255.0
    labels = np.array(labels)
    return images, labels


def resize_and_rename_dataset(path, dest_path):
    '''
    Goes through all the image paths, renames and resizes the images.
    Safes images in a new direction
    '''

    images = list_paths(path)
    i = 0
    print(images)
    for img in images:
        try:

            label = img.split(os.path.sep)[-2]
            image = cv.imread(img)
            image = cv.resize(image, (300, 300))
            i += 1
            name = '{}/{}/{}_{}.png'.format(dest_path, label, label, str(i))
            cv.imwrite(name, image)
        except:
            print(path + ' not found')
            continue


def take_pictures(start_name, end_name, camera_num):
    ''' 
    Starts taking pictures with a one second pause,
    Functions created to fast generate images for the
    dataset.  
    '''

    video = cv.VideoCapture(camera_num)
    name = start_name
    while name <= end_name:

        name += 1
        ret, frame = video.read()
        cv.imwrite("./dataset/{}.png".format(str(name)), frame)
        print(name)
        time.sleep(1)

    video.release()
    cv.destroyAllWindows()


def predict_sign(frame, model):
    ''' 
    Gets video frame, converts it to array and uses model.predict()
    function to get prediction with loaded trained model.
    Prints prediction in terminal and creates empty dictionary.
    Assigns values to particular classes and checks which one has 
    the highest propability.
    '''
    # List of all the classes
    sign_class = ["A", "B", "C", "D", "F", "G", "L",
                  "M", "N", "P", "Q", "R", "V", "W", "X", "Y", "Z"]

    # Loads trained model

    # model_path = '/models/FourthModel'

    model = load_model(os.path.join('./models/FourthModel'))

    frame = cv.resize(frame, (300, 300))
    frame = img_to_array(frame, dtype='float32')
    prediction = model.predict(expand_dims(frame,  axis=0))
    prediction = prediction[0]
    predictions = {}
    predictions[sign_class[0]] = prediction[0]
    predictions[sign_class[1]] = prediction[1]
    predictions[sign_class[2]] = prediction[2]
    predictions[sign_class[3]] = prediction[3]
    predictions[sign_class[4]] = prediction[4]
    predictions[sign_class[5]] = prediction[5]
    predictions[sign_class[6]] = prediction[6]
    predictions[sign_class[7]] = prediction[7]
    predictions[sign_class[8]] = prediction[8]
    predictions[sign_class[9]] = prediction[9]
    predictions[sign_class[9]] = prediction[10]
    predictions[sign_class[9]] = prediction[11]
    predictions[sign_class[9]] = prediction[12]
    predictions[sign_class[9]] = prediction[13]
    predictions[sign_class[9]] = prediction[14]
    predictions[sign_class[9]] = prediction[15]
    predictions[sign_class[9]] = prediction[16]

    predicted = max(predictions.items(), key=operator.itemgetter(1))[0]
    return predicted
