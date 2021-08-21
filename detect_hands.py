#
# Code is based on documentation provided by:
# "https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html"
# Functions presented in this file need special environment to be run 
#

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from cv2 import cv2 as cv
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


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


def load_image_into_numpy_array(path):
    """
    Loads and transforms image into numpy array
    Args:
        path: image path

    Returns:
        np array  (h, w, 3)
    """
    return np.array(Image.open(path))

def detect_hands(path, dest_path, model_path):
    """
    Function uses trained model to find hands and separate
    them from the background. It later creates a new black 
    background and saves the image in a new directory. 

    Args:
        path: path to images
        dest_path: path where new images will be saved
        model_path: path to the model
    Returns:   
        Saves images without the background in 
        specified directory
    """
    
    # Load model and path
    detect_fn = tf.saved_model.load(model_path)
    images = list_paths(path)
    
    
    j = 0
    for img_path in images:
        image_np = load_image_into_numpy_array(img_path)


        # Converts np to tensor
        input_tensor = tf.convert_to_tensor(image_np)
        # Expand the tensor to be compatible
        input_tensor = input_tensor[tf.newaxis, ...]

        # Start detections
        detections = detect_fn(input_tensor)
        category_index = label_map_util.create_category_index_from_labelmap('label_map.pbtxt',
                                                                        use_display_name=True)
        
        # Revowe earlier added demension
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # Assign detections as integers
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # If the model is 65% or more sure that the image 
        # includes hands it collects the coordinations of
        # the box
        if detections['detection_scores'][0] > 0.65:
            image = image_np.copy()
            (h, w) = image.shape[:2]
            
            # Creates black empty image 
            scanned = np.zeros((h, w, 3), np.uint8)
            scanned.fill(0)

            boxes = detections['detection_boxes']
            for i in range(0,len(detections['detection_scores'])):
                if detections['detection_scores'][i] > 0.65:
                    ymin = int((np.squeeze(boxes)[i][0]*h))
                    xmin = int((np.squeeze(boxes)[i][1]*w))
                    ymax = int((np.squeeze(boxes)[i][2]*h))
                    xmax = int((np.squeeze(boxes)[i][3]*w))
                    scanned[ymin:ymax,xmin:xmax] = image[ymin:ymax,xmin:xmax].copy()
                    
            label = img_path.split(os.path.sep)[-2]
            j += 1
            
            # Puts hands on the black background and saves it to 
            # early specified directory
            name = '{}/{}/{}_{}.jpeg'.format(dest_path, label, label, str(j))
            final_img = cv.cvtColor(scanned, cv.COLOR_BGR2RGB)
            cv.imwrite(name, final_img)
            print(name,": saved")
