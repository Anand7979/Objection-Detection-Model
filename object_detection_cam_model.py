##########################  MODULE 3  ########################################

import numpy as np    #For create the array and mathematical calculation
import os             #It is used for use the operating system
import six.moves.urllib as urllib     #Send the  web request to COCO model
import sys            #Access the variables which is maintained by interpreter
import tarfile        #provides the tools to manage compressed files
import tensorflow as tf   #Used to create deep learning models
import zipfile            #Used to work with ZIP archives

from collections import defaultdict   #It implements specialized container(dict,list,tuple and set)
from io import StringIO              #It is used to manage the file related input and output operations
from matplotlib import pyplot as plt  #It is used for plotting the graph
from PIL import  Image               #Used for work with images

from utils import label_map_util   #It is a collection of small python functions and classes
from utils import visualization_utils as vis_util


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #create environment by tensorflow

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'  # pretrained model in COCO
MODEL_FILE = MODEL_NAME+'.tar.gz'   #model name with extension
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'  #download from this url

PATH_TO_CKPT = MODEL_NAME + './frozen_inference_graph.pb'  #save this model

PATH_TO_LABELS = os.path.join('C:/Users/it/PycharmProjects/objectdetection-master/objectdetection-master/object_recognition_detection/object_recognition_detection/data','mscoco_label_map.pbtxt')  #give the label to the data

NUM_CLASSES = 90  #number of classes

if not os.path.exists(MODEL_NAME + './frozen_inference_graph.pb'):  #model is exists or not
    print('Downloading coco model')
    opener = urllib.request.URLopener()   #send the request
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)   #help in download
    tar_file = tarfile.open(MODEL_FILE)   #open this file and extract
    for file in tar_file.getmembers():     # we can access every member of this file
        file_name = os.path.basename(file.name)   #check the file
        if 'frozen_inference_graph.pb' in file_name:   # if this file exists
            tar_file.extract(file, os.getcwd())     #extract file
    print('Downloading coco model complete!!!')

else:
    print('Coco Model Already Exists!!!')

detection_graph = tf.Graph()  #load the graph
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()    #define the graph
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()   # read the graph
        od_graph_def.ParseFromString(serialized_graph)   #formating of the data
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)   #load the labels
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes = NUM_CLASSES, use_display_name = True) #divide labels in different classes
categories_index = label_map_util.create_category_index(categories) #give the index to the category

#########################  MODULE 4  ####################################
# intializing the web camera device

import cv2
cap = cv2.VideoCapture(0)

# Running the tensorflow session

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        ret = True
        while(ret):
            ret, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)  # expand the dimension of image
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')  # define the scores
            classes = detection_graph.get_tensor_by_name('detection_classes:0')  # define the classes
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')  # detect the number

            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={
                image_tensor: image_np_expanded})  # for running the process

            vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                               np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32),
                                                               # for visulation and work on image
                                                               np.squeeze(scores),
                                                               categories_index,
                                                               use_normalized_coordinates=True,
                                                               line_thickness=8)

            cv2.imshow('image',cv2.resize(image_np,(1280,960)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                cap.release()
                break
