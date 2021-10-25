import numpy as np
import torch

#1 Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#2 Define different filepaths
global_filepath = 'E:/ChemEng788 789/Data Set'
otb50_filepath = 'E:/ChemEng788 789/Data Set/OTB50'
otb100_filepath = 'E:/ChemEng788 789/Data Set/OTB100'
#os.chdir(global_filepath)

#3 Image Pre-Processing
#3.1 To identify the object in the images and crop it out
import cv2
def identify_object(X_image, y_boundingbox):  #Identify the object with a blue rectangular (rectangular width = 1)
    X_image_with_rect = cv2.rectangle(X_image, (y_boundingbox[0]-1, y_boundingbox[1]-1), (y_boundingbox[0]+y_boundingbox[2]+1, y_boundingbox[1]+y_boundingbox[3]+1), (0,255,0), 1)
    return X_image_with_rect
def identify_object_list(X_dataset, y_dataset):  #Add the bounding boxes to all the images
    X_dataset_with_rect = []
    for i in range(len(X_dataset)):
        image = X_dataset[i]
        X_image_with_rect = identify_object(image, y_dataset[i])
        X_dataset_with_rect.append(X_image_with_rect)
    return X_dataset_with_rect
def crop_object(X_image, y_boundingbox): #Crop the object out
    cropped_image = X_image[y_boundingbox[1]:y_boundingbox[1]+y_boundingbox[3], y_boundingbox[0]:y_boundingbox[0]+y_boundingbox[2]]
    return cropped_image
def crop_object_list(X_dataset, y_dataset):  #Add the bounding boxes to all the images
    y_dataset_visualized = []
    for i in range(len(X_dataset)):
        y_value_cropped = crop_object(X_dataset[i], y_dataset[i])
        y_dataset_visualized.append(y_value_cropped)
    return y_dataset_visualized

#Now the object is identified and cropped out in each image
import Preprocessing02_build_dataset as PBD
import matplotlib.pyplot as plt
David_X = PBD.X_dataset('David')
David_y = PBD.y_dataset('David')
Basketball_X = PBD.X_dataset('Basketball')
plt.figure(); plt.imshow(Basketball_X[0]); plt.show()
a = identify_object_list(David_X, David_y)[0]
plt.figure(); plt.imshow(a); plt.show()
b = crop_object_list(David_X, David_y)[0]
plt.figure(); plt.imshow(b); plt.show()

#3.2 Import different image processing functions, such as to_gray_scale, zoom, etc
