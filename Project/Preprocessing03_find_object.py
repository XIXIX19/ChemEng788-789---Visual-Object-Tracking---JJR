import numpy as np
import torch

#1 Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#2 Define different filepaths
global_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset'
otb50_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset/OTB50'

#3 Image Pre-Processing
#3.1 To identify the object in the images and crop it out
import cv2
import matplotlib.pyplot as plt
def identify_object(X_image, y_boundingbox):  #Identify the object with a blue rectangular (rectangular width = 1)
    #if y_boundingbox[0]<0 or y_boundingbox[1]<0 or y_boundingbox[2]<0 or y_boundingbox[3]<0:
    #    y_boundingbox = [0,0,0,0]
    #    y_boundingbox = np.array(y_boundingbox)
    X_image_with_rect = cv2.rectangle(X_image, (y_boundingbox[0], y_boundingbox[1]), (y_boundingbox[0]+y_boundingbox[2]+1, y_boundingbox[1]+y_boundingbox[3]+1), (0,255,0), 1)
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


"""#Now the object is identified and cropped out in each image
import Preprocessing02_build_dataset as PBD
import matplotlib.pyplot as plt
import os
string1 = 'Basketball'
X = PBD.X_dataset(otb50_filepath, string1)
y = PBD.y_dataset(otb50_filepath, string1)

frame = 0
image = X[frame]
box = y[frame]
plt.figure(); plt.imshow(image); plt.show()
a = identify_object(image, box)
plt.figure(); plt.imshow(a); plt.show()
b = crop_object(image, box)
plt.figure(); plt.imshow(b); plt.show()

file_path = 'E:/ChemEng788_789/DataSet/OTB50/Basketball/img1'
for i in range(300):
    withbox = identify_object(X[i], y[i])
    cv2.imwrite(file_path + "/" + str(i) + ".jpg", withbox)"""
