import os
import numpy as np
import re

#1 Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#2 Define different filepaths
global_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset'
otb50_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset/OTB50'
#os.chdir(global_filepath)

#3 Build the datasets
"""
#There are 100 datasets, and each dataset will be imported seperately
#The images in each folder are the input datasets
#The groundtruth values (each line) in the text files are the output datasets
"""
#3.1 Build the X datasets
import glob
import cv2
from PIL import Image
import matplotlib.pyplot as plt
def X_dataset(datasetpath, read_foldername):   #grab the images and put into the input dataset
    X_eachfolder = []
    folderpath = datasetpath + '/' + read_foldername + '/img'
    file_name = glob.glob(folderpath + "/*.jpg")
    for i in range(len(os.listdir(folderpath))):
        img = cv2.imread(file_name[i])
        X_eachfolder.append(img)
    return X_eachfolder
"""
#Now different input datasets could be imported, for example:
import matplotlib.pyplot as plt
David_X = X_dataset('David')
img = David_X[0]; plt.figure(); plt.imshow(img); plt.show()
"""
#3.2 Build the output dataset
def line_to_array(line):
    new_line1 = []
    new_line2 = line.replace('\n', '')

    if ',' in new_line2:
        for i in new_line2.split(','):
           new_line1.append(int(i))

    if '\t' in new_line2:
        for i in new_line2.split('\t'):
            new_line1.append(int(i))

    new_line = np.array(new_line1)
    return new_line
def y_dataset(datasetpath, read_foldername):
    y_eachline = []
    fileloc = datasetpath + '/' + read_foldername + '/groundtruth_rect.txt'
    text = open(fileloc)
    for line in text.readlines():
        new_line = line_to_array(line)
        y_eachline.append(new_line)
    return y_eachline


#Now different output datasets could be imported, for example:
#David_y = y_dataset('David')
#print(David_y[0])
