import os
import cv2
import warnings
import numpy as np
import pandas as pd
import Preprocessing03_find_object as PFO
import Preprocessing02_build_dataset as PBD

#1 Ignore warnings
warnings.filterwarnings("ignore")

#2 Define different filepaths
global_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset'
otb50_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset/OTB50'
pred_box_path = 'E:/ChemEng788_789/DataSet/pred_bbox.csv'

#3 Create img with true_box and pred_box
def number(filtpath, foldername):
    number = 0
    filepath = filtpath + '/' + foldername + '/' + 'img'
    for fn in os.listdir(filepath):
        number = number + 1
    return number

def create_img_box(datasetpath, foldername):
    filepath_true = datasetpath + '/' + foldername + '/' + 'true'
    filepath_pred = datasetpath + '/' + foldername + '/' + 'pred'
    if os.path.isdir(filepath_true) == False:
            os.mkdir(filepath_true)
    if os.path.isdir(filepath_pred) == False:
            os.mkdir(filepath_pred)
    pred_path = datasetpath + '/' + foldername + '/' + 'pred_bbox.csv'
    img = PBD.X_dataset(datasetpath, foldername)
    true_box = PBD.y_dataset(datasetpath, foldername)
    pred_box = np.loadtxt(open(pred_path,"rb"), dtype=int, delimiter=",", skiprows=0)
    num = number(datasetpath, foldername)
    for i in range(num):
        print(i)
        withtruebox = PFO.identify_object(img[i], true_box[i])
        cv2.imwrite(filepath_true + "/" + str(i) + ".jpg", withtruebox)
        withpredbox = PFO.identify_object(img[i], pred_box[i])
        cv2.imwrite(filepath_pred + "/" + str(i) + ".jpg", withpredbox)

foldername = 'Biker'
create_img_box(otb50_filepath, foldername)