import os
import cv2
import pickle
import warnings
import numpy as np
from numpy.lib.type_check import imag
 
#1 Ignore warnings
warnings.filterwarnings("ignore")

#2 Define different filepaths
global_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset'
otb50_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset/OTB50'

#3 Store the images and labels into the arrays
#X and Z
def create_arrays_input(dataname, foldername, object = None):
    imagepath = dataname + '/' + foldername + '/' + object
    datasetlist = []
    for fn in os.listdir(imagepath):
        print(fn) #to know where the process is
        img = cv2.imread(imagepath + '/' +fn)
        """r, g, b = im.split()
        imageset = []
        im = Image.open(imagepath + '/' +fn)
        r_arrary = plimg.pil_to_array(r)
        g_arrary = plimg.pil_to_array(g)
        b_arrary = plimg.pil_to_array(b)
        #print(r_arrary, g_arrary, b_arrary)
        size = r_arrary.shape[0]
        imageset = np.ones((size,size,3))
        for i in range(size):
            for j in range(size):
                imageset[i][j][0] = int(r_arrary[i][j])
                imageset[i][j][1] = int(g_arrary[i][j])
                imageset[i][j][2] = int(b_arrary[i][j])"""
        datasetlist.append(img)
        #print(datasetlist)
        datasetarray = np.array(datasetlist)
    return datasetarray
#y
def true_label(pos_r,neg_r):
    size = 17
    pos_radius = pos_r
    neg_radius = neg_r
    label = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            a = np.sqrt((i-int(size/2))*(i-int(size/2)) + (j-int(size/2))*(j-int(size/2)))
            if a <= pos_radius:
                label[i,j] = 1
            elif a <= neg_radius:
                label[i,j] = -1
            else:
                label[i,j] = 0
    return label
def create_arrays_label(number,pos_r,neg_r):
    label_list = []
    label = true_label(pos_r,neg_r)
    for i in range(number):
        label_list.append(label)
    labels = np.array(label_list)
    return labels

#4 Store the arrays into a file
#4.1 X and Z
def create_XZ(filetpath,foldername, savefolder):
    template_dataset = create_arrays_input(filetpath, foldername, object = 'template')
    target_dataset = create_arrays_input(filetpath, foldername, object = 'target')
    target_0_dataset = create_arrays_input(filetpath, foldername, object = 'target_0')
    name = savefolder + '/' + foldername + '_dataset.pkl'
    with open(name, 'wb') as file:
        pickle.dump(template_dataset, file)
        pickle.dump(target_dataset, file)
        pickle.dump(target_0_dataset, file)
    print('end')

#4 Store the lists into a file
def number(filetpath, foldername):
    number = 0
    filepath = filetpath + '/' + foldername + '/' + 'img'
    for fn in os.listdir(filepath):
        number = number + 1
    return number
def create_y(filetpath, foldername, savefolder):
    file_num = number(filetpath, foldername)
    label_set = create_arrays_label(file_num,2,0)
    label_0_set = create_arrays_label(file_num,0,0)
    name = savefolder + '/'+ foldername + '_labelset.pkl'
    with open(name, 'wb') as file:
        pickle.dump(label_set, file)
        pickle.dump(label_0_set, file)
    print('end')

"""savefolder = 'C:/Users/lir58/Desktop/Jierui/data'
foldername = 'Woman'
create_XZ(otb50_filepath, foldername, savefolder)
create_y(otb50_filepath, foldername, savefolder)"""


savefolder = 'C:/Users/lir58/Desktop/Jierui/data'

"""foldername = 'Woman'
open1 = savefolder + '/' + foldername + '_dataset.pkl'
open2 = savefolder + '/' + foldername + '_labelset.pkl'
with open(open1, 'rb') as file:
    template = pickle.load(file)
    target = pickle.load(file)
    target_0 = pickle.load(file)
with open(open2, 'rb') as file:
    label = pickle.load(file)
    label_0 = pickle.load(file)

template = np.vstack((template, template))
target = np.vstack((target, target_0))
label = np.vstack((label, label_0))

dataset1 = savefolder + '/' + 'dataset1.pkl'
label1 = savefolder +  '/' +'label1.pkl'
with open(dataset1, 'wb') as file:
    pickle.dump(template, file)
    pickle.dump(target, file)
with open(label1, 'wb') as file:
    pickle.dump(label, file)"""


dataset1 = savefolder + '/' + 'dataset1.pkl'
label1 = savefolder +  '/' +'label1.pkl'
with open(dataset1, 'rb') as file:
    template = pickle.load(file)
    target = pickle.load(file)
with open(label1, 'rb') as file:
    label = pickle.load(file)

template0 = np.array(template)
target0 = np.array(target)
label0 = np.array(label)

foldername = 'Skiing'
open1 = savefolder + '/' + foldername + '_dataset.pkl'
open2 = savefolder + '/' + foldername + '_labelset.pkl'
with open(open1, 'rb') as file:
    template = pickle.load(file)
    target = pickle.load(file)
    target_0 = pickle.load(file)
with open(open2, 'rb') as file:
    label = pickle.load(file)
    label_0 = pickle.load(file)

template = np.vstack((template0, template, template))
target = np.vstack((target0, target, target_0))
label = np.vstack((label0, label, label_0))


dataset1 = savefolder + '/' + 'dataset1.pkl'
label1 = savefolder +  '/' +'label1.pkl'
with open(dataset1, 'wb') as file:
    pickle.dump(template, file)
    pickle.dump(target, file)
with open(label1, 'wb') as file:
    pickle.dump(label, file)