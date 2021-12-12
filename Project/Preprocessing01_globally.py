import os

#1 Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#2 Define different filepaths
global_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset'
otb50_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset/OTB50'

#3 Preprocess the datasets globally
"""
#The input datasets are the images
#The output datasets are the groundtruth bounding box positions listed in each line (4 number) in the ground truth file
#Firstly, figure out wheter the number of images in one folder is consistent to the number of lines in the txt file
#Secondly, ignore the images without groundtruth bounding boxes
"""
#3.1 Figure out the inconsistent folders
#3.1.1 Count the nubmer of images in each folder
def imagenumbers(imagepath):
    return len(os.listdir(imagepath))
#3.1.2 Count the number of lines in the txt files
def linenumbers(txtpath, file):
    os.chdir(txtpath)
    return len(open(file).readlines())
#3.1.3 Compare the number of images and lines, return the name of inconsitent folders (which means in these folders, some images do not have groundtruth bounding boxes)
def globalcomparison(file_path):
    inconsistent_list = []
    for fn in os.listdir(file_path): 
        print(fn)
        images_path = file_path + '/' + fn + '/img'
        txt_path = file_path + '/' + fn
        txt_name = 'groundtruth_rect.txt'
        if imagenumbers(images_path) != linenumbers(txt_path, txt_name):
            inconsistent_list.append(fn)
    return inconsistent_list
"""" #Show the name of the inconsistent folders
print(inconsistent_folder)
"""
#3.2 Delete images without groundtruth bounding boxes accourding to the dataset description in OTB website
import re
def delete_images(file_path, inconsistent_list):
    def imgname_to_int(img_name):   #Convert the names of images into numbers, thus, it will be easier to remove the images
        img_name = re.sub('.jpg', '', img_name)
        img_int = int(img_name)
        return img_int
    def deleteimg_onefoler(foldername, imagefolderpath, start, end):   #Delete the images in the folders
        start = int(start)
        end = int(end)
        if fn1 == foldername:
            for fn2 in os.listdir(imagefolderpath):
                fn2_number = imgname_to_int(fn2)
                deletedimg_path = images_path + '/' + fn2
                if start < fn2_number < end:
                    os.remove(deletedimg_path)
    for fn1 in os.listdir(file_path): 
        print(fn1)
        if fn1 in inconsistent_list:
            images_path = file_path + '/' + fn1 + '/img'
            deleteimg_onefoler('David', images_path, -1, 300)
            deleteimg_onefoler('Diving', images_path, 215, 2000)
            deleteimg_onefoler('Football1', images_path, 74, 2000)
            deleteimg_onefoler('Freeman3', images_path, 460, 2000)
            deleteimg_onefoler('Freeman4', images_path, 283, 2000)

folder = otb50_filepath #choose a dataset
inconsistent_folder = globalcomparison(folder)
delete_images(folder,inconsistent_folder)
#If it returns [], the process is done successfully 
print(inconsistent_folder)
#It will be easier to build the dataset now
