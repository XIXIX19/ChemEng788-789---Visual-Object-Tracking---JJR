import cv2
import numpy as np
import Preprocessing02_build_dataset as PBD
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch

#1 Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#2 Define different filepaths
global_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset'
otb50_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset/OTB50'

# Build the training set
def crop_object(X_image, y_boundingbox): #Crop the object out
    cropped_image = X_image[y_boundingbox[1]:y_boundingbox[1]+y_boundingbox[3]-1, y_boundingbox[0]:y_boundingbox[0]+y_boundingbox[2]-1]
    return cropped_image

def box(origin_c, size, max):
    new = [int(origin_c[0] - size/2), int(origin_c[1] - size/2), size, size]
    a = new[0]
    if a<0:
        new[0] = 1
        print('1')
    if a>= max[0]-size/2:
        new[0] = max[0]-size+1
        print('2')

    b = new[1]
    if b<0:
        new[1] = 1
        print('3')
    if b>= max[1]-size/2:
        new[1] = max[1]-size+1
        print('4')
    print(a,b,size/2,max)
    print('5', new)
    return new

def name_to_number (filename):
    number = int(filename[0])*1000 + int(filename[1])*100 + int(filename[2])*10 + int(filename[3])*1
    return number

def number_to_path(x, imagepath, filename1):
    global new_frame_I
    if len(filename1) == 8:
        if x <10:
            new_frame_I = imagepath + '/000' + str(x) + '.jpg'
        if x > 9 and x < 100:
            new_frame_I = imagepath + '/00' + str(x) + '.jpg'
        if x > 99 and x < 1000:
            new_frame_I = imagepath + '/0' + str(x) + '.jpg'
        if x > 1000:
            new_frame_I = imagepath + '/' + str(x) + '.jpg'
    if len(filename1) == 9:
        if x <10:
            new_frame_I = imagepath + '/0000' + str(x) + '.jpg'
        if x > 9 and x < 100:
            new_frame_I = imagepath + '/000' + str(x) + '.jpg'
        if x > 99 and x < 1000:
            new_frame_I = imagepath + '/00' + str(x) + '.jpg'
        if x > 1000 and x < 10000:
            new_frame_I = imagepath + '/0' + str(x) + '.jpg'  
        if x > 10000:
            new_frame_I = imagepath + '/' + str(x) + '.jpg'
    return new_frame_I

def find_center(bounding_box):
    center = [int(bounding_box[0] + 0.5*bounding_box[2]), int(bounding_box[1] + 0.5*bounding_box[3])]
    return center

def random_range(number, image_number):
    if number-50 <= number:
        start = number
    if number-50 >number:
        start = number-50
    if number+50 >= image_number:
        end = image_number
    if number+50 < image_number:
        end = number+50
    return start, end

def training_set_X(fn, file_path, template_path, target_path): #the first frame -> target patch(127*127), and search region(255*255)
    imagepath = file_path + '/' + fn + '/img'
    for fn1 in os.listdir(imagepath):
        filename1 = fn1
        break
    start = name_to_number(fn1)
    image_number = len(os.listdir(imagepath)) + start - 1

    for fn2 in os.listdir(imagepath):
        print(fn+fn2)
        number = name_to_number(fn2)
        x = random.randint(random_range(number, image_number)[0], random_range(number, image_number)[1])
        target_frame_I = imagepath + '/' + fn2
        template_frame_I = number_to_path(x, imagepath, filename1)

        target_frame_bounding_box = PBD.y_dataset(fn)[number-start]
        last_frame_bounding_box = PBD.y_dataset(fn)[x-start]
        center_of_target = find_center(target_frame_bounding_box)
        center_of_template = find_center(last_frame_bounding_box)
        target_img = Image.open(target_frame_I)
        template_img = Image.open(template_frame_I)
        img_size = target_img.size
        target_patch = box(center_of_target, int(max(target_frame_bounding_box[2], target_frame_bounding_box[3])*1.5), img_size)
        print(center_of_target, target_frame_bounding_box, target_patch)
        search_region = box(center_of_template, int(max(last_frame_bounding_box[2], last_frame_bounding_box[3])*1.8), img_size)

        if os.path.isdir(target_path) == False:
            os.mkdir(target_path)
        target_img = crop_object(cv2.imread(target_frame_I), target_patch)
        target_img = cv2.resize(target_img, (127, 127), interpolation = cv2.INTER_AREA)
        cv2.imwrite(target_path + "/" + fn + fn2 + ".jpg", target_img)

        
        if os.path.isdir(template_path) == False:
            os.mkdir(template_path)
        template_img = crop_object(cv2.imread(template_frame_I), search_region)
        template_img = cv2.resize(template_img, (255, 255), interpolation = cv2.INTER_AREA)
        cv2.imwrite(template_path + "/" + fn + fn2 + ".jpg", template_img)

template_path = global_filepath + '/template'
target_path = global_filepath + '/target'
training_set_X('CarScale',otb50_filepath, template_path, target_path)

