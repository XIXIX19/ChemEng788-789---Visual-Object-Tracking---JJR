import os
import csv
import cv2
import random
import warnings
import numpy as np
import Preprocessing02_build_dataset as PBD

#1 Ignore warnings
warnings.filterwarnings("ignore")

#2 Define different filepaths
global_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset'
otb50_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset/OTB50'

#3 Build the training set - save the images into folders and save the bounding boxes into csv files
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
            new_frame_I = imagepath + + '/0' + str(x) + '.jpg'  
        if x > 10000:
            new_frame_I = imagepath + + '/' + str(x) + '.jpg'
    return new_frame_I
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
def template_info(templatebbox):
    x = templatebbox[0]
    y = templatebbox[1]
    w = templatebbox[2]
    h = templatebbox[3]
    p = (w+h)/2
    templatesize = np.sqrt((w+p)*(h+p))
    templatecenter = [int(x+w/2), int(y+h/2)]
    templatecenter = np.array(templatecenter)
    return templatesize, templatecenter
def target_info(templatebbox, targetbbox, templateoutsize, targetoutsize):
    x = targetbbox[0]
    y = targetbbox[1]
    w = targetbbox[2]
    h = targetbbox[3]
    p = (w+h)/2
    targetcenter = [int(x+w/2), int(y+h/2)]
    targetcenter = np.array(targetcenter)
    #templatesize = template_info(templatebbox)[0]
    targetsize = np.sqrt((w+p)*(h+p))
    targetsize = float(targetsize) * float((targetoutsize/templateoutsize))
    return targetsize, targetcenter
def avg_padding(img):
    border = img.mean(1).mean(0)
    return border
def crop_and_resize(img, center, size, out_size, border_value, border_type=cv2.BORDER_CONSTANT, interp=cv2.INTER_LINEAR):
    """
    to crop the img and resize + padding
    """
    size = np.round(size)
    corners = np.concatenate((np.round(center-size/2), np.round(center+size/2)))
    corners = np.round(corners).astype(int)
    pads = np.concatenate((-corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(img, npad, npad, npad, npad,border_type, value=border_value)
    corners = (corners + npad).astype(int)
    patch = img[corners[1]:corners[3], corners[0]:corners[2]]
    patch = cv2.resize(patch, (out_size, out_size),interpolation=interp)
    return patch
def training_set(file_path): #create the training set
    for fn in os.listdir(file_path): 
        imagepath = file_path + '/' + fn + '/img'

        #Create csv file to record template bbox, folders to store cropped template, target, target_0 imgs
        #target_0 -> imgs without object
        savefilename =  file_path + '/' + fn + '/' + 'raw_bbox_template.csv'
        f = open(savefilename,'w',encoding='utf-8',newline='' "")
        csv_writer = csv.writer(f)
        template_path = file_path + '/' + fn + '/' + 'template'
        if os.path.isdir(template_path) == False:
            os.mkdir(template_path)
        target_path = file_path + '/' + fn + '/' + 'target'
        if os.path.isdir(target_path) == False:
            os.mkdir(target_path)
        target_path_0 = file_path + '/' + fn + '/' + 'target_0'
        if os.path.isdir(target_path_0) == False:
            os.mkdir(target_path_0)

        #Figureout the name_id of the first img
        for fn1 in os.listdir(imagepath):
            filename1 = fn1
            break
        start = name_to_number(fn1)
        image_number = len(os.listdir(imagepath)) + start - 1

        #Create and store the template, template bbox, target, target_0 imgs
        for fn2 in os.listdir(imagepath):
            print(fn+fn2) #to know the progress
            number = name_to_number(fn2)
            x = random.randint(random_range(number, image_number)[0], random_range(number, image_number)[1])
            #template
            template_frame_I = number_to_path(x, imagepath, filename1)
            template_frame_bounding_box = PBD.y_dataset(file_path, fn)[x-start]
            template_img = cv2.imread(template_frame_I)
            template_size, template_center = template_info(template_frame_bounding_box)
            template_border = avg_padding(template_img)
            #target
            target_frame_I = imagepath + '/' + fn2
            target_frame_bounding_box = PBD.y_dataset(file_path, fn)[number-start]
            target_img = cv2.imread(target_frame_I)
            target_size, target_center = target_info(template_frame_bounding_box,target_frame_bounding_box, 127, 255)
            target_border = avg_padding(target_img)
            #target_0
            w = target_frame_bounding_box[2]
            h = target_frame_bounding_box[3]
            disp = [3*w, 3*h, 0, 0]
            disp = np.array(disp)
            target_frame_0_bounding_box = PBD.y_dataset(file_path, fn)[number-start] - disp
            target_size_0, target_center_0 = target_info(template_frame_bounding_box,target_frame_0_bounding_box, 127, 255)
            target_border = avg_padding(target_img)
            #generate imgs
            template_region = crop_and_resize(template_img, template_center, template_size, 127, template_border)
            search_region = crop_and_resize(target_img, target_center, target_size, 255, target_border)
            search_region_0 = crop_and_resize(target_img, target_center_0, target_size_0, 255, target_border)
            #save data
            csv_writer.writerow(template_frame_bounding_box)
            cv2.imwrite(template_path + "/" + fn + fn2 + ".jpg", template_region)
            cv2.imwrite(target_path + "/" + fn + fn2 + ".jpg", search_region)
            cv2.imwrite(target_path_0 + "/" + fn + fn2 + ".jpg", search_region_0)

#4 Run and Create
#training_set(otb50_filepath)
