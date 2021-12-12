import os
import csv
import cv2
import warnings
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers.core import Activation
from tensorflow.keras.models import Model
import tensorflow.keras.initializers as tfi
import Preprocessing02_build_dataset as PBD
from Datapreparing import crop_and_resize, avg_padding, template_info, name_to_number
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Lambda

#1 Ignore warnings
warnings.filterwarnings("ignore")

#2 Define different filepaths
global_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset'
otb50_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset/OTB50'

#3 Model reconstruction
class AlexNet:
    def __init__(self):

        self.initializer1 = tfi.VarianceScaling(scale=1, mode='fan_out',distribution='normal',seed=None)
        self.initializer2 = tfi.Ones()
        self.initializer3 = tfi.Zeros

        self.conv1 = Conv2D(96, (11,11), strides=(2,2), kernel_initializer=self.initializer1, bias_initializer=self.initializer3, name='conv1')
        self.batch1 = BatchNormalization(axis=3, gamma_initializer=self.initializer2, beta_initializer=self.initializer3, name='batch1')
        self.relu1 = Activation('relu')
        self.pool1 = MaxPooling2D((3,3), strides=(2,2), name='pool1')
        self.conv2 = Conv2D(128, (5,5), strides=(1,1), kernel_initializer=self.initializer1, bias_initializer=self.initializer3, name='conv2')
        self.batch2 = BatchNormalization(axis=3, gamma_initializer=self.initializer2, beta_initializer=self.initializer3, name='batch2')
        self.relu2 = Activation('relu')
        self.pool2 = MaxPooling2D((3,3), strides=(2,2), name='pool2')
        self.conv3 = Conv2D(192, (3,3), strides=(1,1), kernel_initializer=self.initializer1, bias_initializer=self.initializer3, name='conv3')
        self.batch3 = BatchNormalization(axis=3, gamma_initializer=self.initializer2, beta_initializer=self.initializer3, name='batch3')
        self.relu3 = Activation('relu')
        self.conv4 = Conv2D(192, (3,3), strides=(1,1), kernel_initializer=self.initializer1, bias_initializer=self.initializer3, name='conv4')
        self.batch4 = BatchNormalization(axis=3, gamma_initializer=self.initializer2, beta_initializer=self.initializer3, name='batch4')
        self.relu4 = Activation('relu')
        self.conv5 = Conv2D(128, (3,3), strides=(1,1), kernel_initializer=self.initializer1, bias_initializer=self.initializer3, name='conv5')

    def model_build(self, input):
        x = input
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu4(x)
        output = self.conv5(x)
        return output

def score_layer():
    def correlation(input):
        target = input[0]
        template = input[1]
        #target = tf.reshape(target, target.shape.as_list()+[1])
        #template = tf.reshape(template, [1]+template.shape.as_list())
        target = tf.expand_dims(target,-1)
        template = tf.expand_dims(template, 0)
        return tf.nn.conv2d(template, target, padding = 'VALID',strides=(1,1))

    def score_map(input):
        return K.reshape(tf.map_fn(correlation, input, dtype=tf.float32, infer_shape=False), shape=(-1,17,17))
    
    return Lambda(score_map, output_shape=(17,17))

def normalization_layer():
    def normalization(input):
        mean = K.mean(input,[0])
        std = K.std(input,[0]) + 0.0001
        output = (input-mean)/std
        #mean, variance = tf.nn.moments(input, [0])
        return output#tf.nn.batch_normalization(input, mean, variance, offset=0, scale=1, variance_epsilon=0.0001)
    return Lambda(normalization, output_shape=(17,17)) 

def SiameseFC(inputshape1, inputshape2):
    alexnet = AlexNet()
    target_input = Input(shape=inputshape1)
    template_input = Input(shape=inputshape2)
    target_feature = alexnet.model_build(target_input)
    template_feature = alexnet.model_build(template_input)

    scoremap = score_layer()([target_feature, template_feature])
    #scoremap = normalization_layer()(scoremap)

    image_pair = [target_input, template_input]
    scoremap = [scoremap]

    model = Model(image_pair, scoremap)
    return model

target = (127, 127, 3)
template = (255, 255, 3)
model_recon = SiameseFC(target, template)
model_recon.build([target,template])
model_recon.load_weights('trained_model')  #drag all the files in the 'log_i_j%' folder to the pyhton default read file path

#4 Track the object
def last_frame_info(lastbbox):
    w = lastbbox[2]
    h = lastbbox[3]
    return w, h

def scale_coef():
    f = [np.power(1.0375,-2), np.power(1.0375,-0.5), np.power(1.0375,1)]
    f = np.array(f)
    f1 = f[0]
    f2 = f[1]
    f3 = f[2]
    return f1, f2, f3

def last_search_patch(target, last, lastbbox):
    f1, f2, f3 = scale_coef()
    size, center = template_info(lastbbox)
    border_last = avg_padding(last)
    border_target = avg_padding(target)
    last = crop_and_resize(last, center, size, 127, border_last,border_type=cv2.BORDER_CONSTANT, interp=cv2.INTER_LINEAR)
    target1 = crop_and_resize(target, center, size*(255/127)*f1, 255, border_target, border_type=cv2.BORDER_CONSTANT, interp=cv2.INTER_LINEAR)
    target2 = crop_and_resize(target, center, size*(255/127)*f2, 255, border_target, border_type=cv2.BORDER_CONSTANT, interp=cv2.INTER_LINEAR)
    target3 = crop_and_resize(target, center, size*(255/127)*f3, 255, border_target, border_type=cv2.BORDER_CONSTANT, interp=cv2.INTER_LINEAR)
    return target1, target2, target3, last

def get_score_map(target1, target2, target3, last): 
    target1 = np.expand_dims(target1, axis=0)
    target2 = np.expand_dims(target2, axis=0)
    target3 = np.expand_dims(target3, axis=0)
    last = np.expand_dims(last, axis=0)
    scoremap1 = model_recon.predict([last, target1])
    scoremap2 = model_recon.predict([last, target2])
    scoremap3 = model_recon.predict([last, target3])
    scoremap1 = np.squeeze(scoremap1, axis=0)
    scoremap2 = np.squeeze(scoremap2, axis=0)
    scoremap3 = np.squeeze(scoremap3, axis=0)
    scoremap1 = cv2.resize(scoremap1, (257,257), interpolation=cv2.INTER_CUBIC)*0.9745
    scoremap2 = cv2.resize(scoremap2, (257,257), interpolation=cv2.INTER_CUBIC)
    scoremap3 = cv2.resize(scoremap3, (257,257), interpolation=cv2.INTER_CUBIC)*0.9745
    return scoremap1, scoremap2, scoremap3

def process_scoremap(scoremap):
    max = np.amax(scoremap, axis=(0,1))
    return max

def track_box(scoremap1, scoremap2, scoremap3, lastbbox):
    init_w, init_h = last_frame_info(lastbbox)
    init_size, init_center = template_info(lastbbox)
    init_cx, init_cy = init_center[0], init_center[1]
    LR = 0.59
    f1, f2, f3 = scale_coef()
    influence = 0.176
    scoremaps = [scoremap1, scoremap2, scoremap3]
    scoremaps = np.array(scoremaps)
    f = [f1, f2, f3]
    f = np.array(f)
    finalsize = 16*16+1

    hann_window = np.expand_dims(np.hanning(finalsize), axis=0)
    penalty = np.transpose(hann_window)*hann_window
    penalty = penalty/np.sum(penalty)

    max1 = process_scoremap(scoremap1)
    max2 = process_scoremap(scoremap2)
    max3 = process_scoremap(scoremap3)
    maxs = np.array([max1,max2,max3])
    new_id = np.argmax(maxs)

    update_size = (1-LR)*init_size + LR*init_size*(255/127)*f[new_id]
    update_w = (1-LR)*init_w + LR*init_w*f[new_id]
    update_h = (1-LR)*init_h + LR*init_h*f[new_id]

    update_scoremap = scoremaps[new_id,:,:]
    update_scoremap = update_scoremap - np.min(update_scoremap)
    update_scoremap = update_scoremap / np.sum(update_scoremap)
    update_scoremap = (1-influence)*update_scoremap + influence*penalty
    p = np.asarray(np.unravel_index(np.argmax(update_scoremap), np.shape(update_scoremap)))
    c = float(finalsize - 1)/ 2
    disp_area = p - c
    disp_targetcrop = disp_area * float(8/16)
    disp_frame = disp_targetcrop * update_size/255
    update_cx = init_cx + disp_frame[1]
    update_cy = init_cy + disp_frame[0] 

    update_x = update_cx - update_w/2
    update_y = update_cy - update_h/2

    pred_box = [round(update_x), round(update_y), round(update_w), round(update_h)]
    pred_box = np.array(pred_box)
    return pred_box

def tracking(datasetname, foldername):
    targetpath = datasetname + '/' + foldername + '/' + 'img'
    predbbox_path = datasetname + '/' + foldername + '/' + 'pred_bbox.csv'
    f = open(predbbox_path,'w',encoding='utf-8',newline='' "")
    csv_writer2 = csv.writer(f)
    for fn in os.listdir(targetpath):
        firstnumber = name_to_number(fn)
        firstframepath = targetpath + '/' + fn
        firstframe =  cv2.imread(firstframepath)
        firstframe_boundingbox = PBD.y_dataset(datasetname, foldername)[0]
        csv_writer2.writerow(firstframe_boundingbox)
        break
    for fn in os.listdir(targetpath):
        number = name_to_number(fn)
        print(fn)
        if number != firstnumber:
            targetframepath = targetpath + '/' + fn
            targetframe = cv2.imread(targetframepath)
            target1, target2, target3, last = last_search_patch(targetframe, firstframe, firstframe_boundingbox)
            scoremap1, scoremap2, scoremap3 = get_score_map(target1, target2, target3, last)
            pred_target_boudingbox = track_box(scoremap1, scoremap2, scoremap3, firstframe_boundingbox)
            csv_writer2.writerow(pred_target_boudingbox)

            firstframe = targetframe
            firstframe_boundingbox = pred_target_boudingbox

tracking(otb50_filepath, 'Biker')