import pickle
import warnings
import numpy as np
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from keras.layers.core import Activation
from tensorflow.keras.models import Model
import tensorflow.keras.initializers as tfi
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Lambda, Dropout

#1 Ignore warnings
warnings.filterwarnings("ignore")

#2 Define different filepaths
global_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset'
otb50_filepath = 'C:/Users/lir58/Desktop/Jierui/Dataset/OTB50'

#3 Prepare dataset
#3.1 Load dataset
savefolder = 'C:/Users/lir58/Desktop/Jierui/data'
dataset1 = savefolder + '/' + 'dataset1.pkl'
label1 = savefolder +  '/' +'label1.pkl'
with open(dataset1, 'rb') as file:
    template = pickle.load(file)
    target = pickle.load(file)
with open(label1, 'rb') as file:
    label = pickle.load(file)

template = np.array(template)
target = np.array(target)
label = np.array(label)
print(template.shape, label.shape)

#3.2 Split dataset and proprocess dataset -> X-template  Z-target  y-template_bbox  y1-targe_tbbx
template = template/255.0
target = target/255.0
Z_train, Z_test, X_train, X_test,  y_train, y_test= train_test_split(template, target, label, test_size = 0.2, random_state = 42)

#4 Build model
class AlexNet:
    def __init__(self):

        self.initializer1 = tfi.VarianceScaling(scale=2, mode='fan_out',distribution='normal',seed=None)
        self.initializer2 = tfi.Ones()
        self.initializer3 = tfi.Zeros

        self.conv1 = Conv2D(48, (11,11), strides=(2,2), kernel_initializer=self.initializer1, bias_initializer=self.initializer3, name='conv1')
        self.batch1 = BatchNormalization(axis=3, gamma_initializer=self.initializer2, beta_initializer=self.initializer3, name='batch1')
        self.relu1 = Activation('relu')
        self.pool1 = MaxPooling2D((3,3), strides=(2,2), name='pool1')
        self.conv2 = Conv2D(128, (5,5), strides=(1,1), kernel_initializer=self.initializer1, bias_initializer=self.initializer3, name='conv2')
        self.batch2 = BatchNormalization(axis=3, gamma_initializer=self.initializer2, beta_initializer=self.initializer3, name='batch2')
        self.relu2 = Activation('relu')
        self.pool2 = MaxPooling2D((3,3), strides=(2,2), name='pool2')
        self.conv3 = Conv2D(96, (3,3), strides=(1,1), kernel_initializer=self.initializer1, bias_initializer=self.initializer3, name='conv3')
        self.batch3 = BatchNormalization(axis=3, gamma_initializer=self.initializer2, beta_initializer=self.initializer3, name='batch3')
        self.relu3 = Activation('relu')
        self.conv4 = Conv2D(96, (3,3), strides=(1,1), kernel_initializer=self.initializer1, bias_initializer=self.initializer3, name='conv4')
        self.batch4 = BatchNormalization(axis=3, gamma_initializer=self.initializer2, beta_initializer=self.initializer3, name='batch4')
        self.relu4 = Activation('relu')
        self.conv5 = Conv2D(128, (3,3), strides=(1,1), kernel_initializer=self.initializer1, bias_initializer=self.initializer3, name='conv5')
        self.dropout = Dropout(0.25)
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
        x = self.conv5(x)
        #x = self.dropout(x)
        return x

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
        #mean = K.mean(input,[0])
        #std = K.std(input,[0]) + 0.0001
        #output = (input-mean)/std
        #return output
        mean, variance = tf.nn.moments(input, [0])
        return tf.nn.batch_normalization(input, mean, variance, offset=0, scale=1, variance_epsilon=0.0001)
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

def loss_function(y, y1):
    y1 = y1*0.01
    #y1_max = K.max(y1)
    #y1 = y1/y1_max
    n_pos = tf.reduce_sum(tf.compat.v1.to_float(tf.equal(y[0], 1)))
    n_neg = tf.reduce_sum(tf.compat.v1.to_float(tf.equal(y[0], 0)))
    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg
    weights = tf.where(tf.equal(y, 1), w_pos * tf.ones_like(y), tf.ones_like(y))
    weights = tf.where(tf.equal(y, 0), w_neg * tf.ones_like(y), weights)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y1, labels=y)
    loss = loss*weights
    loss = tf.reduce_sum(loss, [1, 2])
    return loss

#5 Train model and save weight
target = (127, 127, 3)
template = (255, 255, 3)
epoch = 50
batchsize = 5
mom = 0.9
initlr = 6e-2
endlr = 6e-4
decaystep = 1
decayrate = 0.9 #0.8685113737513527 #np.power(endlr/initlr, 1/(epoch))
weight_decay = 5e-2
lr = tf.keras.optimizers.schedules.ExponentialDecay(initlr, decaystep, decayrate, staircase=True)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=3)
opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=mom, decay=weight_decay, nesterov=False)

model_build = SiameseFC(target, template)
model_build.compile(optimizer=opt, loss=loss_function, metrics=['accuracy'])
history = model_build.fit([Z_train, X_train], y_train, batch_size = batchsize, epochs = epoch, callbacks=[callback])
model_build.save_weights('trained_model')
test_loss, test_acc = model_build.evaluate([Z_test, X_test], y_test, verbose=2)
print('\nTest loss:', test_loss)
print('\nTest accuracy:', test_acc)

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

"""#5 Test: reconstructed model
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
        #x = self.batch1(x)
        #x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        #x = self.batch2(x)
        #x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        #x = self.batch3(x)
        #x = self.relu3(x)
        x = self.conv4(x)
        #x = self.batch4(x)
        #x = self.relu4(x)
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

model_recon = SiameseFC(target, template)
model_recon.build([target,template])
model_recon.load_weights('trained_model')
scoremap = model_recon.predict([Z_train[0:1], X_train[0:1]])
scoremap = np.reshape(scoremap,(17,17))
x = np.array(scoremap)
print(x.shape)
x_min = np.min(x)
x_sum = np.sum(x)
x_norm = (x-x_min)/x_sum
print(x_norm)
plt.matshow(x_norm)
plt.show()"""