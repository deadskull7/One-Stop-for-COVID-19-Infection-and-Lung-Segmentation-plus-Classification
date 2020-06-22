	#!/usr/bin/env python
	# coding: utf-8

	# In[ ]:

def runner_classification():

	get_ipython().system('pip install imgaug')                     # for image augmentation')


	# In[ ]:


	get_ipython().system('pip install -U segmentation-models')    # ONLY used for dice metric and IOU metric computation, models are made from scratch')


	# In[ ]:

	get_ipython().system('pip install plot-metric')
	                 # ONLY used for ROC curve, confusion matrix plotting


	# In[1]:


	import glob
	import pandas  as pd
	import numpy   as np
	import nibabel as nib
	import matplotlib.pyplot as plt
	import tensorflow as tf
	from tensorflow.keras import datasets, layers, models
	from zipfile import ZipFile
	from shutil import copyfile, copyfileobj
	import gzip
	from IPython.display import clear_output
	import cv2
	import os
	from pylab import rcParams
	import PIL
	from PIL import Image
	import scipy
	from google.colab import files
	from sklearn.model_selection import train_test_split
	from google.colab import drive
	from sklearn.decomposition import PCA
	from sklearn.cluster import KMeans, MeanShift
	import imgaug as ia
	import imgaug.augmenters as iaa
	from sklearn.utils import class_weight

	print("Version: ", tf.__version__)
	print("Eager mode: ", tf.executing_eagerly())
	print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


	# In[ ]:


	import sys
	import random
	import warnings

	import math
	import seaborn as sns; sns.set()
	from keras.callbacks import Callback
	from keras.losses import binary_crossentropy
	from tqdm import tqdm_notebook, tnrange
	from itertools import chain
	from skimage.io import imread, imshow, concatenate_images
	from skimage.transform import resize
	from skimage.morphology import label

	from keras.models import Model, load_model
	from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
	from keras.layers.core import Lambda, RepeatVector, Reshape
	from keras.layers.convolutional import Conv2D, Conv2DTranspose
	from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
	from keras.layers.merge import concatenate, add
	from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
	from keras.optimizers import Adam
	from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
	from keras import backend as K
	import joblib
	import gc
	import segmentation_models as sm


	# * Setting up environment to connect kaggle and colab

	# In[ ]:


	os.environ['KAGGLE_USERNAME'] = "deadskull7" # username from the json file
	os.environ['KAGGLE_KEY'] = "396fa22ac6546fab343d3b7f3a8e547b" # key from the json file


	# * kaggle dataset api

	# In[3]:


	get_ipython().system('kaggle datasets download -d andrewmvd/covid19-ct-scans')


	# In[4]:


	os.listdir()


	# * Extracting zip file here

	# In[ ]:


	#copyfile("/content/drive/My Drive/covid19-ct-scans.zip","/content/covid19-ct-scans.zip")
	with ZipFile('covid19-ct-scans.zip', 'r') as zipObj:
	   # Extract all the contents of zip file in current directory
	   zipObj.extractall('covid19-ct-scans')


	# In[6]:


	# Read and examine metadata
	raw_data = pd.read_csv('/content/covid19-ct-scans/metadata.csv')
	raw_data = raw_data.replace('../input/covid19-ct-scans/','/content/covid19-ct-scans/',regex=True)
	raw_data.head(5)


	# In[7]:


	raw_data.shape


	# * img_size is the preferred image size to which the image is to be resized

	# In[ ]:


	img_size = 512


	# * Used (CLAHE) Contrast Limited Adaptive Histogram Equalization to enhance the contrast of the images since medical images suffer a lot from the contrast problems.
	# * Here Clip limit and the grid size are the hyperparameters to tune, generally clip limit should be between 2 to 4 as higher clip limit won't clip most of the histogram and treat it as AHE.
	# * Higher clip limit might not even prevent the image from overamplifying the noise (the core advantage the CLAHE gives over AHE)

	# * demo is a boolean variable, if 1 then giving the ability to plot the before and image after enhancement along with their respective histogram plots.

	# In[ ]:


	def clahe_enhancer(test_img, demo):

	  test_img = test_img*255
	  test_img = np.uint8(test_img)
	  test_img_flattened = test_img.flatten()
	  
	  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	  clahe_image = clahe.apply(test_img)
	  clahe_image_flattened = clahe_image.flatten()

	  if demo == 1:

	    fig = plt.figure()
	    rcParams['figure.figsize'] = 10,10
	    
	    plt.subplot(2, 2, 1)
	    plt.imshow(test_img, cmap='bone')
	    plt.title("Original CT-Scan")

	    plt.subplot(2, 2, 2)
	    plt.hist(test_img_flattened)
	    plt.title("Histogram of Original CT-Scan")

	    plt.subplot(2, 2, 3)
	    plt.imshow(clahe_image, cmap='bone')
	    plt.title("CLAHE Enhanced CT-Scan")

	    plt.subplot(2, 2, 4)
	    plt.hist(clahe_image_flattened)
	    plt.title("Histogram of CLAHE Enhanced CT-Scan")

	  return(clahe_image)


	# * The images possess a lot of black space containing nothing and parts in which we are not interested like diaphragm below lungs. They will take the valuable RAM and also the unnecessary convolutions (computing power)
	# * A possible solution is cropping the images and taking out the Region Of Interest (ROI) as per problem statement and use case.
	# * Other pros are that we can now have greater area of ROI in the same resolution.
	# * For cropping, though we can slice the rows and columns using trial and error but that process would be limited to this dataset only. 
	# * If I input the same images with the lungs translated vertically a bit, the above technique would fail miserably and would rather crop the lungs too.
	# * So, a more better approach is to draw contours over the image and then cropping out the rectangle with the biggest contour with the largest area. 
	# * The contour (largest closed boundary) with the largest area would be the contour covering both the lungs.
	# * This will require thresholding as a prior step. After plotting the histogram of the images, the images were found near bimodal i.e. roughly two peaks. So, used the Otsu's method of binarization
	# * In global thresholding, we can use an arbitrary chosen value as a threshold. In contrast, Otsu's method avoids having to choose a value and determines it automatically and returns it.

	# * Another important point is while cropping a CT scan we have to make sure that its corresponding segmentation map is also cropped by same limits otherwise pixel level labeling will go wrong and the model might map a good area with an infectious area and vice versa which we don't want.
	# * So, in order to preserve the 4 end points, I am using a list by the name "points" for the same.

	# In[ ]:


	def cropper(test_img, demo):

	  test_img = test_img*255
	  test_img = np.uint8(test_img)

	  # ret, thresh = cv2.threshold(test_img, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
	  # ret, thresh = cv2.threshold(test_img, ret, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

	  contours,hierarchy = cv2.findContours(test_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	  areas = [cv2.contourArea(c) for c in contours]

	  x = np.argsort(areas)

	  max_index = x[x.size - 1]
	  cnt1=contours[max_index]
	  second_max_index = x[x.size - 2]
	  cnt2 = contours[second_max_index]

	  # max_index = np.argmax(areas)
	  # cnt=contours[max_index]

	  x,y,w,h = cv2.boundingRect(cnt1)
	  p,q,r,s = cv2.boundingRect(cnt2)

	  cropped1 = test_img[y:y+h, x:x+w]
	  cropped1 = cv2.resize(cropped1, dsize=(125,250), interpolation=cv2.INTER_AREA)
	  cropped2 = test_img[q:q+s, p:p+r]
	  cropped2 = cv2.resize(cropped2, dsize=(125,250), interpolation=cv2.INTER_AREA)

	  fused = np.concatenate((cropped1, cropped2), axis=1)

	  # super_cropped = test_img[y+7:y+h-20, x+25:x+w-25]
	  points_lung1 = []
	  points_lung2 = []

	  points_lung1.append(x); points_lung1.append(y); points_lung1.append(w); points_lung1.append(h)
	  points_lung2.append(p); points_lung2.append(q); points_lung2.append(r); points_lung2.append(s)
	  
	  if demo == 1:

	    fig = plt.figure()
	    rcParams['figure.figsize'] = 35, 35

	    plt.subplot(1, 3, 1)
	    plt.imshow(test_img, cmap='bone')
	    plt.title("Original CT-Scan")

	    plt.subplot(1, 3, 2)
	    plt.imshow(thresh, cmap='bone')
	    plt.title("Binary Mask")

	    plt.subplot(1, 3, 3)
	    plt.imshow(fused, cmap='bone')
	    plt.title("Cropped CT scan after making bounding rectangle")

	    # plt.subplot(1, 4, 4)
	    # plt.imshow(super_cropped, cmap='bone')
	    # plt.title("Cropped further manually")

	    plt.show()

	  return(fused, points_lung1, points_lung2)


	# * Dataset contain 20 files of .nii type, though each file contained multiple channels or silces each as a separate gray scale image.
	# * Total slices are 3520. These have been sliced out by 20% in the front and by 20% in the last of each file since in general these didn't had any infection masks and some didn't had the lungs, removed as noise.
	# * Also, images had pixel values from -998 to 1000+. Did min-max scaling.

	# In[ ]:


	def read_nii_demo(filepath, data):
	    '''
	    Reads .nii file and returns pixel array
	    '''
	    ct_scan = nib.load(filepath)
	    array   = ct_scan.get_fdata()
	    array   = np.rot90(np.array(array))
	    slices = array.shape[2]
	    array = array[:,:,round(slices*0.2):round(slices*0.8)]
	    array = np.reshape(np.rollaxis(array, 2),(array.shape[2],array.shape[0],array.shape[1], 1))

	    for img_no in range(0, array.shape[0]):
	        # array = Image.resize(array[...,img_no], (img_size,img_size))
	        img = cv2.resize(array[img_no], dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
	        xmax, xmin = img.max(), img.min()
	        img = (img - xmin)/(xmax - xmin)
	        data.append(img)


	# In[ ]:


	all_points1 = []
	all_points2 = []


	# In[ ]:


	def read_nii(filepath, data, string):
	    '''
	    Reads .nii file and returns pixel array

	    '''
	    global all_points1
	    global all_points2
	    ct_scan = nib.load(filepath)
	    array   = ct_scan.get_fdata()
	    array   = np.rot90(np.array(array))
	    slices = array.shape[2]
	    array = array[:,:,round(slices*0.2):round(slices*0.8)]
	    array = np.reshape(np.rollaxis(array, 2),(array.shape[2],array.shape[0],array.shape[1],1))
	    #print(array.shape[2])
	    #array = skimage.transform.resize(array, (array.shape[2], img_size, img_size))
	    #array = cv2.resize(array, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
	    

	    if string == "lungs":
	      all_points1 = []
	      all_points2 = []

	    for img_no in range(0, array.shape[0]):
	        if string == 'lungs' and np.unique(array[img_no]).size == 1:
	          continue
	        img = cv2.resize(array[img_no], dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
	        xmax, xmin = img.max(), img.min()
	        img = (img - xmin)/(xmax - xmin)

	        if string == 'lungs':
	          # img = np.uint8(img*255) 
	          img[img>0]=1
	          img, points1, points2 = cropper(img, demo = 0)
	          all_points1.append((points1[0], points1[1], points1[2], points1[3]))
	          all_points2.append((points2[0], points2[1], points2[2], points2[3]))
	          continue 

	        if string == "cts" and img_no < len(all_points1):
	          img = clahe_enhancer(img, demo = 0)
	          # img, points1, points2 = cropper(img, demo = 0)
	          # all_points1.append((points1[0], points1[1], points1[2], points1[3]))
	          # all_points2.append((points2[0], points2[1], points2[2], points2[3]))   
	          a,b,c,d = all_points1[img_no]
	          e,f,g,h = all_points2[img_no]
	          img1 = img[b:b+d, a:a+c]
	          img1 = cv2.resize(img1, dsize=(125,250), interpolation=cv2.INTER_AREA)
	          img2 = img[f:f+h, e:e+g]
	          img2 = cv2.resize(img2, dsize=(125,250), interpolation=cv2.INTER_AREA)
	          img = np.concatenate((img1, img2), axis=1)    

	        if string == "infections" and img_no < len(all_points1):
	          a,b,c,d = all_points1[img_no]
	          e,f,g,h = all_points2[img_no]
	          img = np.uint8(img*255)
	          img1 = img[b:b+d, a:a+c]
	          img1 = cv2.resize(img1, dsize=(125,250), interpolation=cv2.INTER_AREA)
	          img2 = img[f:f+h, e:e+g]
	          img2 = cv2.resize(img2, dsize=(125,250), interpolation=cv2.INTER_AREA)
	          img = np.concatenate((img1, img2), axis=1)


	        # img = cv2.resize(img, dsize=(192, 192), interpolation=cv2.INTER_LINEAR)
	        # img = img/255
	        #  remember to normalize again
	        # also resize images and masks for all
	        
	        data.append(img)


	# In[ ]:


	cts = []
	lungs = []
	infections = []


	# In[15]:


	for i in range(0, 20):
	    read_nii(raw_data.loc[i,'lung_mask'], lungs, 'lungs')
	    read_nii(raw_data.loc[i,'ct_scan'], cts, 'cts') 
	    read_nii(raw_data.loc[i,'infection_mask'], infections, 'infections')


	# * See the following fully processed sample

	# In[16]:


	x = 120

	rcParams['figure.figsize'] = 7, 7

	plt.imshow(cts[x], cmap='bone')
	plt.title("Final preprocessed (CLAHE Enhanced + Cropped) Image")

	print(infections[x].shape, cts[x].shape)


	# * Also, figuring out that 498 slices were of complete black masks i.e. no infection. For right now, kept out as didn't want to bother the segmentation model with this. 
	# * A better approach would be to make a sub-task and use that sub-task's output as input in our main task. Train a separate binary classifier to classify the CT scan as complete black or not, then the not ones to be passed from the segmentation model trained in this notebook.

	# In[ ]:


	#new
	y_label = []
	for i in range(0, len(infections)):
	  if len(np.unique(infections[i]))!=1:
	    y_label.append(1)
	  else:
	    y_label.append(0)


	# In[18]:


	#new
	print(y_label.count(0), y_label.count(1))


	# * Following is the demo for the CLAHE enhanced images with histograms.
	# * Notice how the seahorse shaped infection in the left lung can be distinguised clearly after enhancement.

	# In[19]:


	test_file = []
	read_nii_demo(raw_data.loc[0,'ct_scan'], test_file)
	test_file = np.array(test_file)
	rcParams['figure.figsize'] = 10, 10
	clahe_image = clahe_enhancer(test_file[155], demo = 1)


	# * A demo for the cropped images, notice how the unwanted part including the diaphragm got cut

	# In[ ]:


	# test_mask = []
	# read_nii_demo(raw_data.loc[0,'infection_mask'], test_mask)
	# test_mask = np.array(test_mask)
	# test_mask = np.uint8(test_mask*255)
	# rcParams['figure.figsize'] = 10,10
	# plt.imshow(test_mask[120][20:155, 4:217], cmap = 'bone')
	# test_mask[120][20:155, 4:217].shape


	# * Finally 1614 samples which will later be split into train and test

	# In[21]:


	print(len(cts) , len(infections))


	# * Also, since we copped the images and masks, all cannot be of same size, so again a dimension needs to be decided to which all could be resized to. 
	# * Used median of the all width and height but couldn't fit the RAM so reduced the size to 224, though images with larger resolution with more clear features will possibly give better results on the same model.

	# In[22]:


	dim1=[]
	dim2=[]
	for i in range(0, len(cts)):
	  dim1.append(cts[i].shape[0])
	  dim2.append(cts[i].shape[1])
	dim1 = np.array(dim1)
	dim2 = np.array(dim2)

	print("An idea about the new net dimension to which all must be resized to (some will increase and some decrease) --->", np.median(dim1),'x', np.median(dim2))


	# In[ ]:


	# 32*11 = 352


	# In[ ]:


	new_dim = 224


	# In[ ]:


	for i in range(0,len(cts)):
	  cts[i] = cv2.resize(cts[i], dsize=(new_dim, new_dim), interpolation=cv2.INTER_LINEAR)


	# In[ ]:


	# for i in range(0, len(cts)):
	#   cts[i] = cv2.cvtColor(cts[i], cv2.COLOR_GRAY2RGB)


	# In[27]:


	cts = np.array(cts)
	cts = cts.reshape(len(cts), new_dim,new_dim,1)
	y_label = np.array(y_label)
	print(cts.shape, y_label.shape)


	# In[ ]:


	# from keras.utils.np_utils import to_categorical
	# y_label = to_categorical(y_label, 2)


	# * Saving the numpy arrays to later reuse the same preprocessing for other models rather than doing it again and again.

	# * Using joblib which is way faster to dump and load than numpy and other techniques. Refer to the benchmarks:
	#  http://gael-varoquaux.info/programming/new_low-overhead_persistence_in_joblib_for_big_data.html

	# In[ ]:


	# import joblib
	# joblib.dump(cts, 'cts_192.pkl')


	# In[ ]:





	# In[ ]:





	# In[ ]:


	# No Augmentation added this time


	# * Data augmentation pipeline

	# In[ ]:


	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	seq = iaa.Sequential([
	    iaa.Fliplr(0.5), # horizontally flip 50% of all images
	    iaa.Flipud(0.2), # vertically flip 20% of all images
	    sometimes(iaa.Affine(
	            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
	            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
	            rotate=(-45, 45), # rotate by -45 to +45 degrees
	            shear=(-16, 16), # shear by -16 to +16 degrees
	            # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
	        ))
	], random_order=True)


	# In[ ]:


	no_of_aug_imgs = 100
	random_indices = np.random.randint(0, cts.shape[0], size=no_of_aug_imgs)
	sample_cts = cts[random_indices]
	sample_inf = infections[random_indices]


	# In[ ]:


	cts_aug, infections_aug = seq(images=sample_cts, segmentation_maps=sample_inf)


	# In[ ]:


	rcParams['figure.figsize'] = 60,60
	rand = np.random.randint(0, no_of_aug_imgs, size=8)

	cells1 = cts_aug[rand]
	grid_image1 = np.hstack(cells1)
	plt.imshow(grid_image1, cmap = 'bone')


	# In[ ]:


	cells2 = infections_aug[rand]
	grid_image2 = np.hstack(cells2)
	plt.imshow(grid_image2, cmap = 'bone')


	# In[ ]:


	print(cts_aug.shape, infections_aug.shape)


	# In[ ]:


	cts = np.concatenate((cts, cts_aug), axis=0)
	infections = np.concatenate((infections, infections_aug), axis = 0)
	np.random.shuffle(cts)
	np.random.shuffle(infections)
	print(cts.shape, infections.shape)


	# In[ ]:





	# In[ ]:





	# * Normalizing images and masks from 0 to 1

	# In[ ]:


	cts = cts/255


	# * Just overlaying infection masks over the corresponding CT scans

	# In[ ]:


	sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
	for train_index, test_index in sss.split(cts, y_label):
	  x_train, x_valid = cts[train_index], cts[test_index]
	  y_train, y_valid = y_label[train_index], y_label[test_index]
	# x_train, x_valid, y_train, y_valid = StratifiedShuffleSplit(cts, y_label, test_size=0.3, random_state=42, stratify=True)


	# In[30]:


	np.unique(y_label, return_counts=True)


	# In[31]:


	np.unique(y_train, return_counts=True)


	# In[32]:


	np.unique(y_valid, return_counts=True)


	# In[33]:


	print(x_train.shape, x_valid.shape)
	print(y_train.shape, y_valid.shape)


	# In[ ]:





	# In[ ]:


	def recall(y_true, y_pred):
	    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	    recall = true_positives / (possible_positives + K.epsilon())
	    return recall

	def precision(y_true, y_pred):
	    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	    precision = true_positives / (predicted_positives + K.epsilon())
	    return precision

	def f1(y_true, y_pred):
	    precision_t = precision(y_true, y_pred)
	    recall_t = recall(y_true, y_pred)
	    return 2*((precision_t*recall_t)/(precision_t+recall_t+K.epsilon()))


	class RocCallback(Callback):
	    def __init__(self,training_data,validation_data):
	        self.x = training_data[0]
	        self.y = training_data[1]
	        self.x_val = validation_data[0]
	        self.y_val = validation_data[1]


	    def on_train_begin(self, logs={}):
	        return

	    def on_train_end(self, logs={}):
	        return

	    def on_epoch_begin(self, epoch, logs={}):
	        return

	    def on_epoch_end(self, epoch, logs={}):
	        global best_val_auc
	        global model
	        y_pred_train = self.model.predict_proba(self.x)
	        roc_train = roc_auc_score(self.y, y_pred_train)
	        y_pred_val = self.model.predict_proba(self.x_val)
	        roc_val = roc_auc_score(self.y_val, y_pred_val)
	        print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
	        if best_val_auc < roc_val:
	          best_val_auc = roc_val
	          model.save_weights('best_val_auc_weights.h5')
	          print("Saving best validation AUC weights")
	        return

	    def on_batch_begin(self, batch, logs={}):
	        return

	    def on_batch_end(self, batch, logs={}):
	        return


	# In[36]:


	model = Sequential()
	model.add(Conv2D(16, (3, 3), activation='relu', padding="same", kernel_initializer="he_normal", input_shape=(new_dim,new_dim,1)))
	model.add(BatchNormalization())
	model.add(Conv2D(16, (3, 3), padding="same", activation='relu' , kernel_initializer="he_normal"))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3), activation='relu', padding="same" , kernel_initializer="he_normal"))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), padding="same", activation='relu' , kernel_initializer="he_normal"))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3), activation='relu', padding="same" , kernel_initializer="he_normal"))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), padding="same", activation='relu' , kernel_initializer="he_normal"))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2))) 

	# model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
	# model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))

	# model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
	# model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())

	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.4))             
	model.add(Dense(1 , activation='sigmoid'))      

	print(model.summary())


	# In[ ]:


	# import math
	# def step_decay(epoch):
	#     initial_lrate = 0.0008
	#     drop = 0.8
	#     epochs_drop = 10
	#     lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	#     print('New learning rate', lrate)
	#     return lrate

	# lrate = LearningRateScheduler(step_decay)


	# In[38]:


	weights = class_weight.compute_class_weight('balanced',
	                                            np.unique(y_train),
	                                            y_train)
	print(weights)


	# In[ ]:


	batch_size = 32
	epochs = 25
	best_val_auc = -1
	roc = RocCallback(training_data=(x_train, y_train),
	                  validation_data=(x_valid, y_valid))

	#lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=7, verbose=1)
	# filepath_auc="covid_weights_aucroc.hdf5"
	filepath_loss = "covid_weights_val_loss.hdf5"
	# filepath_acc = "covid_weights_val_acc.hdf5"
	# checkpoint_auc = ModelCheckpoint(filepath_auc, monitor='roc-auc_val', verbose=1, save_best_only=True, mode='max')
	checkpoint_loss = ModelCheckpoint(filepath_loss, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	# checkpoint_acc = ModelCheckpoint(filepath_acc, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


	# In[ ]:


	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005), metrics=[f1])


	# In[ ]:


	results = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
	                    validation_data=(x_valid, y_valid) ,
	                    callbacks = [roc, checkpoint_loss], class_weight=weights)


	# * Model saved in json format and its weight in .hdf5 format at local

	# In[ ]:


	gc.collect()


	# In[ ]:


	# model.load_weights(filepath_loss)
	model.load_weights("best_val_auc_weights.h5")


	# In[ ]:


	model.save_weights('best_val_auc_weights.h5')


	# In[ ]:


	files.download('best_val_auc_weights.h5')


	# In[ ]:


	model_json = model.to_json()
	with open("best_val_auc_weights.json","w") as json_file:
	     json_file.write(model_json)

	files.download("best_val_auc_weights.json")


	# In[ ]:


	print("Best saved AUCROC on validation set :" , best_val_auc)


	# In[ ]:


	score = model.evaluate(x_valid, y_valid, batch_size=32)
	print("test loss:" , score[0], "\ntest f1 score:" , score[1])


	# In[ ]:


	rcParams['figure.figsize'] = 10,7
	plt.grid('True')
	plt.plot(results.history['loss'], color='m')
	plt.plot(results.history['val_loss'], color='k')
	plt.title('Loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()


	# In[42]:


	drive.mount('/content/drive')


	# In[ ]:


	# model.load_weights(filepath_dice_coeff)
	model.load_weights('/content/drive/My Drive/cts and infections/best_val_auc_weights.h5')


	# In[ ]:


	predictions = model.predict(x_valid)
	predictions = np.array(predictions.flatten())


	# In[ ]:


	bc1 = BinaryClassification(y_valid, predictions, labels=["Class 0", "Class 1"], threshold = 0.50)


	# In[53]:


	rcParams['figure.figsize'] = 8,8
	bc1.plot_roc_curve()[3]


	# In[63]:


	bc1.plot_class_distribution(threshold = 0.5, pal_colors=['r','g','m','k'])


	# In[54]:


	a = bc1.plot_confusion_matrix()
	tn = a[0][0]
	fp = a[0][1]
	fn = a[1][0]
	tp = a[1][1]
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	print('Accuracy:', (tp+tn)/(tp+tn+fp+fn))
	print('Precision:', precision)
	print('Recall:', recall)
	print('F1 score:', 2*precision*recall/(precision+recall))


	# In[ ]:


	bc2 = BinaryClassification(y_valid, predictions, labels=["Class 0", "Class 1"], threshold = 0.81)


	# In[65]:


	bc2.plot_roc_curve()[3]


	# In[66]:


	bc2.plot_class_distribution(threshold = 0.81, pal_colors=['r','g','m','k'])


	# In[57]:


	b = bc2.plot_confusion_matrix()
	tn = b[0][0]
	fp = b[0][1]
	fn = b[1][0]
	tp = b[1][1]
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	print('Accuracy:', (tp+tn)/(tp+tn+fp+fn))
	print('Precision:', precision)
	print('Recall:', recall)
	print('F1 score:', 2*precision*recall/(precision+recall))


	# In[ ]:


	gc.collect()


	# In[ ]:
	return


if __name__ == "__main__":
	runner_classification()




	# In[ ]:




