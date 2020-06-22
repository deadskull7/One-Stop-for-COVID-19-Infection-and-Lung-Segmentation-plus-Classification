	#!/usr/bin/env python
	# coding: utf-8

	# In[ ]:

def three_fold_runner_unet_infection_segmentation():

	get_ipython().system('pip install imgaug')                     # for image augmentation')


	# In[ ]:


	get_ipython().system('pip install -U segmentation-models')     # ONLY used for dice metric and IOU metric computation, models are made from scratch')


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


	# In[ ]:


	get_ipython().system('kaggle datasets download -d deadskull7/unetpp')


	# In[ ]:


	#copyfile("/content/drive/My Drive/covid19-ct-scans.zip","/content/covid19-ct-scans.zip")
	with ZipFile('unetpp.zip', 'r') as zipObj:
	   # Extract all the contents of zip file in current directory
	   zipObj.extractall('unetpp')


	# In[ ]:


	os.listdir()


	# * Extracting zip file here

	# In[ ]:


	#copyfile("/content/drive/My Drive/covid19-ct-scans.zip","/content/covid19-ct-scans.zip")
	with ZipFile('covid19-ct-scans.zip', 'r') as zipObj:
	   # Extract all the contents of zip file in current directory
	   zipObj.extractall('covid19-ct-scans')


	# In[ ]:


	# Read and examine metadata
	raw_data = pd.read_csv('/content/covid19-ct-scans/metadata.csv')
	raw_data = raw_data.replace('../input/covid19-ct-scans/','/content/covid19-ct-scans/',regex=True)
	raw_data.head(5)


	# In[ ]:


	raw_data.shape


	# * img_size is the preferred image size to which the image is to be resized

	# In[ ]:


	img_size = 512


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


	# In[ ]:


	for i in range(0, 20):
	    read_nii(raw_data.loc[i,'lung_mask'], lungs, 'lungs')
	    read_nii(raw_data.loc[i,'ct_scan'], cts, 'cts') 
	    read_nii(raw_data.loc[i,'infection_mask'], infections, 'infections')


	# * See the following fully processed sample

	# In[ ]:


	x = 60

	rcParams['figure.figsize'] = 10,10

	plt.subplot(1, 2, 1)
	plt.imshow(cts[x], cmap='bone')
	plt.title("Final preprocessed (CLAHE Enhanced + Cropped) Image")

	plt.subplot(1, 2, 2)
	plt.imshow(infections[x], cmap='bone')
	plt.title("Final preprocessed corresponding Mask")

	print(cts[x].shape, infections[x].shape)


	# In[ ]:


	no_masks = []
	for i in range(0, len(infections)):
	  if np.unique(infections[i]).size == 1:
	    no_masks.append(i)
	print("Number of complete black masks :" , len(no_masks))

	for index in sorted(no_masks, reverse = True):  
	    del infections[index]  
	    del cts[index]


	# * Following is the demo for the CLAHE enhanced images with histograms.
	# * Notice how the seahorse shaped infection in the left lung can be distinguised clearly after enhancement.

	# In[ ]:


	test_file = []
	read_nii_demo(raw_data.loc[0,'ct_scan'], test_file)
	test_file = np.array(test_file)
	rcParams['figure.figsize'] = 10, 10
	clahe_image = clahe_enhancer(test_file[60], demo = 1)


	# * A demo for the cropped images, notice how the unwanted part including the diaphragm got cut

	# In[ ]:


	# fig = plt.figure()
	# rcParams['figure.figsize'] = 35, 35

	# cropped_image, points1, points2 = cropper(test_file[120], demo = 1)
	# #print(ret)
	# print(points1, points2)


	# In[ ]:


	# test_mask = []
	# read_nii_demo(raw_data.loc[0,'infection_mask'], test_mask)
	# test_mask = np.array(test_mask)
	# test_mask = np.uint8(test_mask*255)
	# rcParams['figure.figsize'] = 10,10
	# plt.imshow(test_mask[120][20:155, 4:217], cmap = 'bone')
	# test_mask[120][20:155, 4:217].shape


	# * Finally 1614 samples which will later be split into train and test

	# In[ ]:


	print(len(cts) , len(infections))


	# In[ ]:


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
	  # cts[i] = cts[i]/255
	  infections[i] = cv2.resize(infections[i], dsize=(new_dim, new_dim), interpolation=cv2.INTER_LINEAR)
	  # infections[i] = infections[i]/255


	# In[ ]:


	# for i in range(0, len(cts)):
	#   cts[i] = cv2.cvtColor(cts[i], cv2.COLOR_GRAY2RGB)
	# for i in range(0, len(infections)):
	#   infections[i] = cv2.cvtColor(infections[i], cv2.COLOR_GRAY2RGB)


	# In[ ]:


	cts = np.array(cts)
	infections = np.array(infections)


	# * Saving the numpy arrays to later reuse the same preprocessing for other models rather than doing it again and again.

	# In[ ]:


	# cts = cts.reshape( len(cts), new_dim, new_dim)
	# infections = infections.reshape( len(infections), new_dim, new_dim)


	# In[ ]:


	cts = np.uint8(cts)
	infections = np.uint8(infections)


	# In[ ]:


	# No Augmentation added this time


	# In[ ]:





	# In[ ]:





	# * Data augmentation pipeline

	# In[ ]:


	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	seq = iaa.Sequential([
	    iaa.Fliplr(0.5), # horizontally flip 50% of all images
	    iaa.Flipud(0.2), # vertically flip 20% of all images
	    sometimes(iaa.Affine(
	            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
	            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
	            rotate=(-40, 40), # rotate by -45 to +45 degrees
	            shear=(-16, 16), # shear by -16 to +16 degrees
	            # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
	        ))
	], random_order=True)


	# In[ ]:


	no_of_aug_imgs = 200
	random_indices = np.random.randint(0, cts.shape[0], size=no_of_aug_imgs)
	sample_cts = cts[random_indices]
	sample_inf = infections[random_indices]


	# In[ ]:


	# sample_cts = np.floor(sample_cts)
	# sample_inf = np.floor(sample_inf)
	# sample_cts = np.uint64(sample_cts)
	# sample_inf = np.uint64(sample_inf)


	# In[ ]:


	cts_aug, infections_aug = seq(images=sample_cts, 
	                              segmentation_maps=sample_inf)


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


	# cts = np.concatenate((cts, cts_aug), axis=0)
	# infections = np.concatenate((infections, infections_aug), axis = 0)
	# np.random.shuffle(cts)
	# np.random.shuffle(infections)
	# print(cts.shape, infections.shape)


	# In[ ]:


	cts_aug = cts_aug/255
	infections_aug = infections_aug/255
	cts_aug = cts_aug.reshape(len(cts_aug), new_dim, new_dim, 1)
	infections_aug = infections_aug.reshape(len(infections_aug), new_dim, new_dim, 1)


	# In[ ]:





	# In[ ]:





	# * Normalizing images and masks from 0 to 1

	# In[ ]:


	joblib.dump(cts, 'cts_cropped_lungs_224.pkl')


	# In[1]:


	files.download('cts_cropped_lungs_224.pkl')


	# In[ ]:


	joblib.dump(infections, 'infections_cropped_lungs_224.pkl')


	# In[2]:


	files.download('infections_cropped_lungs_224.pkl')


	# In[ ]:


	# temp = joblib.load('infections_cropped_lungs_224.pkl')


	# In[ ]:





	# In[5]:


	drive.mount('/content/drive')


	# In[6]:


	cts = joblib.load('/content/drive/My Drive/cts and infections/cts_cropped_lungs_224.pkl')
	infections = joblib.load('/content/drive/My Drive/cts and infections/infections_cropped_lungs_224.pkl')
	print(cts.shape, infections.shape)


	# In[ ]:





	# In[ ]:


	cts = cts/255
	infections = infections/255


	# In[ ]:


	cts = cts.reshape(len(cts), new_dim, new_dim, 1)
	infections = infections.reshape(len(infections), new_dim, new_dim, 1)


	# In[ ]:


	# cts_new = []
	# # lungs_infections_new = []
	# infections_new = []


	# In[ ]:


	# for i in range(0, 2112):
	#   cts_new.append(np.array(cts[i]))
	#   # lungs_infections_new.append(np.array(lungs_infections[i]))
	#   infections_new.append(np.array(infections[i]))


	# In[ ]:


	# cts_new = np.array(cts_new)
	# # lungs_infections_new = np.array(lungs_infections_new)
	# infections_new = np.array(infections_new)


	# * Just overlaying infection masks over the corresponding CT scans

	# In[ ]:


	def plot_sample(array_list, color_map = 'nipy_spectral'):
	    '''
	    Plots and a slice with all available annotations
	    '''
	    fig = plt.figure(figsize=(10,30))

	    plt.subplot(1,2,1)
	    plt.imshow(array_list[0].reshape(new_dim, new_dim), cmap='bone')
	    plt.title('Original Image')

	    # plt.subplot(1,2,2)
	    # plt.imshow(array_list[0], cmap='bone')
	    # plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
	    # plt.title('Lung Mask')

	    plt.subplot(1,2,2)
	    plt.imshow(array_list[0].reshape(new_dim, new_dim), cmap='bone')
	    plt.imshow(array_list[1].reshape(new_dim, new_dim), alpha=0.5, cmap=color_map)
	    plt.title('Infection Mask')

	    # plt.subplot(1,2,2)
	    # plt.imshow(array_list[0].reshape(img_size,img_size), cmap='bone')
	    # plt.imshow(array_list[1].reshape(img_size, img_size), alpha=0.5, cmap=color_map)
	    # plt.title('Lung and Infection Mask')

	#     plt.subplot(1,4,4)
	#     plt.imshow(array_list[0], cmap='bone')
	#     plt.imshow(array_list[3], alpha=0.5, cmap=color_map)
	#     plt.title('Lung and Infection Mask')

	    plt.show()


	# In[ ]:


	for index in [100,110,120,130,140,150]:
	    plot_sample([cts[index], infections[index]])


	# In[ ]:


	def dice_coeff(y_true, y_pred):
	    smooth = 1.
	    y_true_f = K.flatten(y_true)
	    y_pred_f = K.flatten(y_pred)
	    intersection = K.sum(y_true_f * y_pred_f)
	    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
	    return score

	def dice_loss(y_true, y_pred):
	    loss = 1 - dice_coeff(y_true, y_pred)
	    return loss


	def bce_dice_loss(y_true, y_pred):
	    loss = 0.5*binary_crossentropy(y_true, y_pred) + 0.5*dice_loss(y_true, y_pred)
	    return loss

	def tversky_loss(y_true, y_pred):
	    alpha = 0.5
	    beta  = 0.5
	    
	    ones = K.ones(K.shape(y_true))
	    p0 = y_pred      # proba that voxels are class i
	    p1 = ones-y_pred # proba that voxels are not class i
	    g0 = y_true
	    g1 = ones-y_true
	    
	    num = K.sum(p0*g0, (0,1,2))
	    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
	    
	    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
	    
	    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
	    return Ncl-T

	def weighted_bce_loss(y_true, y_pred, weight):
	    epsilon = 1e-7
	    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
	    logit_y_pred = K.log(y_pred / (1. - y_pred))
	    loss = weight * (logit_y_pred * (1. - y_true) + 
	                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
	    return K.sum(loss) / K.sum(weight)

	def weighted_dice_loss(y_true, y_pred, weight):
	    smooth = 1.
	    w, m1, m2 = weight, y_true, y_pred
	    intersection = (m1 * m2)
	    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
	    loss = 1. - K.sum(score)
	    return loss

	def weighted_bce_dice_loss(y_true, y_pred):
	    y_true = K.cast(y_true, 'float32')
	    y_pred = K.cast(y_pred, 'float32')
	    # if we want to get same size of output, kernel size must be odd
	    averaged_mask = K.pool2d(
	            y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
	    weight = K.ones_like(averaged_mask)
	    w0 = K.sum(weight)
	    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
	    w1 = K.sum(weight)
	    weight *= (w0 / w1)
	    loss = 0.5*weighted_bce_loss(y_true, y_pred, weight) + 0.5*dice_loss(y_true, y_pred)
	    return loss


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


	# In[ ]:


	class CosineAnnealingScheduler(Callback):
	    """Cosine annealing scheduler.
	    """

	    def __init__(self, T_max, eta_max, eta_min=0, verbose=1):
	        super(CosineAnnealingScheduler, self).__init__()
	        self.T_max = T_max
	        self.eta_max = eta_max
	        self.eta_min = eta_min
	        self.verbose = verbose

	    def on_epoch_begin(self, epoch, logs=None):
	        if not hasattr(self.model.optimizer, 'lr'):
	            raise ValueError('Optimizer must have a "lr" attribute.')
	        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
	        K.set_value(self.model.optimizer.lr, lr)
	        print('\nEpoch %05d: CosineAnnealingScheduler setting learning ''rate to %s.' % (epoch + 1, lr))

	    def on_epoch_end(self, epoch, logs=None):
	        logs = logs or {}
	        logs['lr'] = K.get_value(self.model.optimizer.lr)


	# In[ ]:


	cosine_annealer = CosineAnnealingScheduler(T_max=7, eta_max=0.0005, eta_min=0.0001)


	# In[ ]:


	plt.grid('True')
	rcParams['figure.figsize'] = 5,5
	T_max=7
	eta_max=0.002
	eta_min = 0.0001
	lr=[]
	for epoch in range(100):    
	    lr.append(eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2)
	lr = np.array(lr)
	plt.plot(lr)


	# In[13]:


	inputs = Input((new_dim, new_dim, 1))
	# s = Lambda(lambda x: x / 255) (inputs)

	# def mish(inputs):
	#     return inputs * tf.math.tanh(tf.math.softplus(inputs))
	    
	c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (inputs)
	c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c1)
	c1 = BatchNormalization()(c1)
	p1 = MaxPooling2D((2, 2)) (c1)
	p1 = Dropout(0.25)(p1)

	c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p1)
	c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c2)
	c2 = BatchNormalization()(c2)
	p2 = MaxPooling2D((2, 2)) (c2)
	p2 = Dropout(0.25)(p2)

	c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p2)
	c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c3)
	c3 = BatchNormalization()(c3)
	p3 = MaxPooling2D((2, 2)) (c3)
	p3 = Dropout(0.25)(p3)

	c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p3)
	c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c4)
	c4 = BatchNormalization()(c4)
	p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
	p4 = Dropout(0.25)(p4)

	c5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (p4)
	c5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c5)

	u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
	u6 = concatenate([u6, c4])
	u6 = BatchNormalization()(u6)
	c6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u6)
	c6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c6)


	u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c6)
	u7 = concatenate([u7, c3])
	u7 = BatchNormalization()(u7)
	c7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u7)
	c7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c7)


	u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
	u8 = concatenate([u8, c2])
	u8 = BatchNormalization()(u8)
	c8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u8)
	c8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c8)


	u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
	u9 = concatenate([u9, c1], axis=3)
	u9 = BatchNormalization()(u9)
	c9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (u9)
	c9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c9)

	outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

	model = Model(inputs=[inputs], outputs=[outputs])
	model.summary()


	# In[ ]:


	batch_size = 32
	epochs = 80
	#lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=7, verbose=1)
	filepath_dice_coeff_1="unet_covid_fold1.hdf5"
	filepath_dice_coeff_2="unet_covid_fold2.hdf5"
	filepath_dice_coeff_3="unet_covid_fold3.hdf5"

	checkpoint_dice_1 = ModelCheckpoint(filepath_dice_coeff_1, monitor='val_dice_coeff', verbose=1, save_best_only=True, mode='max')
	checkpoint_dice_2 = ModelCheckpoint(filepath_dice_coeff_2, monitor='val_dice_coeff', verbose=1, save_best_only=True, mode='max')
	checkpoint_dice_3 = ModelCheckpoint(filepath_dice_coeff_3, monitor='val_dice_coeff', verbose=1, save_best_only=True, mode='max')

	# checkpoint_loss = ModelCheckpoint(filepath_loss, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


	# In[15]:


	start = timeit.default_timer()

	kf = KFold(n_splits=3, random_state=42, shuffle=True)

	fold_number = 1

	for train_index, test_index in kf.split(cts):

	  print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
	  print("Current fold number going:", fold_number)
	  print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

	  x_train, x_valid = cts[train_index], cts[test_index]
	  y_train, y_valid = infections[train_index], infections[test_index]
	  print("Shapes:", x_train.shape, x_valid.shape)

	  if fold_number == 1:
	    model.compile(optimizer=Adam(lr = 0.0005), loss=bce_dice_loss, metrics=[dice_coeff])

	    results_1 = model.fit(x_train, y_train, batch_size=batch_size, epochs=80,
	                    validation_data=(x_valid, y_valid),
	                    callbacks = [checkpoint_dice_1])
	    
	  if fold_number == 2:
	    model.compile(optimizer=Adam(lr = 0.0005), loss=bce_dice_loss, metrics=[dice_coeff])

	    results_2 = model.fit(x_train, y_train, batch_size=batch_size, epochs=20,
	                    validation_data=(x_valid, y_valid),
	                    callbacks = [checkpoint_dice_2])
	    
	  if fold_number == 3:
	    model.compile(optimizer=Adam(lr = 0.0005), loss=bce_dice_loss, metrics=[dice_coeff])

	    results_3 = model.fit(x_train, y_train, batch_size=batch_size, epochs=20,
	                    validation_data=(x_valid, y_valid),
	                    callbacks = [checkpoint_dice_3])
	    
	    
	  fold_number = fold_number + 1


	stop = timeit.default_timer()


	# In[26]:


	print('Time of 3-fold cross validation: ', stop - start) 


	# In[ ]:


	model.save_weights(filepath_dice_coeff_1)
	model.save_weights(filepath_dice_coeff_2)
	model.save_weights(filepath_dice_coeff_3)


	# In[3]:


	files.download(filepath_dice_coeff_1)


	# In[ ]:


	files.download(filepath_dice_coeff_2)


	# In[ ]:


	files.download(filepath_dice_coeff_3)


	# In[ ]:


	# os.listdir('drive/My Drive/cts and infections/cross_val_model_weights/')


	# In[ ]:


	# model.load_weights('drive/My Drive/cts and infections/cross_val_model_weights/unet_covid_fold2.hdf5')


	# In[ ]:


	# x_train = np.concatenate((x_train, cts_aug), axis=0)
	# y_train = np.concatenate((y_train, infections_aug), axis = 0)
	# print(x_train.shape, y_train.shape)


	# * Loss functions and metrics

	# * All the hyperparameters are put in place after repeating trial and error for a fixed number of epochs.

	# * Some callbacks (model checkpointing with least validation loss, highest validation dice coefficient, learning rate reduction after some patience number of epochs)
	# * Also experimented with exponential decaying learning rate but found ReduceLROnPlateau a bit effective in this case.

	# In[ ]:


	gc.collect()


	# In[ ]:





	# In[ ]:


	paths = [filepath_dice_coeff_1, filepath_dice_coeff_2, filepath_dice_coeff_3]


	# In[28]:


	kf.get_n_splits(cts)


	# In[24]:


	split_number = 1
	for train_index, test_index in kf.split(cts):


	  print("......................................................................................................................")
	  print("Current split number going:", split_number)
	  print("......................................................................................................................")
	  x_train, x_valid = cts[train_index], cts[test_index]
	  y_train, y_valid = infections[train_index], infections[test_index]
	  model.load_weights(paths[split_number-1])
	  score = model.evaluate(x_valid, y_valid, batch_size=32)
	  print("test loss, test dice coefficient:", score)

	  split_number = split_number + 1


	# In[ ]:





	# In[30]:


	split_number = 1
	the_range = np.arange(0.30,0.80, 0.05)
	print(len(the_range))

	total_dices=[]
	total_ious=[]
	total_precisions=[]
	total_recalls=[]

	for train_index, test_index in kf.split(cts):

	  dices_per_threshold = []
	  ious_per_threshold = []
	  precisions_per_threshold = []
	  recalls_per_threshold = []

	  print(".................................................................................................................................................")
	  print("Current split number going:", split_number)
	  print(".................................................................................................................................................")
	  x_train, x_valid = cts[train_index], cts[test_index]
	  y_train, y_valid = infections[train_index], infections[test_index]
	  model.load_weights(paths[split_number-1])

	  for t in the_range:

	    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
	    print("Calculating for threshold:", t)
	    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

	    model.compile(optimizer=Adam(lr = 0.0005), loss=bce_dice_loss, metrics=[sm.metrics.FScore(threshold=t)])
	    dices_per_threshold.append(model.evaluate(x_valid, y_valid, batch_size=32)[1])

	    model.compile(optimizer=Adam(lr = 0.0005), loss=bce_dice_loss, metrics=[sm.metrics.IOUScore(threshold=t)])
	    ious_per_threshold.append(model.evaluate(x_valid, y_valid, batch_size=32)[1])

	    model.compile(optimizer=Adam(lr = 0.0005), loss=bce_dice_loss, metrics=[sm.metrics.Precision(threshold=t)])
	    precisions_per_threshold.append(model.evaluate(x_valid, y_valid, batch_size=32)[1])
	    
	    model.compile(optimizer=Adam(lr = 0.0005), loss=bce_dice_loss, metrics=[sm.metrics.Recall(threshold=t)])
	    recalls_per_threshold.append(model.evaluate(x_valid, y_valid, batch_size=32)[1])

	  total_dices.append(dices_per_threshold)
	  total_ious.append(ious_per_threshold)
	  total_precisions.append(precisions_per_threshold)
	  total_recalls.append(recalls_per_threshold)

	  split_number = split_number + 1


	# In[31]:


	total_dices = np.array(total_dices)
	total_dices = np.transpose(total_dices)
	df_dices = pd.DataFrame(data=total_dices, index = the_range, columns = [1,2,3])
	print("3-fold Dices dataframe")
	print("Rows indices: Thresholds, Column indices: Split Number")
	df_dices


	# In[32]:


	print("Maximum validation dice on any splits:", np.max(total_dices))
	print("Maximum validation dice on each of the 3 splits (any threshold chosen):", np.array(df_dices.max(axis=0)))
	print("Best threshold for each split", the_range[df_dices[1].argmax()], the_range[df_dices[2].argmax()], the_range[df_dices[3].argmax()])
	print("Mean of all obtained dices:", df_dices.mean().mean())


	# In[ ]:





	# In[33]:


	total_ious = np.array(total_ious)
	total_ious = np.transpose(total_ious)
	df_ious = pd.DataFrame(data=total_ious, index = the_range, columns = [1,2,3])
	print("3-fold Ious dataframe")
	print("Rows indices: Thresholds, Column indices: Split Number")
	df_ious


	# In[34]:


	print("Maximum validation iou on any splits:", np.max(total_ious))
	print("Maximum validation iou on each of the 3 splits (any threshold chosen):", np.array(df_ious.max(axis=0)))
	print("Best threshold for each split", the_range[df_ious[1].argmax()], the_range[df_ious[2].argmax()], the_range[df_ious[3].argmax()])
	print("Mean of all obtained ious:", df_ious.mean().mean())


	# In[ ]:





	# In[35]:


	total_precisions = np.array(total_precisions)
	total_precisions = np.transpose(total_precisions)
	df_precision = pd.DataFrame(data=total_precisions, index = the_range, columns = [1,2,3])
	print("3-fold precision dataframe")
	print("Rows indices: Thresholds, Column indices: Split Number")
	df_precision


	# In[36]:


	print("Maximum validation precision on any splits:", np.max(total_precisions))
	print("Maximum validation precision on each of the 3 splits (any threshold chosen):", np.array(df_precision.max(axis=0)))
	print("Best threshold for each split", the_range[df_precision[1].argmax()], the_range[df_precision[2].argmax()], the_range[df_precision[3].argmax()])
	print("Mean of all obtained precisions:", df_precision.mean().mean())


	# In[ ]:





	# In[37]:


	total_recalls = np.array(total_recalls)
	total_recalls = np.transpose(total_recalls)
	df_recall = pd.DataFrame(data=total_recalls, index = the_range, columns = [1,2,3])
	print("3-fold precision dataframe")
	print("Rows indices: Thresholds, Column indices: Split Number")
	df_recall


	# In[38]:


	print("Maximum validation recall on any splits:", np.max(total_recalls))
	print("Maximum validation recall on each of the 3 splits (any threshold chosen):", np.array(df_recall.max(axis=0)))
	print("Best threshold for each split", the_range[df_recall[1].argmax()], the_range[df_recall[2].argmax()], the_range[df_recall[3].argmax()])
	print("Mean of all obtained recalls:", df_recall.mean().mean())


	# In[ ]:





	# In[ ]:





	# In[ ]:


	plt.rcParams.update({'font.size': 22})
	def compare_actual_and_predicted(image_no):

	    fig = plt.figure(figsize=(50,50))

	    plt.subplot(1,5,1)
	    plt.imshow(cts[image_no].reshape(new_dim, new_dim), cmap='bone')
	    plt.title('Original Image (CT)')

	    plt.subplot(1,5,2)
	    plt.imshow(infections[image_no].reshape(new_dim,new_dim), cmap='bone')
	    plt.title('Actual mask')

	    plt.subplot(1,5,3)
	    model.load_weights('drive/My Drive/cts and infections/cross_val_model_weights/unet_covid_fold1.hdf5')
	    temp = model.predict(cts[image_no].reshape(1,new_dim, new_dim, 1))
	    plt.imshow(temp.reshape(new_dim,new_dim), cmap='bone')
	    plt.title('Model 1 output')

	    plt.subplot(1,5,4)
	    model.load_weights('drive/My Drive/cts and infections/cross_val_model_weights/unet_covid_fold2.hdf5')
	    temp = model.predict(cts[image_no].reshape(1,new_dim, new_dim, 1))
	    plt.imshow(temp.reshape(new_dim,new_dim), cmap='bone')
	    plt.title('Model 2 output')

	    plt.subplot(1,5,5)
	    model.load_weights('drive/My Drive/cts and infections/cross_val_model_weights/unet_covid_fold3.hdf5')
	    temp = model.predict(cts[image_no].reshape(1,new_dim, new_dim, 1))
	    plt.imshow(temp.reshape(new_dim,new_dim), cmap='bone')
	    plt.title('Model 3 output')

	    plt.show()
	    
	# plt.imshow(temp.reshape(img_size, img_size), cmap = 'bone')
	# plt.imshow(infections_scaled[120].reshape(img_size, img_size), cmap ='summer')


	# In[40]:


	for i in [30,40,50,55, 355, 380, 90]:
	    compare_actual_and_predicted(i)


	# In[ ]:


	gc.collect()


	# In[ ]:


	return


if __name__ == "__main__":
	three_fold_runner_unet_infection_segmentation()



