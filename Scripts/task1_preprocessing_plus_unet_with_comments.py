	#!/usr/bin/env python
	# coding: utf-8

	# In[ ]:

def holdout_runner_unet_infection_segmentation():

	get_ipython().system('pip install imgaug')                     # for image augmentation')


	# In[ ]:


	get_ipython().system('pip install -U segmentation-models')    # ONLY used for dice metric and IOU metric computation, models are made from scratch')


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


	# In[4]:


	get_ipython().system('kaggle datasets download -d deadskull7/unetpp')


	# In[ ]:


	#copyfile("/content/drive/My Drive/covid19-ct-scans.zip","/content/covid19-ct-scans.zip")
	with ZipFile('unetpp.zip', 'r') as zipObj:
	   # Extract all the contents of zip file in current directory
	   zipObj.extractall('unetpp')


	# In[6]:


	os.listdir()


	# * Extracting zip file here

	# In[ ]:


	#copyfile("/content/drive/My Drive/covid19-ct-scans.zip","/content/covid19-ct-scans.zip")
	with ZipFile('covid19-ct-scans.zip', 'r') as zipObj:
	   # Extract all the contents of zip file in current directory
	   zipObj.extractall('covid19-ct-scans')


	# In[8]:


	# Read and examine metadata
	raw_data = pd.read_csv('/content/covid19-ct-scans/metadata.csv')
	raw_data = raw_data.replace('../input/covid19-ct-scans/','/content/covid19-ct-scans/',regex=True)
	raw_data.head(5)


	# In[9]:


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
	# * The contour (largest closed boundary) with the largest area would be the contour covering both the lungs. We can also take 2nd largest and 3rd argest contours for the 2 lungs individually and concatenating for getting the maximum ROI in the same resolution.
	# * In global thresholding, we can use an arbitrary chosen value as a threshold. In contrast, Otsu's method avoids having to choose a value and determines it automatically and returns it.

	# * Another important point is while cropping a CT scan we have to make sure that its corresponding segmentation map is also cropped by same limits otherwise pixel level labeling will go wrong and the model might map a good area with an infectious area and vice versa which we don't want.

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
	  
	#   if demo == 1:

	#     fig = plt.figure()
	#     from pylab import rcParams
	#     rcParams['figure.figsize'] = 35, 35

	#     plt.subplot(1, 3, 1)
	#     plt.imshow(test_img, cmap='bone')
	#     plt.title("Original CT-Scan")

	#     plt.subplot(1, 3, 2)
	#     plt.imshow(thresh, cmap='bone')
	#     plt.title("Binary Mask")

	#     plt.subplot(1, 3, 3)
	#     plt.imshow(fused, cmap='bone')
	#     plt.title("Cropped CT scan after making bounding rectangle")

	#     # plt.subplot(1, 4, 4)
	#     # plt.imshow(super_cropped, cmap='bone')
	#     # plt.title("Cropped further manually")

	#     plt.show()

	  return(fused, points_lung1, points_lung2)


	# * Total slices are 3520. These have been sliced out by 20% in the front and by 20% in the last of each file since in general these didn't had any infection masks and some didn't had the lungs, removed as noise. This level could be adjusted too.

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


	# In[26]:


	for i in range(0, 20):
	    read_nii(raw_data.loc[i,'lung_mask'], lungs, 'lungs')
	    read_nii(raw_data.loc[i,'ct_scan'], cts, 'cts') 
	    read_nii(raw_data.loc[i,'infection_mask'], infections, 'infections')


	# * See the following fully processed sample

	# In[21]:


	x = 60
	rcParams['figure.figsize'] = 10,10

	plt.subplot(1, 2, 1)
	plt.imshow(cts[x], cmap='bone')
	plt.title("Final preprocessed (CLAHE Enhanced + Cropped) Image")

	plt.subplot(1, 2, 2)
	plt.imshow(infections[x], cmap='bone')
	plt.title("Final preprocessed corresponding Mask")

	print(cts[x].shape, infections[x].shape)


	# * Also, figuring out that 497 slices were of complete black masks i.e. no infection. For right now, kept out as didn't want to bother the segmentation model with this. 
	# * A better approach would be to make a sub-task and use that sub-task's output as input in our main task. Train a separate binary classifier to classify the CT scan as complete black or not, then the not ones to be passed from the segmentation model trained in this notebook.

	# In[27]:


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


	# * Finally 1615 samples which will later be split into train and test

	# In[28]:


	print(len(cts) , len(infections))


	# * Also, since we copped the images and masks, all cannot be of same size, so again a dimension needs to be decided to which all could be resized to. 
	# * Used median of the all width and height but couldn't fit the RAM so reduced the size to 224, though images with larger resolution with more clear features will possibly give better results on the same model.

	# In[29]:


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


	x_train, x_valid, y_train, y_valid = train_test_split(cts, infections, test_size=0.3, random_state=42)


	# In[ ]:


	print(x_train.shape, x_valid.shape)


	# In[ ]:


	# x_train = np.concatenate((x_train, cts_aug), axis=0)
	# y_train = np.concatenate((y_train, infections_aug), axis = 0)
	# print(x_train.shape, y_train.shape)


	# * Loss functions and metrics

	# In[4]:


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


	z = []
	y = []
	def step_decay(epoch):
	    initial_lrate = 0.0008
	    drop = 0.8
	    epochs_drop = 10
	    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	    lrate_ = initial_lrate * math.pow(drop, (1+epoch)/epochs_drop)
	    # print('New learning rate', lrate)
	    z.append(lrate)
	    y.append(lrate_)
	    # return lrate

	# lrate = LearningRateScheduler(step_decay)


	# In[ ]:


	for i in range(100):
	  step_decay(i)
	plt.grid('True')
	plt.title("Exponential decaying learning rate")
	plt.xlabel('Epochs')
	plt.ylabel('Learning rates')
	plt.plot(np.array(z), color='g')
	plt.plot(np.array(y), color='r')
	plt.legend(['Step decay', 'Continuous decay'])


	# In[ ]:


	for i in range(2):
	  for i in range(100):
	    step_decay(i)
	  plt.grid('True')
	  plt.title("Exponential decaying learning rate")
	  plt.xlabel('Epochs')
	  plt.ylabel('Learning rates')
	  plt.plot(np.array(z), color='g')
	  plt.plot(np.array(y), color='r')
	  plt.legend(['Step decay', 'Continuous decay'])


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
	# rcParams['figure.figsize'] = 9,5
	T_max=7
	eta_max=0.002
	eta_min = 0.0001
	lr=[]
	initial_lrate = eta_max
	drop = 0.8
	epochs_drop = 10

	for epoch in range(100):    
	    lr.append(eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2)
	    eta_max = initial_lrate * math.pow(drop, (1+epoch)/epochs_drop)
	lr = np.array(lr)
	plt.xlabel("Epochs")
	plt.title("Cosine Annealing Learning rate + Exponential decaying max lr")
	plt.plot(lr, 'c')


	# In[ ]:


	plt.grid('True')
	# rcParams['figure.figsize'] = 9,5
	T_max=7
	eta_max=0.002
	eta_min = 0.0001
	lr=[]
	for epoch in range(100):    
	    lr.append(eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2)
	lr = np.array(lr)
	plt.xlabel("Epochs")
	plt.title("Cosine Annealing Learning rate")
	plt.plot(lr, 'grey')


	# In[ ]:


	batch_size = 32
	epochs = 80
	#lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=7, verbose=1)
	filepath_dice_coeff="unet_covid_weights_dice_coeff.hdf5"
	filepath_loss = "unet_covid_weights_val_loss.hdf5"
	checkpoint_dice = ModelCheckpoint(filepath_dice_coeff, monitor='val_dice_coeff', verbose=1, save_best_only=True, mode='max')
	checkpoint_loss = ModelCheckpoint(filepath_loss, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


	# In[ ]:


	model.compile(optimizer=Adam(lr = 0.0005), loss=bce_dice_loss, metrics=[dice_coeff])


	# In[ ]:


	results = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
	                    validation_data=(x_valid, y_valid),
	                    callbacks = [checkpoint_dice, checkpoint_loss])


	# In[ ]:


	gc.collect()


	# In[ ]:


	model.load_weights(filepath_dice_coeff)


	# In[ ]:


	model.save_weights('unet_0.8954_cosine_annealer.h5')


	# In[ ]:


	files.download('unet_0.8954_cosine_annealer.h5')


	# In[ ]:


	model_json = model.to_json()
	with open("unet_0.8954_cosine_annealer.json","w") as json_file:
	     json_file.write(model_json)

	files.download("unet_0.8954_cosine_annealer.json")


	# In[ ]:


	score = model.evaluate(x_valid, y_valid, batch_size=32)
	print("test loss, test dice coefficient:", score)


	# In[ ]:


	rcParams['figure.figsize'] = 5, 5
	plt.plot(results.history['dice_coeff'])
	plt.plot(results.history['val_dice_coeff'])
	plt.title('Dice Coefficient')
	plt.ylabel('Dice coefficient')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.ylim(0, 3)
	plt.plot(results.history['loss'])
	plt.plot(results.history['val_loss'])
	plt.title('Dice Loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()


	# In[ ]:


	plt.rcParams["axes.grid"] = False


	# In[ ]:


	def compare_actual_and_predicted(image_no):
	    temp = model.predict(cts[image_no].reshape(1,new_dim, new_dim, 1))

	    fig = plt.figure(figsize=(15,15))

	    plt.subplot(1,3,1)
	    plt.imshow(cts[image_no].reshape(new_dim, new_dim), cmap='bone')
	    plt.title('Original Image (CT)')

	    plt.subplot(1,3,2)
	    plt.imshow(infections[image_no].reshape(new_dim,new_dim), cmap='bone')
	    plt.title('Actual mask')

	    plt.subplot(1,3,3)
	    plt.imshow(temp.reshape(new_dim,new_dim), cmap='bone')
	    plt.title('Predicted mask')

	    plt.show()
	    
	# plt.imshow(temp.reshape(img_size, img_size), cmap = 'bone')
	# plt.imshow(infections_scaled[120].reshape(img_size, img_size), cmap ='summer')


	# In[ ]:


	for i in [30,40,50,55, 355, 380, 90]:
	    compare_actual_and_predicted(i)


	# In[ ]:


	gc.collect()


	# In[ ]:





	# # Some Significant Post-Processing

	# In[ ]:


	drive.mount('/content/drive')


	# In[ ]:


	# model.load_weights(filepath_dice_coeff)
	model.load_weights('/content/drive/My Drive/cts and infections/unet_0.90_lr_0.0005.h5')


	# In[ ]:


	the_range = np.arange(0.10,0.80, 0.05)


	# In[ ]:


	dices=[]
	ious=[]

	for t in the_range:
	  iou = sm.metrics.IOUScore(threshold=t)
	  dice = sm.metrics.FScore(threshold=t)
	  model.compile(optimizer=Adam(lr = 0.0005), loss=bce_dice_loss, metrics=[dice, iou])
	  score = model.evaluate(x_valid, y_valid, batch_size=32)
	  dices.append(score[1])
	  ious.append(score[2])


	# In[ ]:


	print('DICES:',dices)
	print("IOUS:",ious)
	print("Best Threshold:", the_range[np.argmax(dices)])
	print("Best dice score:", dices[np.argmax(dices)])
	print("Best iou score:", ious[np.argmax(ious)])


	# In[ ]:





	# In[ ]:


	print("Best Threshold:", the_range[np.argmax(dices)])
	fig = plt.figure(figsize=(15,5))

	plt.subplot(1, 2, 1)
	plt.plot(the_range, dices)
	plt.title("Validation Dice coefficient vs Threshold")

	plt.subplot(1, 2, 2)
	plt.plot(the_range, ious)
	plt.title("Validation IOU vs Threshold")

	plt.show()


	# In[ ]:


	the_new_range = np.arange(0.52,0.60, 0.001)


	# In[ ]:


	new_dices=[]
	new_ious=[]

	for t in the_new_range:
	  iou = sm.metrics.IOUScore(threshold=t)
	  dice = sm.metrics.FScore(threshold=t)
	  model.compile(optimizer=Adam(lr = 0.0005), loss=bce_dice_loss, metrics=[dice, iou])
	  score = model.evaluate(x_valid, y_valid, batch_size=32)
	  new_dices.append(score[1])
	  new_ious.append(score[2])


	# In[ ]:


	print("We just checked for",len(the_new_range), "steps between 0.52 and 0.60")


	# In[ ]:


	print('NEW DICES:',new_dices)
	print("NEW IOUS:",new_ious)
	print("New Best Threshold:", the_new_range[np.argmax(new_dices)])
	print("Best new dice score:", new_dices[np.argmax(new_dices)])
	print("Best new iou score:", new_ious[np.argmax(new_ious)])


	# In[ ]:


	print("Best Threshold:", the_new_range[np.argmax(new_dices)])
	fig = plt.figure(figsize=(15,5))

	plt.subplot(1, 2, 1)
	plt.plot(the_new_range, new_dices)
	plt.title("Validation Dice coefficient vs Threshold")

	plt.subplot(1, 2, 2)
	plt.plot(the_new_range, new_ious)
	plt.title("Validation IOU vs Threshold")

	plt.show()


	# In[ ]:


	the_prec_rec_range = np.arange(0,1, 0.05)


	# In[ ]:


	precisions=[]
	recalls=[]

	for t in the_prec_rec_range:
	  precision = sm.metrics.Precision(threshold=t)
	  recall = sm.metrics.Recall(threshold=t)
	  model.compile(optimizer=Adam(lr = 0.0005), loss=bce_dice_loss, metrics=[precision, recall])
	  score = model.evaluate(x_valid, y_valid, batch_size=32)
	  precisions.append(score[1])
	  recalls.append(score[2])


	# In[ ]:


	print('PRECISIONS:',precisions)
	print("RRECALLS:",recalls)
	print("Best Threshold for Precision:", the_prec_rec_range[np.argmax(precisions)])
	print("Best Threshold for Recall:", the_prec_rec_range[np.argmax(recalls)])
	print("Best precision score:", precisions[np.argmax(precisions)])
	print("Best recall score:", recalls[np.argmax(recalls)])


	# In[ ]:


	rcParams['figure.figsize'] = 7,7
	plt.rcParams["axes.grid"] = True
	plt.title("Precision and Recalls vs Thresholds")
	plt.xlabel('Threshold')
	plt.ylabel('Precision and Recall Values')
	plt.plot(the_prec_rec_range,precisions, color='red')
	plt.plot(the_prec_rec_range, recalls, color='green')
	plt.legend(['Precision', 'Recall'])


	# In[ ]:





	# In[ ]:





	# In[ ]:


	drive.mount('/content/drive')


	# In[ ]:


	# os.listdir('/content/drive/My Drive/cts and infections/')


	# In[ ]:


	# model.load_weights(filepath_dice_coeff)
	model.load_weights('/content/drive/My Drive/cts and infections/unet_0.8832_lr_0.0005.h5')


	# In[ ]:


	# os.listdir('/content/drive/My Drive/cts and infections/')


	# In[ ]:


	layer_name = 'conv2d_9'   # name could be changed also
	intermediate_layer_model = Model(inputs=model.input,
	                                 outputs=model.get_layer(layer_name).output)


	# * Now I tried to investigate the hypothesis that some set of CT scans belong to a distribution that is easy to segment between infection and background while other CT scans are belonging to a distribution that is not easy to segment between infection and the background.
	# * Though till now I have just done it with one sample, it is just a speculation right now, not thoroughly investigated.
	# * My target is to cluster the data into 2 parts, one good and easily segmentable and the other bad (tough to segment). Then, fitting two distinct models on both these sections of the data separately. This division into sub-task will help to gain better metric. 
	# * So, whenever a new point is to be predicted, we have to first cluster the point into one of the class, then based on its label, we will assign one of the 2 models which will predict its final segmentation maskk. So, converting this single step process into 2 step for greater reliability.

	# In[ ]:





	# In[5]:


	intermediate_output = intermediate_layer_model.predict(cts)
	print("Extracted feature shape:", intermediate_output.shape)
	data = []
	for i in range(0, len(intermediate_output)):
	  curr_img = intermediate_output[i]
	  temp_output = np.reshape(np.rollaxis(curr_img, 2),(curr_img.shape[2],curr_img.shape[0],curr_img.shape[1]))
	  data.append(temp_output.flatten())
	data = np.array(data)
	print("Flattened features for the input of PCA:",data.shape)

	pca = PCA(n_components=1000)
	pca.fit(data)
	print("Total variance explained:",np.sum(pca.explained_variance_ratio_))
	new_data = pca.transform(data)
	print("Input data shape for Clustering:",new_data.shape)

	kmeans = KMeans(n_clusters=2, random_state=0).fit(new_data)
	cts_labels_kmeans = kmeans.labels_
	unique_cts_kmeans, counts_cts_kmeans = np.unique(cts_labels_kmeans, return_counts=True)
	print("Label count for Kmeans on cts:",dict(zip(unique_cts_kmeans, counts_cts_kmeans)))

	# mean_shift = MeanShift().fit(new_data)
	# cts_labels_mean_shift = mean_shift.labels_
	# unique_cts_mean_shift, counts_cts_mean_shift = np.unique(cts_labels_mean_shift, return_counts=True)
	# print("Label count for Mean Shift on cts:",dict(zip(unique_cts_mean_shift, counts_cts_mean_shift)))
	# print("---------------------------------------------------------------------------------------------")


	# In[6]:


	intermediate_output = intermediate_layer_model.predict(x_valid)
	print("Extracted feature shape:", intermediate_output.shape)
	data = []
	for i in range(0, len(intermediate_output)):
	  curr_img = intermediate_output[i]
	  temp_output = np.reshape(np.rollaxis(curr_img, 2),(curr_img.shape[2],curr_img.shape[0],curr_img.shape[1]))
	  data.append(temp_output.flatten())
	data = np.array(data)
	print("Flattened features for the input of PCA:",data.shape)

	new_data_valid = pca.transform(data)

	valid_labels_kmeans = kmeans.predict(new_data_valid)
	unique_valid_kmeans, counts_valid_kmeans = np.unique(valid_labels_kmeans, return_counts=True)
	print("Label count for Kmeans on valid:", dict(zip(unique_valid_kmeans, counts_valid_kmeans)))

	# valid_labels_mean_shift = mean_shift.predict(new_data_valid)
	# unique_valid_mean_shift, counts_valid_mean_shift = np.unique(valid_labels_mean_shift, return_counts=True)
	# print("Label count for Kmeans on valid:", dict(zip(unique_valid_mean_shift, counts_valid_mean_shift)))


	# In[ ]:


	x_valid_zeroes = x_valid[np.where(valid_labels_kmeans == 0)[0]]
	y_valid_zeroes = y_valid[np.where(valid_labels_kmeans == 0)[0]]
	x_valid_ones = x_valid[np.where(valid_labels_kmeans == 1)[0]]
	y_valid_ones = y_valid[np.where(valid_labels_kmeans == 1)[0]]


	# In[ ]:


	print(x_valid_zeroes.shape, x_valid_ones.shape)


	# In[ ]:


	model.compile(optimizer=Adam(lr = 0.0005), loss=bce_dice_loss, metrics=[sm.metrics.FScore(threshold=0.547), 'accuracy', sm.metrics.IOUScore(threshold=0.547)])


	# In[11]:


	score = model.evaluate(x_valid, y_valid, batch_size=32)
	print("test loss, test dice coefficient, test accuracy, test iou:", score)


	# In[10]:


	score = model.evaluate(x_valid_zeroes, y_valid_zeroes, batch_size=32)
	print("test loss, test dice coefficient:", score)


	# In[8]:


	score = model.evaluate(x_valid_ones, y_valid_ones, batch_size=32)
	print("test loss, test dice coefficient:", score)


	# In[ ]:





	# In[ ]:


	return


if __name__ == "__main__":
	holdout_runner_unet_infection_segmentation()

