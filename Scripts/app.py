#!/usr/bin/env python
# coding: utf-8

# In[6]:


from task1_crossval_3folds_unet import *
from task1_crossval_4folds_unet import *
from task1_preprocessing_plus_unet_with_comments import *
from task1_unet_plus_plus import *
from task2_covid19_classifcation import *
from task3_lung_segmentation_unet import *




print("\n\n\n\n")
print("--------------------------------------------------------------------------------------")
print(" 'one' --> Task1: 3-fold cross-validation UNet (Infection Segmentation)")
print(" 'two' --> Task1: 4-fold cross-validation UNet (Infection Segmentation)")
print(" 'three' --> Task1: UNet original holdout method (Infection Segmentation)")
print(" 'four' --> Task1: UNet++ holdout method (Infection Segmentation)")
print(" 'five' --> Task2: COVID-19 Classification")
print(" 'six' --> Task3: Lung Segmentation")
print("--------------------------------------------------------------------------------------")
print("\n\n\n\n\n")


print("Enter from one of the {'one', 'two', 'three', 'four', 'five', 'six', 'seven'}")
num = input()


# In[ ]:


if num == 'one':
	three_fold_runner_unet_infection_segmentation()
    

if num == 'two':
	four_fold_runner_unet_infection_segmentation()


if num == 'three':
	holdout_runner_unet_infection_segmentation()


if num == 'four':
	holdout_runner_unetplusplus_infection_segmentation()


if num == 'five':
	runner_classification()


if num == 'six':
	runner_lung_segmentation()


