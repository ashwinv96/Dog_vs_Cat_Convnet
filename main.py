import cv2
import numpy as np
import pandas as pd
import os
import joblib
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam
import convnet as cnn #custom Keras CNN 

# Check if training images already processed to save time
if os.path.isfile('train_data.npy'): #if file exists
	train_data = np.load('train_data.npy') #load file
else:
	print("train data cant be found!")
	train_data = cnn.create_train_set() #call train set generator

model = cnn.custom_convnet(train_data) #custom CNN method
model.save("cat_dog_classifier_model.h5") #save model

# Check if testing images already processed to save time
if os.path.isfile('testing_data.npy'): #if file exists
	test_set = np.load('testing_data.npy') #load file
else:
	print("testing data cant be found!")
	test_set, gt_test_ids = cnn.create_test_set(TEST_DIR) #call test set generator
	#gt_test_ids are the filename without extension (which is a number of each image)
	# ex: 1.png --> 1, 4345.png --> 4345 (can be used for matching with test set for visual inspection)

cnn.predict_and_plot(test_set, 36, 6, 6, model) # predict on test set and plot predictions

