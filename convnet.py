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


TRAIN_DIR = '/media/60f9fb22-1078-40d7-bd50-65ab0a349b19/kaggle_projects/dogs_vs_cats/train'
TEST_DIR = '/media/60f9fb22-1078-40d7-bd50-65ab0a349b19/kaggle_projects/dogs_vs_cats/test1'
MIXED_DIR = '/media/60f9fb22-1078-40d7-bd50-65ab0a349b19/kaggle_projects/dogs_vs_cats/mixed'
VID_FILEPATH = '/media/60f9fb22-1078-40d7-bd50-65ab0a349b19/kaggle_projects/dogs_vs_cats/dog.mp4'
IMG_SIZE = 50
LR = 1e-2
TEST_MODE = False
USE_SMALL_SET = False
EPOCHS = 1
DECAY = LR/EPOCHS


def lr_time_based_decay(EPOCHS,LR):
	return LR * 1 / (1 + DECAY * EPOCHS)
	

def predict_and_plot(testset, samples, dim1, dim2, model):
	fig = plt.figure()
	for num, data in enumerate(testset[:samples]):
		y = fig.add_subplot(dim1,dim2,num+1)
		orig = data
		data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

		model_out = model.predict([data])[0]
		print(model_out)
		
		if np.round(model_out)[0] == 1:
			str_label = 'Dog'
		elif np.round(model_out)[0] == 0:
			str_label = 'Cat'
		
		y.imshow(orig, cmap='gray')
		plt.title(str_label)
		y.axes.get_xaxis().set_visible(False)
		y.axes.get_yaxis().set_visible(False)
	plt.show()
	fig.clear()

	
def create_train_set():
	gt_labels = []
	train_data = []
	count = 0
	for img in tqdm(os.listdir(TRAIN_DIR)):
		count += 1
		if (TEST_MODE is True) and count > 99:
			break
		elif (USE_SMALL_SET is True) and count > 5000:
			break
		img_raw = cv2.imread(os.path.join(TRAIN_DIR,img), cv2.IMREAD_GRAYSCALE)
		img_raw = cv2.resize(img_raw, (IMG_SIZE, IMG_SIZE))
		
		img_lbl = img.split('.')[-3]
		if img_lbl == "dog":
			train_data.append([np.array(img_raw), [1, 0]]) # [DOG, CAT]
		elif img_lbl == "cat":
			train_data.append([np.array(img_raw), [0, 1]])
		else:
			print('name {} not in proper format!'.format(img))
			break

	shuffle(train_data)
	train_data = np.array(train_data)
	np.save('train_data.npy', train_data)

	return train_data

def create_test_set(direc):
	testing_data = []
	for img in tqdm(os.listdir(direc)):
		path = os.path.join(direc, img)
		
		img_num = img.split('.')[0]
		print('jpg name: ', img)
		img_raw = cv2.imread(os.path.join(direc,img), cv2.IMREAD_GRAYSCALE)
		img_raw = cv2.resize(img_raw, (IMG_SIZE, IMG_SIZE))
		testing_data.append([np.array(img_raw), img_num])
		#print(testing_data)
	
	testing_data = np.array(testing_data)
	test_set = np.array([i[0]/255 for i in testing_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
	gt_test_ids = np.array([i[1] for i in testing_data])
	test_set = np.asarray(test_set).astype('float32')
	np.save('testing_data.npy', test_set)
	return test_set, gt_test_ids

def custom_convnet(train_data):
	X = np.array([i[0]/255 for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #normalized pixel values
	#Review Numpy multi-dimensional indexing
	y = np.array([i[1] for i in train_data])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
	
	X_train, X_test = np.asarray(X_train).astype('float32'), np.asarray(X_test).astype('float32')
	y_train, y_test = np.asarray(y_train).astype('float32'), np.asarray(y_test).astype('float32')
	
	np.save('X_test.npy', X_test)
	np.save('y_test.npy', y_test)

	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)
	
	
	if os.path.isfile('cat_dog_classifier_model.h5'):
		model = load_model('cat_dog_classifier_model.h5')
		return model
	else:
		print("model: cat_dog_classifier_model not found!")


		model = keras.Sequential([
			layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape = (IMG_SIZE, IMG_SIZE, 1), padding='same'),
			layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
			layers.MaxPool2D(),
			layers.Dropout(rate=0.4),
			
			layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
			layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
			layers.MaxPool2D(),
			layers.Dropout(rate=0.2),
			
			layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
			layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
			layers.MaxPool2D(),
			layers.Dropout(rate=0.2),			
			
			layers.Flatten(),
			layers.Dense(1024, activation='relu'),
			layers.Dense(2, activation='sigmoid')
		])

		model.summary()

		es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)

		model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

		history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test,y_test), callbacks=[es, LearningRateScheduler(lr_time_based_decay, verbose=1)])
		
		history_df = pd.DataFrame(history.history)
		history_df.loc[:, ['loss', 'val_loss']].plot();
		print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
		plt.show()
		
		return model
