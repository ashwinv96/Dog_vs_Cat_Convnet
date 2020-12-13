# Dog vs Cat Custom Keras Convnet
Using a custom convolutional neural network (CNN) created with Keras to solve the binary classification problem of determining if an image is that of a cat or dog. The input dataset is provided by the Dogs vs Cats competition from Kaggle. 

## Clone repo and download dataset and model file
Use git clone to clone the repo into your local machine. Download the model file and video sample at https://drive.google.com/drive/folders/1NzylL0iEFiPLxi4_r3yxbvhtNABNaGZQ?usp=sharing. Extract and place all items in the root of your project directory.
To run the demo please enter the following command: 
```python
python3 main.py \
```
If the trained weights (model file) are used then the algorithm will skip the process of training. After the first time running the algorithm, the image datasets (train set and test set) are placed in NumPy arrays once and won't have to be created from scratch again. The algorithm will make predictions with the images in the <strong>test1</strong> directory. How many images to make predictions on (and plot) can be specified by adding the lines: 
```python
import convnet as cnn \
cnn.predict_and_plot(test_set, images_to_test, f1, f2, model) # f1, f2 are the factor pair of (images_to_test) for arranging the matplotlib plot. f1 * f1 = images_to_test
```

## Visualize Output Predictions
To visualize our network's performance, we plot 36 predictions made on the test set images. We can see three missed detections here. 
<p align="center">
  <img src="https://github.com/ashwinv96/Dog_vs_Cat_Detector/blob/master/dog_cat_w_errors.png?raw=true">
</p>

## CNN Network Structure
