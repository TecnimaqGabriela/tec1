import json
import numpy as np 
import skimage
from skimage import io
import matplotlib.pyplot as plt 
import time

with open('/home/tecnimaq/Gabriela/TF-SimpleHumanPose/output/result/Working_1/result.json') as jsonin1:
    inside_1 = json.load(jsonin1)

with open('/home/tecnimaq/Gabriela/TF-SimpleHumanPose/output/result/ch02_20200119130232-imagenes/result.json') as jsonout1:
    outside_1 = json.load(jsonout1)

dimension = 4 
amount = len(inside_1) + len(outside_1)
keypoint_dimension = len(inside_1[0].get('keypoints'))
keypoints = np.zeros((amount, keypoint_dimension), dtype = float)
data = np.zeros((amount, dimension), dtype = float)
label = np.zeros(amount, dtype = int)

for obj in range(amount):

    if obj < len(inside_1):

        keypoints[obj] = inside_1[obj].get('keypoints')
        label[obj] = 1

    else:

        keypoints[obj] = outside_1[obj - len(inside_1)].get('keypoints')
        label[obj] = 0

    data[obj][0] = keypoints[obj][15]
    data[obj][1] = keypoints[obj][16]
    data[obj][2] = keypoints[obj][33]
    data[obj][3] = keypoints[obj][34]
    
data_min = data.min()
data_max = data.max()
normalized_data = (data - data_min)/(data_max - data_min)

import sklearn
from sklearn.model_selection import train_test_split
    
X_train, X_test, y_train, y_test = train_test_split(normalized_data, label, test_size = 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = tf.keras.Sequential()
model.add(Dense(8, activation = 'relu', input_dim = dimension))
model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(4, activation = 'tanh'))
#model.add(tf.keras.layers.Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 10, validation_data = (X_val, y_val))

#scores = model.evaluate(X_train, y_train)
#print('Training Accuracy:{}%'.format(int(scores[1]*100)))
#scores = model.evaluate(X_test, y_test)
#print('Testing Accuracy:{}%'.format(int(scores[1]*100)))

test_dataset = 'corrupted_set_ch02_20200119080225'
#test_dataset = 'Working_1'
result_path = '/home/tecnimaq/Gabriela/TF-SimpleHumanPose/output/result/'+test_dataset+'/result.json'

with open(result_path) as jsontest:
    test = json.load(jsontest)

test_amount = len(test)
test_data = np.zeros((amount, dimension), dtype = float)
test_keypoints = np.zeros((amount, keypoint_dimension), dtype = float)

for obj in range(test_amount):

    test_keypoints[obj] = test[obj].get('keypoints')

    test_data[obj][0] = test_keypoints[obj][15]
    test_data[obj][1] = test_keypoints[obj][16]
    test_data[obj][2] = test_keypoints[obj][33]
    test_data[obj][3] = test_keypoints[obj][34]

test_mean = test_data.mean()
normalized_test = test_data - test_mean
prediction = model.predict(normalized_test)

prediction_mean = prediction.mean()

if prediction_mean > 0.5:
    print('Prediction: ',prediction,', Inside')
else:
    print('Prediction: ',prediction,', Outside')
