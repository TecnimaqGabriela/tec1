import  numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import optimizers, layers
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential

train_number=1000

data = 10*np.random.random((train_number,2))
labels = np.zeros((train_number,1), dtype= int) 

for i in range(train_number):
    if ((data[i][0]-3)**2 + (data[i][1] -3)**2 > 9) and (data[i][0]-4)**2 + (data[i][1] -4)**2 < 16:
        labels[i]=1
        plt.scatter(data[i][0],data[i][1], c='cyan')
    else:
        plt.scatter(data[i][0],data[i][1], c='lightcyan')

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=2))
model.add(Dense(32, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(data, labels, epochs=80, batch_size=32)

test_number=800

unknown= 10*np.random.random((test_number,2))

predictions=model.predict(unknown)

#for i in range(10):
#    print("For {} the prediction is {}".format(unknown[i], predictions[i]))

for i in range(test_number):
    if predictions[i] > 0.6:
        plt.scatter(unknown[i][0],unknown[i][1], c='red')
    else:
        plt.scatter(unknown[i][0],unknown[i][1], c='orchid')

#print(predictions)

plt.show()