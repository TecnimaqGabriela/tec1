import numpy as np 
import matplotlib.pyplot as plt

number_data = 100

data = 10*np.random.random((number_data, 2))
labels = np.zeros(number_data, dtype = int)

for i in range(number_data):
    if data[i][0] > -1*data[i][1] + 10:
        labels[i] = 1
        plt.scatter(data[i][0], data[i][1], c = "darkgoldenrod")
    else:
        plt.scatter(data[i][0], data[i][1], c = "darkkhaki")
    
plt.show()
    
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2)
    
import tensorflow as tf 

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = (2,)),
    tf.keras.layers.Dense(1, activation = "sigmoid")
])

#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(1, activation = "sigmoid", input_dim = 2))

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
r = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 100)

plt.plot(r.history["loss"], label = "loss")
plt.plot(r.history["val_loss"], label = "val_loss")
plt. show()

print(data[1])
tester = np.zeros(2, dtype = float)
tester[0] = 9.
tester[1] = 1.
tester = np.expand_dims(tester, axis = 0)
print(tester)
print("X_train: ", X_train.shape)
print("Test: ", tester.shape)
prediction = model.evaluate(tester)
print(prediction)
