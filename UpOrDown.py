import numpy as np 
import matplotlib.pyplot as plt 

No = 200
train_data = np.zeros(No, dtype = int)
for i in range(No):
    if i == 0:
        n = 0
    else:
        n = i%20
    #print("i: ", i)
    #print("20%: ", n)
    if n > 10:
        train_data[i] = n - 10
    else:
        train_data[i] = 10 - n
    plt.scatter(i, train_data[i])
plt.show()

T = 4
timeseries_data = []
labels = []
count = 0
for t in range(No - T):
    count = count +1
    x_obj = train_data[t:t+T]
    if count < 10:
        y_obj = 0
    elif count < 19:
        y_obj = 1
    elif count == 20:
        y_obj = 1
        count = 0
    timeseries_data.append(x_obj)
    labels.append(y_obj)

timeseries_data = np.array(timeseries_data)
labels = np.array(labels)
timeseries_data = timeseries_data/10

import tensorflow as tf

i = tf.keras.layers.Input(shape = (T,))
x = tf.keras.layers.Dense(1, activation = "sigmoid")(i)
model = tf.keras.models.Model(i, x)
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.1), loss = "binary_crossentropy", metrics = ["accuracy"])
r = model.fit(timeseries_data, labels, epochs = 50)

plt.plot(r.history["loss"])
plt.show()

last_x = timeseries_data[-1]
z = last_x[-1]
horizont = 50
test_data = []
for i in range(horizont):
    if i == 0:
        n = z
    else:
        n = i%20 + z
    if n > 10:
        test_data.append(n - 10)
        plt.scatter(i, n-10)
    else:
        test_data.append(10 - n)
        plt.scatter(i, 10-n)
plt.title("test_data")
plt.show()

T = 4
test_timeseries = []
labels = []
count = 0
for t in range(horizont - T):
    x_obj = test_data[t:t+T]
    test_timeseries.append(x_obj)
test_timeseries = np.array(test_timeseries)
print(test_timeseries)
test_timeseries = test_timeseries/10

predictions = []
for i in range(len(test_timeseries)):
    pred = model.predict(np.expand_dims(test_timeseries[i], axis = 0))[0][0]
    predictions.append(pred)
    print("Time object: ", test_timeseries[i]*10)
    if pred < 0.5:
        print("prediction: Down(", pred,")")
    else:
        print("prediction: Up(",pred,")")
