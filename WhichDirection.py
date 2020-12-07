import numpy as np
import matplotlib.pyplot as plt 

No = 200
train_data = np.zeros((No,2), dtype = int)
for i in range(No):
    n = i%40
    if n < 10:
        train_data[i][1] = n
    elif n < 20:
        train_data[i][0] = n - 10
        train_data[i][1] = 10
    elif n < 30:
        train_data[i][0] = 10
        train_data[i][1] = 10 - (n - 20)
    else:
        train_data[i][0] = 10 - (n - 30)

T = 4
timeseries_data = []
labels = []
count = 0
for t in range(No - T):
    count = count + 1
    x_obj = train_data[t:t+T]
    if count < 10:
        y_obj = [1,0,0,0]
    elif count < 20:
        y_obj = [0,1,0,0]
    elif count < 30:
        y_obj = [0,0,1,0]
    elif count < 40:
        y_obj = [0,0,0,1]
    elif count == 40:
        y_obj = [1,0,0,0]
        count = 0
    timeseries_data.append(x_obj)
    labels.append(y_obj)
timeseries_data = np.array(timeseries_data)
labels = np.array(labels)
print(timeseries_data[0])
print(timeseries_data[0].shape)
timeseries_data = timeseries_data/10

import tensorflow as tf 

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(6, input_shape = (T,2),
    activation = "relu"))
model.add(tf.keras.layers.Dense(4, activation = "softmax"))
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.01),
    loss = "categorical_crossentropy",
    metrics = ["categorical_accuracy"])
r = model.fit(timeseries_data, labels, epochs = 50)

plt.plot(r.history["loss"])
plt.show()

print(model.layers[0].get_weights())
print(model.layers[1].get_weights())

scores = model.evaluate(timeseries_data, labels)

horizont = 50
test_data = np.zeros((horizont, 2), dtype = int)
for i in range(horizont):
    n = i%40
    if n < 10:
        test_data[i][0] = n
        test_data[i][1] = 10
    elif n < 20:
        test_data[i][0] = 10
        test_data[i][1] = 10 - (n - 10)
    elif n < 30:
        test_data[i][0] = 10 - (n - 20)
    else:
        test_data[i][1] = n - 30

test_timseries = []
for t in range(horizont - T):
    x_obj = test_data[t:t+T]
    test_timseries.append(x_obj)
test_timseries = np.array(test_timseries)
test_timseries = test_timseries/10

print("Training accuracy: {}%".format(int(scores[1]*100)))

predictions = []
for i in range(len(test_timseries)):
    pred = model.predict(np.expand_dims(test_timseries[i],
        axis = 0))[0]
    predictions.append(pred)
    for t in range(T):
        plt.scatter(test_timseries[i][t][0]*10, test_timseries[i][t][1]*10)
    if np.argmax(pred) == 0:
        plt.title("Up")
    if np.argmax(pred) == 1:
        plt.title("Right")
    if np.argmax(pred) == 2:
        plt.title("Down")
    if np.argmax(pred) == 3:
        plt.title("Left")
    print(pred)
    # plt.title(str(pred))
    plt.show()



