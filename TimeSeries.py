import numpy as np
import matplotlib.pyplot as plt 

N = 500
Ox = np.arange(N)*(6/N) -3
Oy = np.cos(2*Ox)

plt.plot(Oy)
plt.show()

import tensorflow as tf 

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.01), loss = "mse")
r = model.fit(Ox, Oy, epochs = 20)

plt.plot(r.history["loss"])
plt.show()

line = np.linspace(-3, 5, 500)
xx = np.meshgrid(line)
prediction = model.predict(xx[0])
plt.plot(xx[0], prediction)
plt.show()

T = 10
data = []
label = []

for t in range(len(Oy) - T):
    x_obj = Oy[t:t+T]
    y_obj = Oy[t+T]
    data.append(x_obj)
    label.append(y_obj)

data = np.array(data)
label = np.array(label)

i = tf.keras.layers.Input(shape = (T,))
x = tf.keras.layers.Dense(1)(i)
model = tf.keras.models.Model(i, x)
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.1), loss = "mse")
r = model.fit(data, label, epochs = 50)

plt.plot(r.history["loss"])
plt.show()

predictions = []
last_x = data[-1]
horizont = 300
while len(predictions) < horizont:
    pred = model.predict(np.expand_dims(last_x, axis = 0))[0][0]
    predictions.append(pred)
    last_x = np.roll(last_x, -1)
    last_x[-1] = pred

plt.plot(Oy)
line2 = np.linspace(N, N+horizont, horizont)
plt.plot(line2, predictions)
plt.show()
