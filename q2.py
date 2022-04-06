import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses

pre_trained_model = tf.keras.models.load_model('mnsit_trained_model')
pre_trained_model.trainable = False

print(pre_trained_model.summary())

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]]) / 255
x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

x_val = x_train[-2000:, :, :, :]
y_val = y_train[-2000:]
x_train = x_train[:-2000, :, :, :]
y_train = y_train[:-2000]

loss0, accuracy0 = pre_trained_model.evaluate(x_test, y_test)
print(loss0, accuracy0)

history = pre_trained_model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_val, y_val))

fig, axs = plt.subplots(2, 1, figsize=(15, 15))

axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].legend(['Train', 'Val'])

axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].legend(['Train', 'Val'])

plt.show()

loss0, accuracy0 = pre_trained_model.evaluate(x_test, y_test)
print(loss0, accuracy0)

model = models.Sequential()
model.add(layers.Conv2D(6, 5, activation='relu', input_shape=x_train.shape[1:]))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(16, 5, activation='relu'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(120, 5, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='sgd', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=40, validation_data=(x_val, y_val))
fig, axs = plt.subplots(2, 1, figsize=(15, 15))

axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].legend(['Train', 'Val'])

axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].legend(['Train', 'Val'])

loss0, accuracy0 = model.evaluate(x_test, y_test)
print(loss0, accuracy0)
plt.show()

"""
Get metrics nad plots
"""
import numpy as np
from sklearn.metrics import confusion_matrix

y_pred = np.argmax(pre_trained_model.predict(x_test), axis=-1)
matrix = confusion_matrix(y_test, y_pred)
per_class_accuracy = matrix.diagonal() / matrix.sum(axis=1)

small = min(per_class_accuracy) * 100 - 10
loss_all, acc_all = model.evaluate(x_test, y_test)
accuracy_data = []

for num, c in enumerate(per_class_accuracy):
    for i in range(round(float(c) * 100 - small)):
        accuracy_data.append(f'#{num}')

for i in range(round(float(acc_all) * 100 - small)):
    accuracy_data.append(f'All')

ax1 = plt.subplot()
ax1.hist(accuracy_data, bins=11, bottom=small, rwidth=0.7)
plt.show()

(x_train_o,y_train_o),(x_test_o,y_test_o) = datasets.fashion_mnist.load_data()

ax2 = plt.subplot()
ax2.imshow(x_train_o[3], cmap='gray_r')
plt.show()

true_array = []
false_array = []

for i in range(10):
  true_found = False
  false_found = False
  for j in range(len(y_pred)):
    # It is the number we are currently on
    if y_test[j] == i:
      if not true_found and y_test[j] == y_pred[j]:
        true_array.append(j)
        true_found = True

      if not false_found and y_test[j] != y_pred[j]:
        false_array.append(j)
        false_found = True


fig , axs = plt.subplots(10, 2)
for i in range(10):
  axs[i, 0].imshow(x_test_o[true_array[i]], cmap='gray_r')
  axs[i, 1].imshow(x_test_o[false_array[i]], cmap='gray_r')

pre_trained_model.save('mnsit_fashion_pretrained_model')
plt.show()