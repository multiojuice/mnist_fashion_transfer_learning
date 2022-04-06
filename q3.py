import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models, losses, optimizers
from tensorflow.keras import layers as L


(x_train,y_train),(x_test,y_test) = datasets.fashion_mnist.load_data()
x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255
x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)
print(x_train.shape)

x_train = tf.image.grayscale_to_rgb(
    x_train,
    name=None
)

x_test = tf.image.grayscale_to_rgb(
    x_test,
    name=None
)

print(x_train.shape)

x_val = x_train[-12000:,:,:,:]
y_val = y_train[-12000:]
x_train = x_train[:-12000,:,:,:]
y_train = y_train[:-12000]


base_model = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(32,32,3),
    pooling=None,
    classes=10,
    classifier_activation="softmax",
)

base_model.trainable = False

fashion_model = models.Sequential()

fashion_model.add(base_model)

fashion_model.add(L.Flatten())
fashion_model.add(L.Dense(128, activation='relu'))
fashion_model.add(L.Dense(10, activation='softmax'))

fashion_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(x_val.shape, y_val.shape)

history = fashion_model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=64,
                    validation_data=(x_val, y_val))


from sklearn.metrics import confusion_matrix
import numpy as np
y_pred = np.argmax(fashion_model.predict(x_test), axis=-1)
matrix = confusion_matrix(y_test, y_pred)
per_class_accuracy = matrix.diagonal() / matrix.sum(axis=1)
# print(per_class_accuracy)

small = min(per_class_accuracy) * 100 - 10

loss_all, acc_all = fashion_model.evaluate(x_test, y_test)
accuracy_data = []

for num, c in enumerate(per_class_accuracy):
    for i in range(round(float(c) * 100 - small)):
        accuracy_data.append(f'#{num}')

for i in range(round(float(acc_all) * 100 - small)):
    accuracy_data.append(f'All')

ax1 = plt.subplot()
ax1.hist(accuracy_data, bins=11, bottom=small, rwidth=0.7)
plt.show()

(x_train_o, y_train_o), (x_test_o, y_test_o) = datasets.fashion_mnist.load_data()

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

fig, axs = plt.subplots(10, 2)
for i in range(10):
    axs[i, 0].imshow(x_test_o[true_array[i]], cmap='gray_r')
    axs[i, 1].imshow(x_test_o[false_array[i]], cmap='gray_r')
plt.show()