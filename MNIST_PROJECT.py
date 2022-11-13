import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_training = pd.read_csv('mnist_train.csv')
df_test = pd.read_csv('mnist_test.csv')

training_dataset_x = df_training.iloc[:, 1:].to_numpy()
training_dataset_y = df_training.iloc[:, 0].to_numpy()


test_dataset_x = df_test.iloc[:, 1:].to_numpy()
test_dataset_y = df_test.iloc[:, 0].to_numpy()


# To be able to see some examples of dataset
plt.figure(figsize=(20,20))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.title(str(training_dataset_y[i]), fontsize=14)
    plt.imshow(training_dataset_x[i].reshape(28,28), cmap='gray')
    
plt.show()

# One hot encoding with the method of to_categorical
from tensorflow.keras.utils import to_categorical

ohe_training_dataset_y = to_categorical(training_dataset_y)
ohe_test_dataset_y = to_categorical(test_dataset_y)

""" The Min-Max scaling for data between [0, 255] can actually be created by dividing 
this pixel data by 255. """

training_dataset_x = training_dataset_x / 255
test_dataset_x = test_dataset_x / 255

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(256, activation='relu', input_dim = 784, name='Hidden-1'))
model.add(Dense(128, activation='relu', name='Hidden-2'))
model.add(Dense(10, activation='softmax', name='Output'))

model.summary()


model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])

hist=model.fit(training_dataset_x, ohe_training_dataset_y, batch_size=32, epochs=20, validation_split=0.2)


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.title('Epoch-Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, 210, 10))

plt.plot(hist.epoch, hist.history['loss'])
plt.plot(hist.epoch, hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()

plt.figure(figsize=(15, 5))
plt.title('Epoch-Categorical Accuracy Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, 210, 10))

plt.plot(hist.epoch, hist.history['categorical_accuracy'])
plt.plot(hist.epoch, hist.history['val_categorical_accuracy'])
plt.legend(['Categorical Accuracy', 'Validation Categorical Accuracy'])
plt.show()

eval_result = model.evaluate(test_dataset_x, ohe_test_dataset_y)

# See the results of evaluating

for i in range(len(eval_result)):
    print(f'{model.metrics_names[i]}: {eval_result[i]}')


"""
A folder named test-images was created in the current directory and similar studies were made in the data set. With these studies, the model was tested.
"""
import glob

for path in glob.glob('test-images/*.jpg'):
    image_data = plt.imread(path)
    gray_scaled_image_data = np.average(image_data, axis=2, weights = [0.3, 0.59, 0.11])
    gray_scaled_image_data = gray_scaled_image_data / 255


    predict_result = model.predict(gray_scaled_image_data.reshape(1, 784))

    result = np.argmax(predict_result)
    plt.imshow(gray_scaled_image_data, cmap='gray')
    plt.show()
    print(f'{path}: {result}')

"""
It has been added to look at the examples in the data set of the numbers that the model did not recognize.For instance 1.
"""
import itertools

for x in itertools.islice(training_dataset_x[training_dataset_y == 1],10):
    plt.imshow(x.reshape(28,28), cmap='gray')
    plt.show()
    