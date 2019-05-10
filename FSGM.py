from keras.models import load_model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

model = load_model('models/baseline_cnn.h5')

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

sample_input = x_test[100].reshape(1, img_rows, img_cols, 1)

plt.imshow(np.squeeze(sample_input))