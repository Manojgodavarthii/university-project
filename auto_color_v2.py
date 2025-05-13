# Importing Libraries
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, array_to_img, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load training data
X = []
for imagename in os.listdir("Dataset/Train/"):
    X.append(img_to_array(load_img("Dataset/Train/" + imagename)))
X = np.array(X, dtype=float)
Xtrain = X[: int(0.95 * len(X))]

# Define pretraining model architecture
pretrain_model = Sequential()
pretrain_model.add(InputLayer(input_shape=(256, 256, 1)))
pretrain_model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
pretrain_model.add(Conv2D(128, (3, 3), activation="relu", strides=2, padding="same"))
pretrain_model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
pretrain_model.add(Conv2D(256, (3, 3), activation="relu", strides=2, padding="same"))
pretrain_model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
pretrain_model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
pretrain_model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
pretrain_model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
pretrain_model.add(UpSampling2D((2, 2)))
pretrain_model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
pretrain_model.add(UpSampling2D((2, 2)))
pretrain_model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
pretrain_model.add(Conv2D(2, (3, 3), activation="tanh", padding="same"))
pretrain_model.add(UpSampling2D((2, 2)))

# Compile pretraining model
pretrain_model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])

# Image transformer for data augmentation
datagen = ImageDataGenerator(
    shear_range=0.2, zoom_range=0.2, rotation_range=20, horizontal_flip=True
)

# Pretrain the model with grayscale images
batch_size = 10


def grayscale_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        X_batch = rgb2gray(batch)
        yield (X_batch.reshape(X_batch.shape + (1,)), X_batch)


pretrain_model.fit_generator(grayscale_gen(batch_size), epochs=50, steps_per_epoch=30)

# Remove the last layer from the pretraining model
pretrain_model.pop()

# Define the final model architecture
final_model = Sequential()
final_model.add(InputLayer(input_shape=(256, 256, 1)))
final_model.add(pretrain_model)
final_model.add(Conv2D(2, (1, 1), activation="sigmoid"))
final_model.add(UpSampling2D((2, 2)))

# Compile the final model
final_model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
