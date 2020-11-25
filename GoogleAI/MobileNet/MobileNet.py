import numpy as np  # linear algebra
import keras.preprocessing.image as image
import os
from time import time
from keras.preprocessing.image import *
from keras.utils import np_utils
import json
from PIL import Image
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications import MobileNetV2
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

data_dir = "./DataSet/"

data_train_path = data_dir + 'train'
data_valid_path = data_dir + 'validation'
data_test_path = data_dir + 'test'

cat_to_name = data_dir + "cat_to_name.json"

print(os.listdir(data_dir))

with open(cat_to_name, 'r') as json_file:
    cat_2_name = json.load(json_file)

batch_size = 60

# Transforms
datagen_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.1,  # randomly shift images horizontally
    height_shift_range=0.1,  # randomly shift images vertically
    horizontal_flip=True,
    featurewise_std_normalization=True,
    samplewise_std_normalization=True)

datagen_valid = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.1,  # randomly shift images horizontally
    height_shift_range=0.1,  # randomly shift images vertically
    horizontal_flip=True,
    featurewise_std_normalization=True,
    samplewise_std_normalization=True)

datagen_test = ImageDataGenerator(
    featurewise_std_normalization=True,
    samplewise_std_normalization=True,
    rescale=1. / 255)

train_generator = datagen_train.flow_from_directory(
    data_train_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

valid_generator = datagen_valid.flow_from_directory(
    data_valid_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = datagen_test.flow_from_directory(
    data_test_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

# Lets have a look at some of our images
images, labels = train_generator.next()

fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(wspace=0.2, hspace=0.4)

# Lets show the first 32 images of a batch
for i, img in enumerate(images[:32]):
    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(img.astype('uint8'))
    image_idx = np.argmax(labels[i])

int_to_dir = {v: k for k, v in train_generator.class_indices.items()}

input_tensor = Input(shape=(224, 224, 3))
base_model = MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_tensor=input_tensor,
    input_shape=(224, 224, 3),
    pooling='avg')

for layer in base_model.layers:
    layer.trainable = True  # trainable has to be false in order to freeze the layers

x = Dense(512, activation='relu')(base_model.output)
x = Dropout(.8)(x)

predictions = Dense(211, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])

num_train = len(train_generator.filenames)
num_valid = len(valid_generator.filenames)
num_test = len(train_generator.filenames)

# When to save the model
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1,
                               save_best_only=True)

# Reduce learning rate when loss doesn't improve after n epochs
scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=1e-8, verbose=1)

# Stop early if model doesn't improve after n epochs
early_stopper = EarlyStopping(monitor='val_loss', patience=12,
                              verbose=0, restore_best_weights=True)

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=num_train // batch_size,
                    epochs=100,
                    verbose=1,
                    callbacks=[checkpointer, scheduler, early_stopper],
                    validation_data=valid_generator,
                    validation_steps=num_valid // batch_size)

# model.load_weights('../input/mobilenetv2-weights/model.weights.best.hdf5')
model.save('mobilenet.h5')

# Setup TensorBoard callback.
JOB_DIR = os.getenv('JOB_DIR')
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    os.path.join(JOB_DIR, 'keras_tensorboard'),
    histogram_freq=1)


# score = model.evaluate_generator(test_generator, steps=num_test//1, verbose=1)
# print('\n', 'Test accuracy:', score[1])



