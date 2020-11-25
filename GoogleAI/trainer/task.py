from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import *
import os
import tensorflow as tf
import argparse

from . import model


def get_args():
    """Argument parser.
    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args


def train_and_evaluate(args):
    """Trains and evaluates the Keras model.
    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.
    Args:
      args: dictionary of arguments - see get_args() for details
    """

    data_dir = "../DataSet"
    data_train_path = data_dir + "/train"
    data_valid_path = data_dir + "/validation"
    data_test_path = data_dir + "/TestSet"
    batch_size = 60

    # Create the Keras Model
    keras_model = model.create_keras_model()

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

    # When to save the model
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1,
                                   save_best_only=True)

    # Reduce learning rate when loss doesn't improve after n epochs
    scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=5, min_lr=1e-8, verbose=1)

    # Stop early if model doesn't improve after n epochs
    early_stopper = EarlyStopping(monitor='val_loss', patience=12,
                                  verbose=0, restore_best_weights=True)
    # Setup TensorBoard callback.
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        os.path.join(args.job_dir, 'keras_tensorboard'),
        histogram_freq=1)

    num_train = len(train_generator.filenames)
    num_valid = len(valid_generator.filenames)


    # Train the model
    history = keras_model.fit(train_generator,
                              steps_per_epoch=1,
                              epochs=1,
                              verbose=1,
                              callbacks=[checkpointer, scheduler, early_stopper, tensorboard_cb],
                              validation_data=valid_generator,
                              validation_steps=num_valid // batch_size)

    export_path = os.path.join(args.job_dir, 'keras_export')
    keras_model.save(export_path)
    print('Model exported to: {}'.format(export_path))


if __name__ == '__main__':
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
