import tensorflow as tf
from config import *


class VGG16(tf.keras.Model):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = tf.keras.Sequential()
        # 1
        self.model.add(tf.keras.layers.Conv2D(filters=64,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same',
                                        activation=tf.keras.activations.relu,
                                        input_shape=(image_height, image_width, channels)))
        self.model.add(tf.keras.layers.Conv2D(filters=64,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same',
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                            strides=2,
                                            padding='same'))

        # 2
        self.model.add(tf.keras.layers.Conv2D(filters=128,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same',
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=128,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same',
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                            strides=2,
                                            padding='same'))

        # 3
        self.model.add(tf.keras.layers.Conv2D(filters=256,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same',
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=256,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same',
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=256,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same',
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                            strides=2,
                                            padding='same'))

        # 4
        self.model.add(tf.keras.layers.Conv2D(filters=512,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same',
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=512,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same',
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=512,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same',
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                            strides=2,
                                            padding='same'))

        # 5
        self.model.add(tf.keras.layers.Conv2D(filters=512,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same',
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=512,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same',
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Conv2D(filters=512,
                                        kernel_size=(3, 3),
                                        strides=1,
                                        padding='same',
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                            strides=2,
                                            padding='same'))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(units=4096,
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Dense(units=4096,
                                        activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Dense(units=NUM_CLASSES,
                                        activation=tf.keras.activations.softmax))

    def call(self, inputs, training=None):
        return self.model(inputs)

def vgg_16():
    return VGG16()