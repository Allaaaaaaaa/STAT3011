from __future__ import absolute_import, division, print_function
import tensorflow as tf
from models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
import config
from prepare_data import generate_datasets
import math
from models.vgg import vgg_16
from models.alexnet import alexnet
import matplotlib.pyplot as plt


def get_model():
    model = resnet_50()
    if config.model == "resnet18":
        model = resnet_18()
    if config.model == "resnet34":
        model = resnet_34()
    if config.model == "resnet101":
        model = resnet_101()
    if config.model == "resnet152":
        model = resnet_152()
    if config.model == "vgg16":
        model = vgg_16()
    if config.model == "alexnet":
        model = alexnet()
    model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    model.summary()
    return model


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    # get the original_dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()


    # create model
    model = get_model()

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # optimizer = tf.keras.optimizers.Adadelta()
    # lr = 0.01
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    #                                      name='Adam', )
    optimizer = tf.keras.optimizers.Adam()


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def valid_step(images, labels):
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)

    train_acc_hist = []
    valid_acc_hist = []
    train_loss_hist = []
    valid_loss_hist = []

    # start training
    for epoch in range(config.EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0
        for images, labels in train_dataset:
            step += 1
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                     config.EPOCHS,
                                                                                     step,
                                                                                     math.ceil(train_count / config.BATCH_SIZE),
                                                                                     train_loss.result(),
                                                                                     train_accuracy.result()))

        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)

        train_acc_hist.append(train_accuracy.result())
        valid_acc_hist.append(valid_accuracy.result())
        train_loss_hist.append(train_loss.result())
        valid_loss_hist.append(valid_loss.result())

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  config.EPOCHS,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))

    epochs_range = range(config.EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, train_acc_hist, label='Training Accuracy')
    plt.plot(epochs_range, valid_acc_hist, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig('plot/acc.png')

    plt.figure(figsize=(8, 8))
    plt.plot(epochs_range, train_loss_hist, label='Training Loss')
    plt.plot(epochs_range, valid_loss_hist, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('plot/loss.png')

    model.save_weights(filepath=config.save_model_dir, save_format='tf')
