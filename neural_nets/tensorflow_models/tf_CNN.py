import tensorflow as tf
from tensorflow.keras import regularizers
from neural_nets.data_utils import *
import os
import config


def init_letters_CNN_model():
    num_classes = 26
    input_shape = (28, 28, 1)
    initializer = tf.initializers.VarianceScaling(scale=2.0)
    layers = [

        tf.keras.layers.Conv2D(input_shape=input_shape, filters=32, kernel_size=(3, 3),
                               activation='relu', strides=(1, 1), padding='same',
                               kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                               strides=(1, 1), padding='same',
                               kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                               strides=(1, 1), padding='same',
                               kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu',
                              kernel_initializer=initializer),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_initializer=initializer)

    ]
    model = tf.keras.Sequential(layers)

    return model


def init_digits_CNN_model():
    num_classes = 10
    input_shape = (28, 28, 1)
    initializer = tf.initializers.VarianceScaling(scale=2.0)
    layers = [

        tf.keras.layers.Conv2D(input_shape=input_shape, filters=32, kernel_size=(3, 3),
                               activation='relu', strides=(1, 1), padding='same',
                               kernel_initializer=initializer),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                               strides=(1, 1), padding='same',
                               kernel_initializer=initializer),

        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'),

        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu',
                              kernel_initializer=initializer),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_initializer=initializer)

    ]
    model = tf.keras.Sequential(layers)

    return model


def optimizer_init_fn(learning_rate):
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True, momentum=0.9)
    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return optimizer


def generate_random_hyperparams(lr_min, lr_max, reg_min, reg_max):
    lr = 10 ** np.random.uniform(lr_min, lr_max)
    reg = 10 ** np.random.uniform(reg_min, reg_max)

    return lr, reg


def update_hyper_params(params, lr, reg):
    params["lr"] = lr
    params["reg"] = reg

    return params


x_train, y_train, x_val, y_val, x_test, y_test = get_emnist_letters_data_TF()
x_train_search = x_train[:100]
y_train_search = y_train[:100]
x_val_search = x_val[:100]
y_val_search = y_val[:100]
x_test_search = x_test[:1000]
y_test_search = y_test[:1000]
best_model = None
best_val = 0
best_train = 0
best_params = {}
for i in range(0, 10):
    lr, reg = generate_random_hyperparams(-4, -2, -6, -2)
    optimizer = optimizer_init_fn(lr)
    model = init_letters_CNN_model()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    history = model.fit(x_train_search, y_train_search, batch_size=128, epochs=10, verbose=0,
                        validation_data=(x_val_search, y_val_search))
    train_acc = history.history['sparse_categorical_accuracy'][-1]
    val_acc = history.history['val_sparse_categorical_accuracy'][-1]
    # loss, val_acc = model.evaluate(X_test_search, y_test_search, verbose =0)
    print("-------------------------------------------------------------")
    print("train_accuracy = {0}, val_accuracy={1}".format(train_acc, val_acc))
    print("lr = {0}, reg={1}".format(lr, reg))
    print("-------------------------------------------------------------")
    if train_acc > best_train:
        best_model = model
        best_train = train_acc
        update_hyper_params(best_params, lr, reg)

print(best_params)
print(best_train)
best_model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_val, y_val))
best_model.save(filepath=os.path.join(config.MODELS_DIR, config.letters_CNN_tf))
# model = init_letters_CNN_model()
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.fit(x_train, y_train,
#           batch_size=128,
#           epochs=10,
#           verbose=1,
#           validation_data=(x_val, y_val))
# score = model.evaluate(x_test, y_test, verbose=0)
# print(score)
# model.save(filepath=os.path.join(config.MODELS_DIR, config.letters_CNN_tf))
