import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from neural_nets.data_utils import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import LearningRateScheduler
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
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'),

        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',
                               strides=(1, 1), padding='same',
                               kernel_initializer=initializer),
        # tf.keras.layers.BatchNormalization(),

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


def init_letters_CNN_model_2():
    num_classes = 26
    input_shape = (28, 28, 1)
    initializer = tf.initializers.VarianceScaling(scale=2.0)
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))

    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size=4, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def print_learning_curves(loss_history, train_acc_history, val_acc_history):
    # Отображение функции потерь и точности обучения/валидации
    plt.subplot(2, 1, 1)
    plt.plot(loss_history)
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(train_acc_history, label='train')
    plt.plot(val_acc_history, label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()
    plt.show()


def optimizer_init_fn(learning_rate):
    # optimizer = tf.keras.optimizers.SGD(nesterov=True, momentum=0.9)
    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.Adam()

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

datagen = ImageDataGenerator(rotation_range=10,
                             zoom_range=0.10,
                             width_shift_range=0.1,
                             height_shift_range=0.1)

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

lr, reg = generate_random_hyperparams(-4, -2, -6, -2)
optimizer = optimizer_init_fn(lr)
model = init_letters_CNN_model_2()
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
train_generated_data = datagen.flow(x_train, y_train, batch_size=64)
history = model.fit_generator(train_generated_data,
                              epochs=10, steps_per_epoch=x_train.shape[0] // 64,
                              validation_data=(x_val, y_val),
                              callbacks=[annealer])
loss_history = history.history['loss']
print_learning_curves(loss_history, history.history['sparse_categorical_accuracy'], history.history['val_sparse_categorical_accuracy'])
score = model.evaluate(x_test, y_test, verbose=0)
print("Score:", score)
#model.save(os.path.join(config.MODELS_DIR, config.letters_CNN_DataGen_tf))
