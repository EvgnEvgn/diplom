import mnist
from neural_nets.TwoLayerNet.two_layer_net import TwoLayerNet
import numpy as np
import matplotlib.pyplot as plt
from neural_nets.solver import Solver


def get_mnist_data(num_training=59000, num_validation=1000, num_test=10000):
    x_train = mnist.train_images()
    y_train = mnist.train_labels()
    x_test = mnist.test_images()
    y_test = mnist.test_labels()

    # Выборка данных
    mask = list(range(num_training, num_training + num_validation))
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    x_test = x_test[mask]
    y_test = y_test[mask]

    # Вытягивание матриц в векторы
    x_train = x_train.reshape(num_training, -1)
    x_val = x_val.reshape(num_validation, -1)
    x_test = x_test.reshape(num_test, -1)

    return x_train, y_train, x_val, y_val, x_test, y_test


def update_hyper_params(params, lr, reg, batch_size, weight_scale, update_rule, num_epochs):
    params["lr"] = lr
    params["reg"] = reg
    params["batch_size"] = batch_size
    params["weight_scale"] = weight_scale
    params["update_rule"] = update_rule
    params["num_epochs"] = num_epochs


def generate_random_hyperparams(lr_min, lr_max, reg_min, reg_max, wscale_min, wscale_max):
    lr = 10 ** np.random.uniform(lr_min, lr_max)
    reg = 10 ** np.random.uniform(reg_min, reg_max)
    weight_scale = 10 ** np.random.uniform(wscale_min, wscale_max)

    return lr, reg, weight_scale,


def find_best_net():
    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_data()
    data = {
        'X_train': x_train,
        'y_train': y_train,
        'X_val': x_val,
        'y_val': y_val,
        'X_test': x_test,
        'y_test': y_test
    }
    best_model = None
    best_val = 0
    best_solver = None
    best_params = {}
    learning_rates = [1e-3, 5e-3, 7e-3, 2e-4, 6e-4, 8e-4, 3e-5]
    regularization_strengths = [0.25, 0.5, 0.7, 0.9, 1.2, 1.4]
    input_size = 28 * 28
    hidden_size = 100
    num_classes = 10
    # weight_scale = 1e-2
    update_rule = 'adam'
    batch_size = 200
    num_epochs = 2
    for i in range(0, 2):
        lr, reg, weight_scale = generate_random_hyperparams(-5, -2, -3, -1, -7, -6)
        net_model = TwoLayerNet(input_size, hidden_size, num_classes)

        solver = Solver(net_model, data, num_epochs=num_epochs, batch_size=batch_size,
                        update_rule=update_rule, optim_config={'learning_rate': lr},
                        verbose=True, print_every=50)

        solver.train()

        val_accuracy = solver.val_acc_history[-1]
        train_accuracy = solver.train_acc_history[-1]

        if val_accuracy > best_val:
            best_val = val_accuracy
            best_model = net_model
            best_solver = solver
            update_hyper_params(best_params, lr, reg, batch_size, weight_scale, update_rule, num_epochs)

        print("-------------------------------------------------------------")
        print("train_accuracy = {0}, val_accuracy={1}".format(train_accuracy, val_accuracy))
        print("lr = {0}, reg={1}, best_val = {2}, weight_scale = {3}".format(lr, reg, best_val, weight_scale))
        print("train_accuracy = {0}, val_accuracy={1}".format(train_accuracy, val_accuracy))
        print("-------------------------------------------------------------")
    # Отображение функции потерь и точности обучения/валидации
    # plt.subplot(2, 1, 1)
    # plt.plot(best_solver.loss_history)
    # plt.title('Loss history')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.subplot(2, 1, 2)
    # plt.plot(best_solver.train_acc_history, label='train')
    # plt.plot(best_solver.val_acc_history, label='val')
    # plt.title('Classification accuracy history')
    # plt.xlabel('Epoch')
    # plt.ylabel('Clasification accuracy')
    # plt.legend()
    # plt.show()
    # print("best_params: ")
    # print(best_params)
    # y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
    # y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
    # print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
    # print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())

    return best_model
