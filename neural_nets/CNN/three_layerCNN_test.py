import pickle

from neural_nets.CNN.three_layerCNN import ThreeLayerConvNet
import numpy as np
import matplotlib.pyplot as plt
from neural_nets.solver import Solver
import os


def update_hyper_params(params, lr, reg, batch_size, weight_scale, update_rule, num_epochs, dropout):
    params["lr"] = lr
    params["reg"] = reg
    params["batch_size"] = batch_size
    params["weight_scale"] = weight_scale
    params["update_rule"] = update_rule
    params["num_epochs"] = num_epochs
    params["dropout"] = dropout


def generate_random_hyperparams(lr_min, lr_max, reg_min, reg_max, wscale_min, wscale_max):
    lr = 10 ** np.random.uniform(lr_min, lr_max)
    reg = 10 ** np.random.uniform(reg_min, reg_max)
    weight_scale = 10 ** np.random.uniform(wscale_min, wscale_max)

    return lr, reg, weight_scale,


def print_model_params(best_solver, best_params):
    # Отображение функции потерь и точности обучения/валидации
    plt.subplot(2, 1, 1)
    plt.plot(best_solver.loss_history)
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(best_solver.train_acc_history, label='train')
    plt.plot(best_solver.val_acc_history, label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()
    plt.show()
    print("best_params: ")
    print(best_params)


def find_best_letters_three_layer_CNN_model(data):
    best_model = None
    best_val = 0
    best_test = 0
    best_solver = None
    best_params = {}
    input_dim = data['X_train'].shape[1]
    num_classes = 26
    update_rule = 'adam'
    batch_size = 50
    num_epochs = 10
    normalization = 'batchnorm'
    dropout = 1

    for i in range(0, 15):
        lr, reg, weight_scale = generate_random_hyperparams(-3, -2, -9, -8, -2, -1)
        net_model = ThreeLayerConvNet((1, 28, 28), num_filters=32, filter_size=7, hidden_dim=100,
                                      weight_scale=weight_scale, num_classes=num_classes, reg=reg)

        solver = Solver(net_model, data, num_epochs=num_epochs, batch_size=batch_size,
                        update_rule=update_rule, optim_config={'learning_rate': lr},
                        verbose=True, print_every=100)

        solver.train()

        val_accuracy = solver.val_acc_history[-1]
        train_accuracy = solver.train_acc_history[-1]
        y_test_pred = np.argmax(net_model.loss(data['X_test']), axis=1)
        test_accuracy = np.mean(y_test_pred == data['y_test'])

        if val_accuracy > best_val and test_accuracy > best_test:
            best_val = val_accuracy
            best_model = net_model
            best_model.val_accuracy = val_accuracy
            best_model.train_accuracy = train_accuracy
            best_solver = solver
            best_test = test_accuracy
            update_hyper_params(best_params, lr, reg, batch_size, weight_scale, update_rule, num_epochs,
                                dropout=dropout)

        print("-------------------------------------------------------------")
        print("train_accuracy = {0}, val_accuracy={1}".format(train_accuracy, val_accuracy))
        print("lr = {0}, reg={1}, best_val = {2}".format(lr, reg, best_val))
        print("batch_size = {0}, weight_scale={1}, dropout = {2}".format(batch_size, weight_scale, dropout))
        print('Test set accuracy: ', test_accuracy)
        print("-------------------------------------------------------------")

    # Отображение функции потерь и точности обучения/валидации
    # print_model_params(best_solver, best_params)

    y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
    best_model.test_accuracy = np.mean(y_test_pred == data['y_test'])

    return best_model
