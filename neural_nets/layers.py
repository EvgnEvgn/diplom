from builtins import range
import numpy as np
from operator import itemgetter
from itertools import product


def affine_forward(x, w, b):
    """
    Выполняет прямое распространение значений для аффинного (полносвязанного) слоя.

    Входной сигнал x имеет форму (N, d_1, ..., d_k) и представляет мини-блок из N
    примеров, где каждый пример x [i] имеет форму (d_1, ..., d_k). Функция
    преобразует каждый вход в вектор размерности D = d_1 * ... * d_k и
    затем вычмсляет выходной вектор размерности M.

     Входы:
     - x: массив numpy, содержащий входные данные, формы (N, d_1, ..., d_k)
     - w: Множество весовых коэффициентов, формы (D, M)
     - b: Массив смещений, формы (M,)

     Возвращает кортеж из:
     - out: выход, формы (N, M)
     - cache: (x, w, b)
    """
    x_reshaped = x.reshape(x.shape[0], -1)
    out = x_reshaped.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """

    Выполняет обратный проход для аффинного слоя.

     Входы:
     - dout: восходящая производная, форма (N, M)
     - cache: кортеж:
       - x: входные данные формы (N, d_1, ... d_k)
       - w: веса формы (D, M)
       - b: смещения формы (M,)

     Возвращает кортеж:
     - dx: градиент по x, форма (N, d1, ..., d_k)
     - dw: градиент по w, форма (D, M)
     - db: градиент по отношению к b, форма (M,)
    """
    x, w, b = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = dout.sum(axis=0)
    return dx, dw, db


def relu_forward(x):
    """
    Выполняет прямое распространение для слоя блоков ReLU.

     Входные данные:
     - x: входы любой формы

     Возвращает кортеж:
     - out: выход, такой же формы, как x
     - cache: x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
     Выполняет обратный проход для слоя из блоков ReLU.

     Входные данные:
     - dout: восходящие производные
     - cache: вход x, такой же формы, как dout

     Возвращает:
     - dx: градиент по x

    """
    x = cache
    dx = dout
    dx[x <= 0] = 0
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
