import numpy as np
from neural_nets.layers import *
from neural_nets.layers_utils import *


class TwoLayerNet(object):
    """
    Двухслойная полносвязанная нейронная сеть с нелинейностью ReLU и
    softmax loss, которая использует модульные слои. Полагаем, что размер входа
    - D, размер скрытого слоя - H, классификация выполняется по C классам .

     Архитектура:  affine - relu - affine - softmax.

     Обратите внимание, что этот класс не реализует градиентный спуск; вместо этого
     он будет взаимодействовать с отдельным объектом Solver, который отвечает
     за выполнение оптимизации.

     Обучаемые параметры модели хранятся в словаре
     self.params, который связывает имена параметров и массивы numpy.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Инициализирует сеть.

         Входы:
         - input_dim: целое число, задающее размер входа
         - hidden_dim: целое число, задающее размер скрытого слоя
         - num_classes: целое число, указывающее количество классов
         - weight_scale: скаляр, задающий стандартное отклонение при
           инициализация весов случайными числами.
         - reg: скаляр, задающий коэффициент регуляции L2.
        """
        self.params = {}
        self.reg = reg
        self.val_accuracy = None
        self.train_accuracy = None
        self.test_accuracy = None
        ############################################################################
        # ЗАДАНИЕ: Инициализировать веса и смещения двухслойной сети. Веса         #
        # должны инициализироваться с нормальным законом с центром в 0,0 и со      #
        # стандартным отклонением, равным weight_scale,смещения должны быть        #
        # инициализированы нулем. Все веса и смещения должны храниться в           #
        # словаре self.params при использовании обозначений: весов  и смещений     #
        # первого слоя - «W1» и «b1» ,  весов и смещений второго слоя - «W2» и «b2»#
        ############################################################################
        self.params["W1"] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))
        self.params["W2"] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))

        # self.params["W1"] = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0/input_dim)
        # self.params["W2"] = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0/hidden_dim)
        self.params["b1"] = np.zeros(hidden_dim, dtype=float)
        self.params["b2"] = np.zeros(num_classes, dtype=float)
        ############################################################################
        #                             КОНЕЦ ВАШЕГО КОДА                            #
        ############################################################################

    def loss(self, X, y=None):
        """
        Вычисляет потери и градиент на мини-блоке данных.

         Входы:
         - X: массив входных данных формы (N, d_1, ..., d_k)
         - y: массив меток формы (N,). y [i] дает метку для X [i].

         Возвращает:
         Если y - None, то запускает тестовый режим прямого прохода модели и возвращает:
         - scores: массив формы (N, C), содержащий рейтинги классов, где
           scores[i, c] - рейтинг принадлежности примера X [i] к классу c.

         Если y не  None, то запускает режим обучения с прямым и обратным распространением
         и возвращает кортеж из:
         - loss: cкалярное значение потерь
         - grads: словарь с теми же ключами, что и self.params, связывающий имена
           градиентов по параметрам со значениями градиентов.

        """
        out1, cache1 = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        out2, cache2 = affine_forward(out1, self.params["W2"], self.params["b2"])
        scores = out2

        # Если y - None, мы находимся в тестовом режиме, поэтому просто возвращаем scores
        if y is None:
            return scores

        grads = {}
        loss, dscores = softmax_loss(scores, y)
        dout1, grads["W2"], grads["b2"] = affine_backward(dscores, cache2)
        dout2, grads["W1"], grads["b1"] = affine_relu_backward(dout1, cache1)

        for w in ["W2", "W1"]:
            if self.reg > 0:
                loss += 0.5 * self.reg * (self.params[w] ** 2).sum()
            grads[w] += self.reg * self.params[w]

        return loss, grads
