import numpy as np

from neural_nets.layers import *
from neural_nets.layers_utils import *


class ThreeLayerConvNet(object):
    """
    Трехслойная сверточная сеть со следующей архитектурой:

      conv - relu - 2x2 max pool - affine - relu - affine - softmax

    Сеть работает на мини-блоках данных, имеющих форму (N, C, H, W)
    состоящих из N изображений, каждое  высотой H и шириной W и  C
    каналами.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Инициализация новой сети.

        Входы:
         - input_dim: кортеж (C, H, W), задающий размер входных данных
         - num_filters: количество фильтров, используемых в сверточном слое
         - filter_size: ширина / высота фильтров для использования в сверточном слое
         - hidden_dim: количество нейронов, которые будут использоваться в полносвязном скрытом слое
         - num_classes: количество классов для окончательного аффинного слоя.
         - weight_scale: скалярное стандартное отклонение для случайной инициализации
           весов.
         - reg: скалярный коэфициент L2 силы регуляризации
         - dtype: numpy datatype для использования в вычислениях.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # ЗАДАНИЕ: Инициализация весов и смещений для 3-х слойной сверточной сети. #
        # Веса должны инициализироваться гауссовым распределением с центром в 0,0  #
        # и со стандартным отклонением, равным weight_scale; смещения должны       #
        # инициализироваться нулем. Все веса и смещения должны храниться в         #
        # словаре self.params. Храните веса и смещения для сверточного             #
        # слоя, сипользуя ключи «W1» и «b1»; используйте ключи «W2» и «b2» для     #
        # весов и смещений скрытого аффинного слоя, а также ключи «W3» и «b3»      #
        # для весов и смещений выходного аффинного слоя.                           #
        #                                                                          #
        # ВАЖНО: для этого задания вы можете допустить, что дополнение             #
        # и шаг первого сверточного слоя выбраны так, что                          #
        # **ширина и высота входа сохраняются**. Просмотрите                       #
        # начало функции loss (), чтобы увидеть, как это происходит.               #
        ############################################################################
        C, H, W = input_dim
        after_pooling_dim = int(H * W / 4.0)
        assert (after_pooling_dim == H * W / 4.0)

        self.params["W1"] = np.random.normal(0.0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params["b1"] = np.zeros(num_filters)
        self.params["W2"] = np.random.normal(0.0, weight_scale, (after_pooling_dim * num_filters, hidden_dim))
        self.params["b2"] = np.zeros(hidden_dim, dtype=float)
        self.params["W3"] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        self.params["b3"] = np.zeros(num_classes, dtype=float)
        ############################################################################
        #                             КОНЕЦ ВАШЕГО КОДА                            #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Оценивает потери и градиент для трехслойной сверточной сети.

        Вход / выход: тот же API, что и TwoLayerNet, в файле fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # Передаваемый conv_param в прямом направлении для сверточного слоя
        # Дополнение и шаг, выбран для сохранения входного пространственного размера
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # Передаваемый pool_param в прямом наравлении для слоя с макс пулом
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # ЗАДАНИЕ: выполнить прямой проход для трехслойной сверточной сети,        #
        # вычислить рейтинги классов для X и сохранить их в перменной scores       #
        #                                                                          #
        # Вы можете использовать функции, определенные в cs231n / fast_layers.py и #
        # cs231n / layer_utils.py  (уже импортирован).                             #
        ############################################################################
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out1flat = out1.reshape(out1.shape[0], -1)
        out2, cache2 = affine_relu_forward(out1flat, W2, b2)
        out3, cache3 = affine_forward(out2, W3, b3)
        scores = out3
        ############################################################################
        #                             КОНЕЦ ВАШЕГО КОДА                            #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # ЗАДАНИЕ: выполнить обратный проход для трехслойной сверточной сети,      #
        # сохранить потери и градиенты в переменных loss и grads. Вычислить        #
        # потери с помощью softmax и убедитесь, что grads[k] содержит градиенты    #
        # для self.params[k]. Не забудьте добавить регуляризацию L2!               #
        #                                                                          #
        # ПРИМЕЧАНИЕ. Чтобы пройти автоматические тесты убедитесь,                 #
        # что регуляризация L2 включает в себя фактор 0,5 для упрощения            #
        # выражения для градиента.                                                 #
        ############################################################################
        loss, dout3 = softmax_loss(scores, y)
        dout2, grads["W3"], grads["b3"] = affine_backward(dout3, cache3)
        dout1flat, grads["W2"], grads["b2"] = affine_relu_backward(dout2, cache2)
        dout1 = dout1flat.reshape(out1.shape)
        _, grads["W1"], grads["b1"] = conv_relu_pool_backward(dout1, cache1)

        for idx in range(1, 4):
            w = "W%d" % idx
            if self.reg > 0:
                loss += 0.5 * self.reg * (self.params[w] ** 2).sum()
                grads[w] += self.reg * self.params[w]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
