import numpy as np

from neural_nets.layers import *
from neural_nets.layers_utils import *


class FullyConnectedNet(object):
    """
    Полносвязанная нейронная сеть с произвольным количеством скрытых слоев,
    ReLU нелинейностями и функция потерь softmax. Также реализует
    dropout и нормализация на блоке/слея в качестве опции. Для сети с L-слоями,
    архитектура будет иметь вид:

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax,

    где нормализации являются необязательными, а блок {...}
    повторяется L - 1 раз.

    Как и в случае с TwoLayerNet выше, обучаемые параметры сохраняются в
    self.params и будут обучаться с использованием класса Solver. #
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Инициализурует объект FullyConnectedNet.

        Входы:
        - hidden_dims: список целых чисел, задающих размер каждого скрытого слоя.
        - input_dim: целое число, задающее размер входа.
        - num_classes: целое число, представляющее количество классов для классификации.
        - dropout: скаляр между 0 и 1. Если dropout = 1, то
          сеть вообще не должна использовать исключение узлов.
        - normalization: какой тип нормализации должна использовать сеть. Допустимые значения
          "batchnorm", "layernorm" или None для отсутствия нормализации (по умолчанию).
        - reg: скаляр, задающий силу регуляризации L2.
        - weight_scale: скаляр, задающий стандартное отклонение для случайных
          инициализации весов.
        - dtype: объект типа numpy datatype; все вычисления будут выполнены с использованием
          этого типа данных. float32 быстрее, но менее точен, поэтому вы должны использовать
          float64 для проверки числового градиента.
        - seed: если нет, то None, передает случайное seed слоям  dropout. это
          приведет к тому, что уровни  dropout будут детерминированными, чтобы мы могли
          сделать проверку градиента.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.train_accuracy = None
        self.val_accuracy = None
        self.test_accuracy = None

        ############################################################################
        # ЗАДАНИЕ: Инициализировать параметры сети, сохраняя все значения в        #
        # словаре self.params. Хранить веса и смещения для первого слоя            #
        # в W1 и b1; для второго слоя в W2 и b2 и т. д. Вес должен инициализиро-   #
        # ваться нормальным распределением с центром в 0 со стандартным            #
        # отклонением, равным weight_scale.Смещения должны быть инициализированы   #
        # нулем.                                                                   #
        #                                                                          #
        # При использовании блочной нормализации сохраните масштаб и параметры     #
        # сдвига для первого слоя в gamma1 и beta1; для второго слоя используйте   #
        # gamma2 и beta2 и т. д. Параметры масштаба должны быть инициализированы   #
        # единицей, а параметры сдвига - нулями.                                   #
        ############################################################################
        all_dims = [input_dim] + hidden_dims + [num_classes]

        for idx in range(self.num_layers):
            in_d, out_d = all_dims[idx:(idx + 2)]
            self.params["W%d" % (idx + 1)] = np.random.normal(0.0, weight_scale, (in_d, out_d))
            self.params["b%d" % (idx + 1)] = np.zeros(out_d, dtype=float)

        if normalization:
            for idx, dim in enumerate(hidden_dims):
                self.params["gamma%d" % (idx + 1)] = np.ones(dim, dtype=float)
                self.params["beta%d" % (idx + 1)] = np.zeros(dim, dtype=float)

        ############################################################################
        #                            КОНЕЦ ВАШЕГО КОДА                             #
        ############################################################################

        # При использовании dropout нам нужно передать словарь dropout_param каждому
        # dropout слою, чтобы  слой знал вероятность исключения нейронов и знал режим
        # (обучение/ тест). Вы можете передавать то же  значение dropout_param
        # каждому dropout слою.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # При блочной нормализации нам нужно отслеживать текущие средние и
        # дисперсии, поэтому нам нужно передать специальный объект bn_param для каждого слоя
        # BN. Вы должны передавать self.bn_params[0] при прямом проходе для
        # первого слоя блочной нормализации, self.bn_params[1] при прямом проходе для
        # второго слоя блочной нормализации и т. д.
        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Приведение всех параметров к правильному типу
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Вычислить потери и градиент для полносвязанной сети.

         Вход / выход: то же, что и у TwoLayerNet выше.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Устанавливаем train/test режим для параметров batchnorm и dropout, так как они
        # ведут себя по-разному во время обучения и тестирования.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # ЗАДАНИЕ: выполнить прямой проход для полносвязанной сети, вычислить      #
        # рейтинги классов для X и сохранить их в переменной scores.               #
        #                                                                          #
        # При использовании dropout необходимо передать self.dropout_param для     #
        # каждого слоя dropout на прямом пути                                      #
        #                                                                          #
        # При использовании блочной нормализации необходимо передавать             #
        # self.bn_params[0] при прямом проходе для первого слоя BN                 #
        # self.bn_params [1] - при прямом проходе для второго слоя BN  и т.д.      #
        ############################################################################
        cachelist = []
        out = X
        for idx in range(self.num_layers):
            w = self.params["W%d" % (idx + 1)]
            b = self.params["b%d" % (idx + 1)]
            if idx == self.num_layers - 1:
                out, cache = affine_forward(out, w, b)
            elif self.normalization:
                gamma = self.params["gamma%d" % (idx + 1)]
                beta = self.params["beta%d" % (idx + 1)]
                if self.normalization == 'batchnorm':
                    out, cache = affine_batchnorm_relu_forward(out, w, b, gamma, beta, self.bn_params[idx])
                elif self.normalization == 'layernorm':
                    out, cache = affine_layernorm_relu_forward(out, w, b, gamma, beta, self.bn_params[idx])
            else:
                out, cache = affine_relu_forward(out, w, b)

            if not idx == self.num_layers - 1 and self.use_dropout:
                out, cache_do = dropout_forward(out, self.dropout_param)
                cache = (cache, cache_do)

            cachelist.append(cache)

            scores = out

        pass
        ############################################################################
        #                             КОНЕЦ ВАШЕГО КОДА                            #
        ############################################################################

        # если режим тестирования, то ранний выход
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # ЗАДАНИЕ: выполнить обратный проход для полносвязанной сети. Сохраните    #
        # потери в переменной loss и градиенты в словаре grads. Вычислите          #
        # потери данных с помощью softmax и убедитесь, что grads[k] содержит       #
        # градиенты для self.params[k]. Не забудьте добавить регуляризацию L2!     #
        #                                                                          #
        # При использовании нормализации на блоке/слое вам не нужно регуляризировать #
        # параметры масштаба и параметры сдвига.                                   #
        #                                                                          #
        # ПРИМЕЧАНИЕ. Чтобы быть уверенным, что ваша реализация соответствует      #
        # нашей, и  вы пройдете автоматические тесты, убедитесь, что ваша          #
        # регуляризация L2 включает в себя  множитель 0,5 для упрощения            #
        # выражения для градиента.                                                 #
        ############################################################################
        # вычисляем средние потери для миниблока и матрицу градиентов модуля softmax
        loss, dscores = softmax_loss(scores, y)
        dout = dscores
        for idx in reversed(range(self.num_layers)):
            cache = cachelist[idx]
            if idx == self.num_layers - 1:
                dout, dw, db = affine_backward(dout, cache)
            else:
                if self.use_dropout:
                    cache, cache_do = cache
                    dout = dropout_backward(dout, cache_do)

                if self.normalization:
                    if self.normalization == 'batchnorm':
                        dout, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(dout, cache)
                    elif self.normalization == 'layernorm':
                        dout, dw, db, dgamma, dbeta = affine_layernorm_relu_backward(dout, cache)

                    grads["gamma%d" % (idx + 1)] = dgamma
                    grads["beta%d" % (idx + 1)] = dbeta
                else:
                    dout, dw, db = affine_relu_backward(dout, cache)

            grads["W%d" % (idx + 1)] = dw
            grads["b%d" % (idx + 1)] = db

        for idx in range(self.num_layers):
            w = "W%d" % (idx + 1)
            if self.reg > 0:
                loss += 0.5 * self.reg * (self.params[w] ** 2).sum()
                grads[w] += self.reg * self.params[w]
        ############################################################################
        #                             КОНЕЦ ВАШЕГО КОДА                            #
        ############################################################################

        return loss, grads
