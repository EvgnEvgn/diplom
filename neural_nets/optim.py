import numpy as np

"""
Этот файл реализует различные правила обновления первого порядка, которые обычно используются
для обучения нейронных сетей. Каждое правило обновления принимает текущие веса и
градиент потерь по этим весам и создает новый набор
весов. Каждое правило обновления имеет один и тот же интерфейс:

def update(w, dw, config=None):

Входы:
   - w: массив numpy, содержащий текущие веса.
   - dw: массив numpy той же формы, что и w, содержащий градиент
     потерь по отношению к w.
   - config: словарь, содержащий значения гиперпараметров, такие как скорость
     обучения, момент и т. д. Если правило обновления требует кеширования значений
     между итерациями, то config будет хранить эти кешированные значения.

Возвращает:
   - next_w: следующая точка после обновления.
   - config: словарь конфигурации, который будет передан следующей итерации
     правила обновления.

ПРИМЕЧАНИЕ. Для большинства правил обновления скорость обучения по умолчанию, вероятно, не будет
хорошей; однако значения по умолчанию для других гиперпараметров должны
хорошо работать для решения различных проблем.

Для повышения эффективности правила обновления могут выполнять обновления по месту, изменяя w и
устаналивая next_w равным  w.

"""


def sgd(w, dw, config=None):
    """
    Выполняет простой SGD.

    config формат:
    - learning_rate: скорость обучения.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Выполняет SGD с моментом инерции.

    config формат:
    - learning_rate: скорость обучения.
    - momentum: скаляр от 0 and 1, представляющий значение момента.
      Установка momentum = 0 приводит к алгориму sgd.
    - velocity: numpy массив такой же формы, что w и dw; используется для
    хранения скользящих средних градиентов.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))
    mu = config.get('momentum')
    lr = config.get('learning_rate')

    ###########################################################################
    # ЗАДАНИЕ: Реализовать формулу обновления с моментом. Сохраните           #
    # обновленное значение в  переменнj next_w. Вы также должны также         #
    # обновлять скорость v.                                                   #
    ###########################################################################
    v = mu * v - lr * dw
    next_w = w + v
    ###########################################################################
    #                             КОНЕЦ ВАШЕГО КОДА                           #
    ###########################################################################
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Реализует правило обновления RMSProp, которое использует скользящее среднее квадрата
    градиента, чтобы установить адаптивные скорости обучения для каждого параметра.

     config формат:
     - learning_rate: Скалярная скорость обучения.
     - decay_rate: Скаляр между 0 и 1, задающий скорость затухания для квадрата
       градиента.
     - epsilon: малый скаляр, используемый для сглаживания, чтобы избежать деления на ноль.
     - cache: скользящее среднее вторых моментов градиентов.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))

    decay_rate = config.get('decay_rate')
    cache = config.get('cache')
    eps = config.get('epsilon')
    lr = config.get('learning_rate')
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    # ЗАДАНИЕ: Реализовать формулы RMSprop , сохраняя следующее значение w    #
    # в переменной next_w. Не забудьте обновить значение кеша, сохраненное в  #
    # config['cache'].
    ###########################################################################
    cache = decay_rate * cache + (1 - decay_rate) * dw ** 2
    next_w = w - lr * dw / (np.sqrt(cache) + eps)

    config['cache'] = cache
    ###########################################################################
    #                             КОНЕЦ ВАШЕГО КОДА                           #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Использует правило обновления Адам, которое включает скользящие средние как для
    градиента, так  и его квадрата, а также корректирующий коэффициент смещения.

     config формат:
     - learning_rate: скалярная скорость обучения.
     - beta1: скорость затухания для скользящего среднего первого момента градиента.
     - beta2: скорость затухания для скользящего среднего второго момента градиента.
     - epsilon: малый скаляр, используемый для сглаживания, чтобы избежать деления на ноль.
     - m: скользящее среднее градиента.
     - v: скользящее среднее квадрата градиента.
     - t: итерационный номер.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    beta1 = config.get('beta1')
    beta2 = config.get('beta2')
    epsilon = config.get('epsilon')
    learning_rate = config.get('learning_rate')
    m = config.get('m')
    v = config.get('v')
    t = config.get('t')
    ###########################################################################
    # ЗАДАНИЕ: Реализуте Формулы правила обновления Adam, сохраняя следующее  #
    # значение w в переменнq next_w. Не забудьте обновить переменные m, v и t #
    # сохраняемые в config.                                                   #
    #                                                                         #
    # ПРИМЕЧАНИЕ. Чтобы значения совпадали с эталонными, пожалуйста,          #
    # измените t перед использованием его в любых вычислениях.                #
    ###########################################################################
    t = t + 1
    m = beta1 * m + (1 - beta1) * dw
    mt = m / (1 - beta1 ** t)
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    vt = v / (1 - beta2 ** t)

    next_w = w - (learning_rate * mt) / (np.sqrt(vt) + epsilon)

    config['m'] = m
    config['v'] = v
    config['t'] = t
    ###########################################################################
    #                             КОНЕЦ ВАШЕГО КОДА                           #
    ###########################################################################

    return next_w, config
