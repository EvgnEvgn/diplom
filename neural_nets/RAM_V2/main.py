import sys
import numpy as np
import tensorflow.compat.v1 as tf
import pickle

import neural_nets.RAM_V2.read_mnist as loader
from neural_nets.RAM.data_loader import read_data_sets
from neural_nets.RAM_V2.ram import RAMClassification
from neural_nets.RAM_V2.Trainer import Trainer
from neural_nets.RAM_V2.Trainer_v2 import Trainer_v2
from neural_nets.RAM_V2.Predictor import Predictor
from neural_nets.RAM_V2.Predictor_v2 import Predictor_v2
from neural_nets.RAM.Dataset import Dataset

SAVE_PATH = 'saved_models/'
RESULT_PATH = 'result_2/'


def get_test_tchgn_dgts_dataset(filename):
    with open(filename, "rb") as file:
        dataset = pickle.load(file)

    test_X = dataset["test_X"]
    test_Y = dataset["test_Y"]

    return test_X, test_Y


def get_tchng_dgts_small_dataset(filename):
    with open(filename, "rb") as file:
        dataset = pickle.load(file)

    train_X = dataset["train_X"]
    train_Y = dataset["train_Y"]

    val_X = dataset["val_X"]
    val_Y = dataset["val_Y"]

    test_X = dataset["test_X"]
    test_Y = dataset["test_Y"]

    return train_X, train_Y, val_X, val_Y, test_X, test_Y


def get_args():
    flags = {}
    flags['dataset'] = 'center'
    flags['load'] = 57
    flags['lr'] = 1e-3

    # parser.add_argument('--dataset', type=str, default='center',
    #                     help='Use original or tranlated MNIST ("center" or "translate")')
    #
    # parser.add_argument('--step', type=int, default=1,
    #                     help='Number of glimpse')
    # parser.add_argument('--sample', type=int, default=1,
    #                     help='Number of location samples during training')
    # parser.add_argument('--glimpse', type=int, default=12,
    #                     help='Glimpse base size')
    # parser.add_argument('--batch', type=int, default=128,
    #                     help='Batch size')
    # parser.add_argument('--epoch', type=int, default=1000,
    #                     help='Max number of epoch')
    # parser.add_argument('--load', type=int, default=100,
    #                     help='Load pretrained parameters with id')
    # parser.add_argument('--lr', type=float, default=1e-3,
    #                     help='Init learning rate')
    # parser.add_argument('--std', type=float, default=0.11,
    #                     help='std of location')
    # parser.add_argument('--pixel', type=int, default=26,
    #                     help='unit_pixel')
    # parser.add_argument('--scale', type=int, default=3,
    #                     help='scale of glimpse')

    return flags


class config_center():
    step = 6
    sample = 1
    glimpse = 12
    n_scales = 3
    batch = 98
    epoch = 20
    loc_std = 0.13
    unit_pixel = 20
    im_size = 64
    trans = False


class config_transform():
    step = 6
    sample = 1
    glimpse = 12
    n_scales = 3
    batch = 128
    epoch = 2000
    loc_std = 0.03
    unit_pixel = 26
    im_size = 60
    trans = True


def get_config_data(flags):
    if flags['dataset'] == 'translate':
        name = 'trans'
        config = config_transform()
    else:
        name = 'centered'
        config = config_center()

    train_data, valid_data = loader.original_mnist(batch_size=config.batch, shuffle=True)
    return config, name, train_data, valid_data


def get_pred_config_data_v2(flags):
    if flags['dataset'] == 'translate':
        name = 'trans'
        config = config_transform()
    else:
        name = 'centered'
        config = config_center()

    # _, _, _, _, test_X, test_Y = read_data_sets('../RAM/mnist_data')
    # train_data, valid_data = loader.original_mnist(batch_size=config.batch, shuffle=True)
    # train_X = train_data.im_list
    # train_Y = train_data.label_list
    # val_X = valid_data.im_list
    # val_Y = valid_data.label_list
    # train_X, train_Y, val_X, val_Y, test_X, test_Y = get_tchng_dgts_small_dataset('../touching_dgts_90-99_64.bin')
    test_X, test_Y = get_test_tchgn_dgts_dataset("../../test_2tch_dgts_from_10-99_98examples_float32.bin")
    #test_Y += 80
    test_X = np.reshape(test_X, (test_X.shape[0], 64, 64, 1))

    return config, name, test_X, test_Y


def get_config_data_v2(flags):
    if flags['dataset'] == 'translate':
        name = 'trans'
        config = config_transform()
    else:
        name = 'centered'
        config = config_center()

    train_X, train_Y, val_X, val_Y, test_X, test_Y = get_tchng_dgts_small_dataset('../touching_dgts_90-99_64.bin')
    train_X = np.reshape(train_X, (train_X.shape[0], 64, 64, 1))
    val_X = np.reshape(val_X, (val_X.shape[0], 64, 64, 1))
    # train_X, train_Y, val_X, val_Y, test_X, test_Y = read_data_sets('../RAM/mnist_data')

    # train_data, valid_data = loader.original_mnist(batch_size=config.batch, shuffle=True)
    # train_X = train_data.im_list
    # train_Y = train_data.label_list
    # val_X = valid_data.im_list
    # val_Y = valid_data.label_list

    return config, name, train_X, train_Y, val_X, val_Y


def train():
    flags = get_args()

    config, name, train_data, valid_data = get_config_data(flags)

    model = RAMClassification(im_size=config.im_size,
                              im_channel=1,
                              glimpse_base_size=config.glimpse,
                              n_glimpse_scale=config.n_scales,
                              n_loc_sample=config.sample,
                              n_step=config.step,
                              n_class=10,
                              max_grad_norm=5.0,
                              unit_pixel=config.unit_pixel,
                              loc_std=config.loc_std,
                              is_transform=config.trans)
    model.create_train_model()

    trainer = Trainer(model, train_data, init_lr=flags['lr'])
    writer = tf.summary.FileWriter(SAVE_PATH)
    saver = tf.train.Saver()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for step in range(0, config.epoch):
            trainer.train_epoch(sess, summary_writer=writer)
            trainer.valid_epoch(sess, valid_data, config.batch, summary_writer=writer)
            saver.save(sess,
                       '{}ram-{}-mnist-step-{}'
                       .format(SAVE_PATH, name, config.step),
                       global_step=step)
            writer.close()


def train_v2():
    flags = get_args()

    config, name, train_X, train_Y, val_X, val_Y = get_config_data_v2(flags)

    model = RAMClassification(im_size=config.im_size,
                              im_channel=1,
                              glimpse_base_size=config.glimpse,
                              n_glimpse_scale=config.n_scales,
                              n_loc_sample=config.sample,
                              n_step=config.step,
                              n_class=10,
                              max_grad_norm=5.0,
                              unit_pixel=config.unit_pixel,
                              loc_std=config.loc_std,
                              is_transform=config.trans)
    model.create_train_model()
    train_dataset = Dataset(train_X, train_Y, config.batch, reshape_img=False)
    trainer = Trainer_v2(model, train_dataset, init_lr=flags['lr'])
    writer = tf.summary.FileWriter(SAVE_PATH)
    saver = tf.train.Saver()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    val_dataset = Dataset(val_X, val_Y, config.batch, reshape_img=False)
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for epoch in range(0, config.epoch):
            trainer.train_epoch(sess, epoch, summary_writer=writer)
            trainer.valid_epoch(sess, val_dataset, summary_writer=writer)
            saver.save(sess,
                       '{}ram-{}-mnist-step-{}'
                       .format(SAVE_PATH, name, config.step),
                       global_step=epoch)
            writer.close()


def predict():
    flags = get_args()

    config, name, train_data, valid_data = get_config_data(flags)

    model = RAMClassification(im_size=config.im_size,
                              im_channel=1,
                              glimpse_base_size=config.glimpse,
                              n_glimpse_scale=config.n_scales,
                              n_loc_sample=config.sample,
                              n_step=config.step,
                              n_class=10,
                              max_grad_norm=5.0,
                              unit_pixel=config.unit_pixel,
                              loc_std=config.loc_std,
                              is_transform=config.trans)
    model.create_predict_model()

    predictor = Predictor(model)
    saver = tf.train.Saver()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess,
                      '{}ram-{}-mnist-step-6-{}'
                      .format(SAVE_PATH, name, flags['load']))

        batch_data = valid_data.next_batch_dict()
        predictor.test_batch(
            sess,
            batch_data,
            unit_pixel=config.unit_pixel,
            size=config.glimpse,
            scale=config.n_scales,
            save_path=RESULT_PATH)


def predict_v2():
    flags = get_args()

    config, name, test_X, test_Y = get_pred_config_data_v2(flags)

    model = RAMClassification(im_size=config.im_size,
                              im_channel=1,
                              glimpse_base_size=config.glimpse,
                              n_glimpse_scale=config.n_scales,
                              n_loc_sample=config.sample,
                              n_step=config.step,
                              n_class=90,
                              max_grad_norm=5.0,
                              unit_pixel=config.unit_pixel,
                              loc_std=config.loc_std,
                              is_transform=config.trans)
    model.create_predict_model()

    predictor = Predictor_v2(model)
    saver = tf.train.Saver()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    dataset = Dataset(test_X, test_Y, batch_size=config.batch, reshape_img=False)
    batch_X, batch_Y = dataset.next_batch(0)
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess,
                      '{}ram-{}-mnist-step-6-{}'
                      .format(SAVE_PATH, name, flags['load']))

        predictor.test_batch(
            sess,
            batch_X,
            batch_Y,
            unit_pixel=config.unit_pixel,
            size=config.glimpse,
            scale=config.n_scales,
            save_path=RESULT_PATH)


def evaluate():
    flags = get_args()

    config, name, train_data, valid_data = get_config_data(flags)

    model = RAMClassification(im_size=config.im_size,
                              im_channel=1,
                              glimpse_base_size=config.glimpse,
                              n_glimpse_scale=config.n_scales,
                              n_loc_sample=config.sample,
                              n_step=config.step,
                              n_class=10,
                              max_grad_norm=5.0,
                              unit_pixel=config.unit_pixel,
                              loc_std=config.loc_std,
                              is_transform=config.trans)
    model.create_predict_model()

    predictor = Predictor(model)
    saver = tf.train.Saver()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess,
                      '{}ram-{}-mnist-step-6-{}'
                      .format(SAVE_PATH, name, flags['load']))

        predictor.evaluate(sess, valid_data)


def evaluate_v2():
    flags = get_args()

    config, name, test_X, test_Y = get_pred_config_data_v2(flags)

    model = RAMClassification(im_size=config.im_size,
                              im_channel=1,
                              glimpse_base_size=config.glimpse,
                              n_glimpse_scale=config.n_scales,
                              n_loc_sample=config.sample,
                              n_step=config.step,
                              n_class=90,
                              max_grad_norm=5.0,
                              unit_pixel=config.unit_pixel,
                              loc_std=config.loc_std,
                              is_transform=config.trans)
    model.create_predict_model()

    predictor = Predictor_v2(model)
    saver = tf.train.Saver()

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess,
                      '{}ram-{}-mnist-step-6-{}'
                      .format(SAVE_PATH, name, flags['load']))
        dataset = Dataset(test_X, test_Y, batch_size=config.batch, reshape_img=False)
        predictor.evaluate(sess, dataset)


tf.disable_eager_execution()
# train_v2()
evaluate_v2()
# #predict_v2()
