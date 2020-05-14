import os
# import tensorflow as tf

import numpy as np
import time
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import random
from neural_nets.RAM import Dataset
import matplotlib.pyplot as plt
from neural_nets.RAM import ram_config

# tf.disable_v2_behavior()


class RecurrentAttentionModel(object):
    def __init__(self, config: ram_config):
        # todo = add own data loader

        self.save_dir = config.save_dir
        self.save_prefix = config.save_prefix
        self.summaryFolderName = config.summaryFolderName

        self.start_step = config.start_step
        self.model_name = config.model_name
        self.ckpt_path = config.ckpt_path
        self.meta_path = config.meta_path

        # conditions
        self.translateMnist = config.translateMnist
        self.eyeCentered = config.eyeCentered

        # about translation
        self.ORIG_IMG_SIZE = config.ORIG_IMG_SIZE
        self.translated_img_size = config.translated_img_size

        self.fixed_learning_rate = config.fixed_learning_rate

        self.img_size = config.img_size
        self.depth = config.depth
        self.sensorBandwidth = config.sensorBandwidth
        self.minRadius = config.minRadius

        self.initLr = config.initLr
        self.lr_min = config.lr_min
        self.lrDecayRate = config.lrDecayRate
        self.lrDecayFreq = config.lrDecayFreq
        self.momentumValue = config.momentumValue
        self.batch_size = config.batch_size

        self.channels = config.channels # mnist are grayscale images
        self.totalSensorBandwidth = config.totalSensorBandwidth
        self.nGlimpses = config.nGlimpses  # number of glimpses
        self.loc_sd = config.loc_sd  # std when setting the location

        # network units
        self.hg_size = config.hg_size  #
        self.hl_size = config.hl_size  #
        self.g_size = config.g_size #
        self.cell_size = config.cell_size  #
        self.cell_out_size = config.cell_out_size  #

        # paramters about the training examples
        self.n_classes = config.n_classes
        # config("num_classes")  # card(Y)

        # training parameters
        self.max_iters = config.max_iters
        self.max_epochs = config.max_epochs
        # config("max_iters")
        self.SMALL_NUM = config.SMALL_NUM
        # config("small_num")

        # resource prellocation
        self.mean_locs = []  # expectation of locations
        self.sampled_locs = []  # sampled locations ~N(mean_locs[.], loc_sd)
        self.baselines = []  # baseline, the value prediction
        self.glimpse_images = []  # to show in window

        self.global_step = None
        self.inputs_placeholder = None
        self.labels_placeholder = None
        self.onehot_labels_placeholder = None
        self.lr = None
        self.Wg_l_h = None
        self.Bg_l_h = None
        self.Wg_g_h = None
        self.Bg_g_h = None
        self.Wg_hg_gf1 = None
        self.Wg_hl_gf1 = None
        self.Bg_hlhg_gf1 = None
        self.Wc_g_h = None
        self.Bc_g_h = None
        self.Wr_h_r = None
        self.Br_h_r = None
        self.Wb_h_b = None
        self.Bb_h_b = None
        self.Wl_h_l = None
        self.Bl_h_l = None
        self.Wa_h_a = None
        self.Ba_h_a = None

    def weight_variable(self, shape, myname, train):
        initial = tf.random_uniform(shape, minval=-0.1, maxval=0.1)
        return tf.Variable(initial, name=myname, trainable=train)

    def glimpseSensor(self, img, normLoc):

        loc = tf.round(((normLoc + 1) / 2.0) * self.img_size)  # normLoc coordinates are between -1 and 1
        loc = tf.cast(loc, tf.int32)

        img = tf.reshape(img, (self.batch_size, self.img_size, self.img_size, self.channels))

        # process each image individually
        zooms = []
        for k in range(self.batch_size):
            imgZooms = []
            one_img = img[k, :, :, :]
            max_radius = self.minRadius * (2 ** (self.depth - 1))
            offset = 2 * max_radius

            # pad image with zeros
            one_img = tf.image.pad_to_bounding_box(one_img, offset, offset,
                                                   max_radius * 4 + self.img_size,
                                                   max_radius * 4 + self.img_size)

            for i in range(self.depth):
                r = int(self.minRadius * (2 ** (i)))

                d_raw = 2 * r
                d = tf.constant(d_raw, shape=[1])
                d = tf.tile(d, [2])
                loc_k = loc[k, :]
                adjusted_loc = offset + loc_k - r
                one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value, one_img.get_shape()[1].value))

                # crop image to (d x d)
                zoom = tf.slice(one_img2, adjusted_loc, d)

                # resize cropped image to (sensorBandwidth x sensorBandwidth)
                zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw, d_raw, 1)),
                                                (self.sensorBandwidth, self.sensorBandwidth))
                zoom = tf.reshape(zoom, (self.sensorBandwidth, self.sensorBandwidth))
                imgZooms.append(zoom)

            zooms.append(tf.stack(imgZooms))

        zooms = tf.stack(zooms)

        self.glimpse_images.append(zooms)

        return zooms

    # implements the input network
    def get_glimpse(self, loc):
        # get input using the previous location
        glimpse_input = self.glimpseSensor(self.inputs_placeholder, loc)
        glimpse_input = tf.reshape(glimpse_input, (self.batch_size, self.totalSensorBandwidth))

        # the hidden units that process location & the input
        act_glimpse_hidden = tf.nn.relu(tf.matmul(glimpse_input, self.Wg_g_h) + self.Bg_g_h)
        act_loc_hidden = tf.nn.relu(tf.matmul(loc, self.Wg_l_h) + self.Bg_l_h)

        # the hidden units that integrates the location & the glimpses
        glimpseFeature1 = tf.nn.relu(
            tf.matmul(act_glimpse_hidden, self.Wg_hg_gf1) + tf.matmul(act_loc_hidden,
                                                                      self.Wg_hl_gf1) + self.Bg_hlhg_gf1)
        # return g
        # glimpseFeature2 = tf.matmul(glimpseFeature1, Wg_gf1_gf2) + Bg_gf1_gf2
        return glimpseFeature1

    def get_next_input(self, output):
        # the next location is computed by the location network
        core_net_out = tf.stop_gradient(output)

        # baseline = tf.sigmoid(tf.matmul(core_net_out, Wb_h_b) + Bb_h_b)
        baseline = tf.sigmoid(tf.matmul(core_net_out, self.Wb_h_b) + self.Bb_h_b)
        self.baselines.append(baseline)

        # compute the next location, then impose noise
        if self.eyeCentered:
            # add the last sampled glimpse location
            # TODO max(-1, min(1, u + N(output, sigma) + prevLoc))
            mean_loc = tf.maximum(-1.0, tf.minimum(1.0, tf.matmul(core_net_out, self.Wl_h_l) + self.sampled_locs[-1]))
        else:
            # mean_loc = tf.clip_by_value(tf.matmul(core_net_out, Wl_h_l) + Bl_h_l, -1, 1)
            mean_loc = tf.matmul(core_net_out, self.Wl_h_l) + self.Bl_h_l
            mean_loc = tf.clip_by_value(mean_loc, -1, 1)
        # mean_loc = tf.stop_gradient(mean_loc)
        self.mean_locs.append(mean_loc)

        # add noise
        # sample_loc = tf.tanh(mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd))
        sample_loc = tf.maximum(-1.0,
                                tf.minimum(1.0, mean_loc + tf.random_normal(mean_loc.get_shape(), 0, self.loc_sd)))

        # don't propagate throught the locations
        sample_loc = tf.stop_gradient(sample_loc)
        self.sampled_locs.append(sample_loc)

        return self.get_glimpse(sample_loc)

    @staticmethod
    def affine_transform(x, output_dim):
        """
        affine transformation Wx+b
        assumes x.shape = (batch_size, num_features)
        """
        w = tf.get_variable("w", [x.get_shape()[1], output_dim])
        b = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b

    def model(self):
        # initialize the location under unif[-1,1], for all example in the batch
        initial_loc = tf.random_uniform((self.batch_size, 2), minval=-1, maxval=1)
        self.mean_locs.append(initial_loc)

        # initial_loc = tf.tanh(initial_loc + tf.random_normal(initial_loc.get_shape(), 0, loc_sd))
        initial_loc = tf.clip_by_value(initial_loc + tf.random_normal(initial_loc.get_shape(), 0, self.loc_sd), -1, 1)

        self.sampled_locs.append(initial_loc)

        # get the input using the input network
        initial_glimpse = self.get_glimpse(initial_loc)

        # set up the recurrent structure
        inputs = [0] * self.nGlimpses
        outputs = [0] * self.nGlimpses
        glimpse = initial_glimpse
        REUSE = None
        for t in range(self.nGlimpses):
            if t == 0:  # initialize the hidden state to be the zero vector
                hiddenState_prev = tf.zeros((self.batch_size, self.cell_size))
            else:
                hiddenState_prev = outputs[t - 1]

            # forward prop
            with tf.variable_scope("coreNetwork", reuse=REUSE):
                # the next hidden state is a function of the previous hidden state and the current glimpse
                hiddenState = tf.nn.relu(
                    self.affine_transform(hiddenState_prev, self.cell_size) + (
                            tf.matmul(glimpse, self.Wc_g_h) + self.Bc_g_h))

            # save the current glimpse and the hidden state
            inputs[t] = glimpse
            outputs[t] = hiddenState
            # get the next input glimpse
            if t != self.nGlimpses - 1:
                glimpse = self.get_next_input(hiddenState)
            else:
                first_hiddenState = tf.stop_gradient(hiddenState)
                # baseline = tf.sigmoid(tf.matmul(first_hiddenState, Wb_h_b) + Bb_h_b)
                baseline = tf.sigmoid(tf.matmul(first_hiddenState, self.Wb_h_b) + self.Bb_h_b)
                self.baselines.append(baseline)
            REUSE = True  # share variables for later recurrence

        return outputs

    # to use for maximum likelihood with input location
    def gaussian_pdf(self, mean, sample):
        Z = 1.0 / (self.loc_sd * tf.sqrt(2.0 * np.pi))
        a = -tf.square(sample - mean) / (2.0 * tf.square(self.loc_sd))
        return Z * tf.exp(a)

    def dense_to_one_hot(self, labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        # copied from TensorFlow tutorial
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def convertTranslated(self, images, initImgSize, finalImgSize):
        batch_size = len(images)
        size_diff = finalImgSize - initImgSize
        newimages = np.zeros([batch_size, finalImgSize * finalImgSize])
        imgCoord = np.zeros([batch_size, 2])
        for k in range(batch_size):
            image = images[k, :]
            image = np.reshape(image, (initImgSize, initImgSize))
            # generate and save random coordinates
            randX = random.randint(0, size_diff)
            randY = random.randint(0, size_diff)
            imgCoord[k, :] = np.array([randX, randY])
            # padding
            image = np.lib.pad(image, ((randX, size_diff - randX), (randY, size_diff - randY)),
                               'constant', constant_values=(0))
            newimages[k, :] = np.reshape(image, (finalImgSize * finalImgSize))

        return newimages, imgCoord

    def preTrain(self, outputs):
        lr_r = 1e-3
        # consider the action at the last time step
        outputs = outputs[-1]  # look at ONLY THE END of the sequence
        outputs = tf.reshape(outputs, (self.batch_size, self.cell_out_size))
        # if preTraining:
        reconstruction = tf.sigmoid(tf.matmul(outputs, self.Wr_h_r) + self.Br_h_r)
        reconstructionCost = tf.reduce_mean(tf.square(self.inputs_placeholder - reconstruction))

        train_op_r = tf.train.RMSPropOptimizer(lr_r).minimize(reconstructionCost)
        return reconstructionCost, reconstruction, train_op_r

    def evaluate(self, dataset: Dataset, sess, reward, predicted_labels, correct_labels, glimpse_images, draw=False):

        batches_in_epoch = dataset.batch_count()
        accuracy = 0
        acc_v2 = 0
        if draw:
            fig = plt.figure(1)
            txt = fig.suptitle("-", fontsize=36, fontweight='bold')
            plt.ion()
            plt.show()
            plt.subplots_adjust(top=0.7)
            plotImgs = []

        for batch_idx in range(batches_in_epoch):
            nextX, nextY = dataset.next_batch(batch_idx)
            if self.translateMnist:
                nextX, _ = self.convertTranslated(nextX, self.ORIG_IMG_SIZE, self.img_size)

            feed_dict = {self.inputs_placeholder: nextX, self.labels_placeholder: nextY,
                         self.onehot_labels_placeholder: self.dense_to_one_hot(nextY, num_classes=self.n_classes)}

            fetches = [reward, predicted_labels, correct_labels, glimpse_images, self.sampled_locs]
            result = sess.run(fetches, feed_dict=feed_dict)
            reward_fetched, prediction_labels_fetched, correct_labels_fetched, glimpse_images_fetched, sampled_locs_fetched = result
            accuracy += reward_fetched
            acc_v2 += np.mean(prediction_labels_fetched == correct_labels_fetched)
            if draw:
                f_glimpse_images = np.reshape(glimpse_images_fetched, \
                                              (
                                                  self.nGlimpses, self.batch_size, self.depth,
                                                  self.sensorBandwidth,
                                                  self.sensorBandwidth))

                fillList = False
                if len(plotImgs) == 0:
                    fillList = True

                # display the first image in the in mini-batch
                nCols = self.depth + 1
                plt.subplot2grid((self.depth, nCols), (0, 1), rowspan=self.depth, colspan=self.depth)
                # display the entire image
                self.plotWholeImg(nextX[0, :], self.img_size, sampled_locs_fetched)

                # display the glimpses
                for y in range(self.nGlimpses):
                    txt.set_text('\nPrediction: %i -- Truth: %i\nStep: %i/%i'
                                 % (prediction_labels_fetched[0], correct_labels_fetched[0],
                                    (y + 1),
                                    self.nGlimpses))

                    for x in range(self.depth):
                        plt.subplot(self.depth, nCols, 1 + nCols * x)
                        if fillList:
                            plotImg = plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'),
                                                 interpolation="nearest")
                            plotImg.autoscale()
                            plotImgs.append(plotImg)
                        else:
                            plotImgs[x].set_data(f_glimpse_images[y, 0, x])
                            plotImgs[x].autoscale()
                    fillList = False

                    # fig.canvas.draw()
                    time.sleep(0.15)
                    plt.pause(0.53)
            #print("Evaluate accuraccy: acc = {0} Batch №{1}".format(reward_fetched, str(i + 1)))

        accuracy /= batches_in_epoch
        acc_v2 /= batches_in_epoch
        print(("ACCURACY: " + str(accuracy)))
        print(("Correct labels: " + str(acc_v2)))

    def calc_reward(self, outputs):
        # consider the action at the last time step
        outputs = outputs[-1]  # look at ONLY THE END of the sequence
        outputs = tf.reshape(outputs, (self.batch_size, self.cell_out_size))

        # get the baseline
        b = tf.stack(self.baselines)
        b = tf.concat(axis=2, values=[b, b])
        b = tf.reshape(b, (self.batch_size, (self.nGlimpses) * 2))
        no_grad_b = tf.stop_gradient(b)

        # get the action(classification)
        p_y = tf.nn.softmax(tf.matmul(outputs, self.Wa_h_a) + self.Ba_h_a)
        max_p_y = tf.arg_max(p_y, 1)
        correct_y = tf.cast(self.labels_placeholder, tf.int64)

        # reward for all examples in the batch
        R = tf.cast(tf.equal(max_p_y, correct_y), tf.float32)
        reward = tf.reduce_mean(R)  # mean reward
        R = tf.reshape(R, (self.batch_size, 1))
        R = tf.tile(R, [1, (self.nGlimpses) * 2])

        # get the location

        p_loc = self.gaussian_pdf(self.mean_locs, self.sampled_locs)
        # p_loc = tf.tanh(p_loc)

        p_loc_orig = p_loc
        p_loc = tf.reshape(p_loc, (self.batch_size, (self.nGlimpses) * 2))

        # define the cost function
        J = tf.concat(axis=1, values=[tf.log(p_y + self.SMALL_NUM) * (self.onehot_labels_placeholder),
                                      tf.log(p_loc + self.SMALL_NUM) * (R - no_grad_b)])
        J = tf.reduce_sum(J, 1)
        J = J - tf.reduce_sum(tf.square(R - b), 1)
        J = tf.reduce_mean(J, 0)
        cost = -J
        var_list = tf.trainable_variables()
        grads = tf.gradients(cost, var_list)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        # define the optimizer
        # lr_max = tf.maximum(lr, lr_min)
        optimizer = tf.train.AdamOptimizer(self.lr)
        # optimizer = tf.train.MomentumOptimizer(lr, momentumValue)
        # train_op = optimizer.minimize(cost, global_step)
        train_op = optimizer.apply_gradients(zip(grads, var_list), global_step=self.global_step)

        return cost, reward, max_p_y, correct_y, train_op, b, tf.reduce_mean(b), tf.reduce_mean(R - b), self.lr

    def toMnistCoordinates(self, coordinate_tanh):
        '''
        Transform coordinate in [-1,1] to mnist
        :param coordinate_tanh: vector in [-1,1] x [-1,1]
        :return: vector in the corresponding mnist coordinate
        '''
        return np.round(((coordinate_tanh + 1) / 2.0) * self.img_size)

    def plotWholeImg(self, img, img_size, sampled_locs_fetched):
        plt.imshow(np.reshape(img, [img_size, img_size]),
                   cmap=plt.get_cmap('gray'), interpolation="nearest")

        plt.ylim((img_size - 1, 0))
        plt.xlim((0, img_size - 1))

        # transform the coordinate to mnist map
        sampled_locs_mnist_fetched = self.toMnistCoordinates(sampled_locs_fetched)
        # visualize the trace of successive nGlimpses (note that x and y coordinates are "flipped")
        plt.plot(sampled_locs_mnist_fetched[0, :, 1], sampled_locs_mnist_fetched[0, :, 0], '-o',
                 color='lawngreen')
        plt.plot(sampled_locs_mnist_fetched[0, -1, 1], sampled_locs_mnist_fetched[0, -1, 0], 'o',
                 color='red')

    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        with tf.device('/cpu:1'):
            with tf.Graph().as_default():

                # set the learning rate
                self.global_step = tf.Variable(0, trainable=False)
                self.lr = tf.train.exponential_decay(self.initLr, self.global_step, self.lrDecayFreq, self.lrDecayRate,
                                                     staircase=True)

                # preallocate x, y, baseline
                labels = tf.placeholder("float32", shape=[self.batch_size, self.n_classes])
                self.labels_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size), name="labels_raw")
                self.onehot_labels_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_classes),
                                                                name="labels_onehot")
                self.inputs_placeholder = tf.placeholder(tf.float32,
                                                         shape=(self.batch_size, self.img_size * self.img_size),
                                                         name="images")

                # declare the model parameters, here're naming rule:
                # the 1st captical letter: weights or bias (W = weights, B = bias)
                # the 2nd lowercase letter: the network (e.g.: g = glimpse network)
                # the 3rd and 4th letter(s): input-output mapping, which is clearly written in the variable name argument

                self.Wg_l_h = self.weight_variable((2, self.hl_size), "glimpseNet_wts_location_hidden", True)
                self.Bg_l_h = self.weight_variable((1, self.hl_size), "glimpseNet_bias_location_hidden", True)

                self.Wg_g_h = self.weight_variable((self.totalSensorBandwidth, self.hg_size),
                                                   "glimpseNet_wts_glimpse_hidden", True)
                self.Bg_g_h = self.weight_variable((1, self.hg_size), "glimpseNet_bias_glimpse_hidden", True)

                self.Wg_hg_gf1 = self.weight_variable((self.hg_size, self.g_size),
                                                      "glimpseNet_wts_hiddenGlimpse_glimpseFeature1", True)
                self.Wg_hl_gf1 = self.weight_variable((self.hl_size, self.g_size),
                                                      "glimpseNet_wts_hiddenLocation_glimpseFeature1", True)
                self.Bg_hlhg_gf1 = self.weight_variable((1, self.g_size),
                                                        "glimpseNet_bias_hGlimpse_hLocs_glimpseFeature1", True)

                self.Wc_g_h = self.weight_variable((self.cell_size, self.g_size), "coreNet_wts_glimpse_hidden", True)
                self.Bc_g_h = self.weight_variable((1, self.g_size), "coreNet_bias_glimpse_hidden", True)

                self.Wr_h_r = self.weight_variable((self.cell_out_size, self.img_size ** 2),
                                                   "reconstructionNet_wts_hidden_action", True)
                self.Br_h_r = self.weight_variable((1, self.img_size ** 2), "reconstructionNet_bias_hidden_action",
                                                   True)

                self.Wb_h_b = self.weight_variable((self.g_size, 1), "baselineNet_wts_hiddenState_baseline", True)
                self.Bb_h_b = self.weight_variable((1, 1), "baselineNet_bias_hiddenState_baseline", True)

                self.Wl_h_l = self.weight_variable((self.cell_out_size, 2), "locationNet_wts_hidden_location", True)
                self.Bl_h_l = self.weight_variable((1, 2), "locationNet_bias_hidden_location", True)

                self.Wa_h_a = self.weight_variable((self.cell_out_size, self.n_classes), "actionNet_wts_hidden_action",
                                                   True)
                self.Ba_h_a = self.weight_variable((1, self.n_classes), "actionNet_bias_hidden_action", True)

                # query the model ouput
                outputs = self.model()

                # convert list of tensors to one big tensor
                self.sampled_locs = tf.concat(axis=0, values=self.sampled_locs)
                self.sampled_locs = tf.reshape(self.sampled_locs, (self.nGlimpses, self.batch_size, 2))
                self.sampled_locs = tf.transpose(self.sampled_locs, [1, 0, 2])
                self.mean_locs = tf.concat(axis=0, values=self.mean_locs)
                self.mean_locs = tf.reshape(self.mean_locs, (self.nGlimpses, self.batch_size, 2))
                self.mean_locs = tf.transpose(self.mean_locs, [1, 0, 2])
                self.glimpse_images = tf.concat(axis=0, values=self.glimpse_images)

                # compute the reward
                # reconstructionCost, reconstruction, train_op_r = self.preTrain(outputs)
                cost, reward, predicted_labels, correct_labels, train_op, b, avg_b, rminusb, lr = \
                    self.calc_reward(outputs)

                ####################################### START RUNNING THE MODEL #######################################

                sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
                sess_config.gpu_options.allow_growth = True
                sess = tf.Session(config=sess_config)

                saver = tf.train.Saver()
                b_fetched = np.zeros((self.batch_size, (self.nGlimpses) * 2))

                init = tf.global_variables_initializer()
                sess.run(init)

                # iterations per epoch except last batch
                iterations_per_epoch = (train_dataset.num_examples // self.batch_size)
                print("iterations_per_epoch: " + str(iterations_per_epoch))

                # fig = plt.figure(1)
                # txt = fig.suptitle("-", fontsize=36, fontweight='bold')
                # plt.ion()
                # plt.show()
                # plt.subplots_adjust(top=0.7)
                # plotImgs = []
                iter = 0
                # training
                for epoch in range(0, self.max_epochs):
                    for batch_idx in range(0, train_dataset.batch_count()):

                        start_time = time.time()

                        # get the next batch of examples
                        nextX, nextY = train_dataset.next_batch(batch_idx)
                        nextX_orig = nextX
                        if self.translateMnist:
                            nextX, nextX_coord = self.convertTranslated(nextX, self.ORIG_IMG_SIZE, self.img_size)

                        feed_dict = {self.inputs_placeholder: nextX, self.labels_placeholder: nextY,
                                     self.onehot_labels_placeholder: self.dense_to_one_hot(nextY,
                                                                                           num_classes=self.n_classes)}

                        fetches = [train_op, cost, reward, predicted_labels, correct_labels, self.glimpse_images, avg_b,
                                   rminusb, self.mean_locs, self.sampled_locs, self.lr]
                        # feed them to the model
                        results = sess.run(fetches, feed_dict=feed_dict)

                        _, cost_fetched, reward_fetched, prediction_labels_fetched, correct_labels_fetched, \
                        glimpse_images_fetched, avg_b_fetched, rminusb_fetched, mean_locs_fetched, sampled_locs_fetched, lr_fetched = results

                        duration = time.time() - start_time

                        if iter % 50 == 0:
                            print(('Step %d: cost = %.5f reward = %.5f (%.3f sec) b = %.5f R-b = %.5f, LR = %.5f'
                                   % (iter, cost_fetched, reward_fetched, duration, avg_b_fetched, rminusb_fetched,
                                      lr_fetched)))
                            # f_glimpse_images = np.reshape(glimpse_images_fetched, \
                            #                               (
                            #                                   self.nGlimpses, self.batch_size, self.depth,
                            #                                   self.sensorBandwidth,
                            #                                   self.sensorBandwidth))
                            #
                            # fillList = False
                            # if len(plotImgs) == 0:
                            #     fillList = True
                            #
                            # # display the first image in the in mini-batch
                            # nCols = self.depth + 1
                            # plt.subplot2grid((self.depth, nCols), (0, 1), rowspan=self.depth, colspan=self.depth)
                            # # display the entire image
                            # self.plotWholeImg(nextX[0, :], self.img_size, sampled_locs_fetched)
                            #
                            # # display the glimpses
                            # for y in range(self.nGlimpses):
                            #     txt.set_text('Epoch: %.6d \nPrediction: %i -- Truth: %i\nStep: %i/%i'
                            #                  % (iter, prediction_labels_fetched[0], correct_labels_fetched[0],
                            #                     (y + 1),
                            #                     self.nGlimpses))
                            #
                            #     for x in range(self.depth):
                            #         plt.subplot(self.depth, nCols, 1 + nCols * x)
                            #         if fillList:
                            #             plotImg = plt.imshow(f_glimpse_images[y, 0, x], cmap=plt.get_cmap('gray'),
                            #                                  interpolation="nearest")
                            #             plotImg.autoscale()
                            #             plotImgs.append(plotImg)
                            #         else:
                            #             plotImgs[x].set_data(f_glimpse_images[y, 0, x])
                            #             plotImgs[x].autoscale()
                            #     fillList = False
                            #
                            #     # fig.canvas.draw()
                            #     time.sleep(1.15)
                            #     plt.pause(0.003)

                        iter += 1

                    if iter % iterations_per_epoch == 0:
                        print("EPOCH: " + str(epoch))
                        saver.save(sess, self.ckpt_path)
                        self.evaluate(val_dataset, sess, reward, predicted_labels, correct_labels,
                                      glimpse_images=self.glimpse_images)

                        train_dataset.on_epoch_end()
                        val_dataset.on_epoch_end()

                sess.close()

    def predict(self, dataset: Dataset, loaded_model=False, batch_size=64, draw=False):
        with tf.device('/cpu:1'):
            with tf.Graph().as_default() as g:
                # resource prellocation
                self.batch_size = batch_size
                self.mean_locs = []  # expectation of locations
                self.sampled_locs = []  # sampled locations ~N(mean_locs[.], loc_sd)
                self.baselines = []  # baseline, the value prediction
                self.glimpse_images = []  # to show in window

                # set the learning rate
                self.global_step = tf.Variable(0, trainable=False)
                self.lr = tf.train.exponential_decay(self.initLr, self.global_step, self.lrDecayFreq, self.lrDecayRate,
                                                     staircase=True)

                # preallocate x, y, baseline
                labels = tf.placeholder("float32", shape=[self.batch_size, self.n_classes])
                self.labels_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size), name="labels_raw")
                self.onehot_labels_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.n_classes),
                                                                name="labels_onehot")
                self.inputs_placeholder = tf.placeholder(tf.float32,
                                                         shape=(self.batch_size, self.img_size * self.img_size),
                                                         name="images")

                # declare the model parameters, here're naming rule:
                # the 1st captical letter: weights or bias (W = weights, B = bias)
                # the 2nd lowercase letter: the network (e.g.: g = glimpse network)
                # the 3rd and 4th letter(s): input-output mapping, which is clearly written in the variable name argument

                self.Wg_l_h = self.weight_variable((2, self.hl_size), "glimpseNet_wts_location_hidden", True)
                self.Bg_l_h = self.weight_variable((1, self.hl_size), "glimpseNet_bias_location_hidden", True)

                self.Wg_g_h = self.weight_variable((self.totalSensorBandwidth, self.hg_size),
                                                   "glimpseNet_wts_glimpse_hidden", True)
                self.Bg_g_h = self.weight_variable((1, self.hg_size), "glimpseNet_bias_glimpse_hidden", True)

                self.Wg_hg_gf1 = self.weight_variable((self.hg_size, self.g_size),
                                                      "glimpseNet_wts_hiddenGlimpse_glimpseFeature1", True)
                self.Wg_hl_gf1 = self.weight_variable((self.hl_size, self.g_size),
                                                      "glimpseNet_wts_hiddenLocation_glimpseFeature1", True)
                self.Bg_hlhg_gf1 = self.weight_variable((1, self.g_size),
                                                        "glimpseNet_bias_hGlimpse_hLocs_glimpseFeature1", True)

                self.Wc_g_h = self.weight_variable((self.cell_size, self.g_size), "coreNet_wts_glimpse_hidden", True)
                self.Bc_g_h = self.weight_variable((1, self.g_size), "coreNet_bias_glimpse_hidden", True)

                self.Wr_h_r = self.weight_variable((self.cell_out_size, self.img_size ** 2),
                                                   "reconstructionNet_wts_hidden_action", True)
                self.Br_h_r = self.weight_variable((1, self.img_size ** 2), "reconstructionNet_bias_hidden_action",
                                                   True)

                self.Wb_h_b = self.weight_variable((self.g_size, 1), "baselineNet_wts_hiddenState_baseline", True)
                self.Bb_h_b = self.weight_variable((1, 1), "baselineNet_bias_hiddenState_baseline", True)

                self.Wl_h_l = self.weight_variable((self.cell_out_size, 2), "locationNet_wts_hidden_location", True)
                self.Bl_h_l = self.weight_variable((1, 2), "locationNet_bias_hidden_location", True)

                self.Wa_h_a = self.weight_variable((self.cell_out_size, self.n_classes), "actionNet_wts_hidden_action",
                                                   True)
                self.Ba_h_a = self.weight_variable((1, self.n_classes), "actionNet_bias_hidden_action", True)

                # query the model ouput
                outputs = self.model()

                # convert list of tensors to one big tensor
                self.sampled_locs = tf.concat(axis=0, values=self.sampled_locs)
                self.sampled_locs = tf.reshape(self.sampled_locs, (self.nGlimpses, self.batch_size, 2))
                self.sampled_locs = tf.transpose(self.sampled_locs, [1, 0, 2])
                self.mean_locs = tf.concat(axis=0, values=self.mean_locs)
                self.mean_locs = tf.reshape(self.mean_locs, (self.nGlimpses, self.batch_size, 2))
                self.mean_locs = tf.transpose(self.mean_locs, [1, 0, 2])
                self.glimpse_images = tf.concat(axis=0, values=self.glimpse_images)

                # compute the reward
                # reconstructionCost, reconstruction, train_op_r = self.preTrain(outputs)
                cost, reward, predicted_labels, correct_labels, train_op, b, avg_b, rminusb, lr = \
                    self.calc_reward(outputs)

                saver = tf.train.Saver()
                sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
                sess_config.gpu_options.allow_growth = True
                sess = tf.Session(config=sess_config)
                saver.restore(sess, self.ckpt_path)

                self.evaluate(dataset, sess, reward, predicted_labels, correct_labels, self.glimpse_images, draw=draw)

                sess.close()
