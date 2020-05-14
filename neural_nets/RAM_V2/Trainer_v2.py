import tensorflow.compat.v1 as tf
from neural_nets.RAM.Dataset import Dataset


class Trainer_v2(object):
    def __init__(self, model, dataset: Dataset, init_lr=1e-3):
        self._model = model
        self._dataset = dataset
        self._lr = init_lr

        self._train_op = model.get_train_op()
        self._loss_op = model.get_loss()
        self._accuracy_op = model.get_accuracy()
        self._sample_loc_op = model.layers['loc_sample']
        self._pred_op = model.layers['pred']
        self._lr_op = model.cur_lr

        self.global_iteration = 0

    def train_epoch(self, sess, epoch, summary_writer=None):
        self._model.set_is_training(True)
        cur_epoch = epoch
        iter = 0
        loss_sum = 0
        acc_sum = 0

        for batch_idx in range(0, self._dataset.batch_count()):

            self.global_iteration += 1
            iter += 1

            X, Y = self._dataset.next_batch(batch_idx)
            im = X
            label = Y
            _, loss, acc, cur_lr = sess.run(
                [self._train_op, self._loss_op, self._accuracy_op, self._lr_op],
                feed_dict={self._model.image: im,
                           self._model.label: label,
                           self._model.lr: self._lr})

            loss_sum += loss
            acc_sum += acc

            if iter % 100 == 0:
                print('Iteration: {}, loss: {:.4f}, accuracy: {:.4f}'
                      .format(self.global_iteration,
                              loss_sum * 1.0 / iter,
                              acc_sum * 1.0 / iter))

        print('epoch: {}, loss: {:.4f}, accuracy: {:.4f}, lr:{}'
              .format(cur_epoch,
                      loss_sum * 1.0 / iter,
                      acc_sum * 1.0 / iter, cur_lr))

        self._dataset.on_epoch_end()

        if summary_writer is not None:
            s = tf.Summary()
            s.value.add(tag='train/loss', simple_value=loss_sum * 1.0 / iter)
            s.value.add(tag='train/accuracy', simple_value=acc_sum * 1.0 / iter)
            summary_writer.add_summary(s, self.global_iteration)

    def valid_epoch(self, sess, dataset, summary_writer=None):
        self._model.set_is_training(False)

        step = 0
        loss_sum = 0
        acc_sum = 0

        for batch_idx in range(0, dataset.batch_count()):
            step += 1
            val_X, val_Y = dataset.next_batch(batch_idx)
            loss, acc = sess.run(
                [self._loss_op, self._accuracy_op],
                feed_dict={self._model.image: val_X,
                           self._model.label: val_Y,
                           })
            loss_sum += loss
            acc_sum += acc
        print('valid loss: {:.4f}, accuracy: {:.4f}'
              .format(loss_sum * 1.0 / step, acc_sum * 1.0 / step))

        if summary_writer is not None:
            s = tf.Summary()
            s.value.add(tag='valid/loss', simple_value=loss_sum * 1.0 / step)
            s.value.add(tag='valid/accuracy', simple_value=acc_sum * 1.0 / step)
            summary_writer.add_summary(s, self.global_iteration)

        self._model.set_is_training(True)
