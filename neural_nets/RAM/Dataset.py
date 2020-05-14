import numpy as np


class Dataset(object):
    def __init__(self, data, labels, batch_size=64, reshape_img=True, shuffle=True):
        """Construct a DataSet. one_hot arg is used only if fake_data is true."""

        assert data.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (data.shape,
                                                       labels.shape))
        self._num_examples = data.shape[0]
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape_img:
            #assert data.shape[3] == 1
            data = data.reshape(data.shape[0],
                                data.shape[1] * data.shape[2])
        # Convert from [0, 255] -> [0.0, 1.0].
            data = data.astype(np.float32)
            data = np.multiply(data, 1.0 / 255.0)

        self._data = data
        self.indexes = np.arange(len(self._data))
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def batch_count(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.data) / self.batch_size))

    def next_batch(self, batch_idx):
        """Generate one batch of data
        :param batch_idx: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        indexes = sorted(indexes)

        # update index of the batch (doesn't support concurrency)

        # Generate data
        X = self.data[list(indexes)]
        Y = self.labels[list(indexes)]

        return X, Y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)