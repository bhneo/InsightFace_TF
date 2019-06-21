import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K


class LearningRateSchedulerOnStep(Callback):
    """Learning rate scheduler.

    Arguments:
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0):
        super(LearningRateSchedulerOnStep, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = float(K.get_value(self.model.optimizer.lr))
            lr = self.schedule(batch, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(batch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (batch + 1, lr))

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


class FaceRecognitionTest(Callback):
    def __init__(self, extractor, valid_list, valid_name_list, period, batch_size, verbose=0):
        super(FaceRecognitionTest, self).__init__()
        self.extractor = extractor
        self.valid_list = valid_list
        self.valid_name_list = valid_name_list
        self.period = period
        self.batch_size = batch_size
        self.verbose = verbose

    def on_batch_end(self, batch, logs=None):
        if batch >= 0 and batch % self.period == 0:
            results = []
            for i in range(len(self.valid_list)):
                acc, std, x_norm, embeddings_list = self.test(self.valid_list[i], self.extractor, self.batch_size, 10)
                print('[%s][%d]XNorm: %f' % (self.valid_name_list[i], batch, x_norm))
                print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.valid_name_list[i], batch, acc, std))
                results.append(acc)

    def test(self, data_set, model, batch_size, folds):
        data_list = data_set[0]
        is_same_list = data_set[1]
        embeddings_list = []
        time_consumed = 0.0
        _label = nd.ones((batch_size,))

        for i in range(len(data_list)):
            data = data_list[i]
            embeddings = None
            ba = 0
            while ba < data.shape[0]:
                bb = min(ba + batch_size, data.shape[0])
                count = bb - ba
                _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
                time0 = datetime.datetime.now()
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
                model.forward(db, is_train=False)
                net_out = model.get_outputs()
                _embeddings = net_out[0].asnumpy()
                time_now = datetime.datetime.now()
                diff = time_now - time0
                time_consumed += diff.total_seconds()
                if embeddings is None:
                    embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
                embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
                ba = bb
            embeddings_list.append(embeddings)

        _xnorm = 0.0
        _xnorm_cnt = 0
        for embed in embeddings_list:
            for i in range(embed.shape[0]):
                _em = embed[i]
                _norm = np.linalg.norm(_em)
                _xnorm += _norm
                _xnorm_cnt += 1
        _xnorm /= _xnorm_cnt

        embeddings = embeddings_list[0] + embeddings_list[1]
        embeddings = sklearn.preprocessing.normalize(embeddings)
        print(embeddings.shape)
        print('infer time', time_consumed)
        _, _, accuracy, val, val_std, far = evaluate(embeddings, is_same_list, nrof_folds=folds)
        acc, std = np.mean(accuracy), np.std(accuracy)
        return acc, std, _xnorm, embeddings_list



