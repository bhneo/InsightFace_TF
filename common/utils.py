import callbacks

import tensorflow as tf
from tensorflow import keras


def get_step_lr_callback(steps, decay=0.1, on_batch=False):
    def lr_schedule(step, lr):
        for i in steps:
            if step == i:
                lr *= decay
        return lr
    return get_lr_callback(lr_schedule, on_batch)


def get_lr_callback(schedule, on_batch=False):
    if on_batch:
        return callbacks.LearningRateSchedulerOnStep(schedule)
    else:
        return keras.callbacks.LearningRateScheduler(schedule)

