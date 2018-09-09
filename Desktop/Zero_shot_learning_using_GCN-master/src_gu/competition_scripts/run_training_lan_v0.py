# Tianchi competition：zero-shot learning compet
# Team: AILAB-ZJU
# Code function：run training process
# Author: Lan Hong


import tensorflow as tf

from config_lan import FLAGS
from utils import *


def main(_):
    if FLAGS.is_train:
        if FLAGS.train_strategy == 'normal':
            train_normal()
        if FLAGS.train_strategy == 'triplet':
            train_triplet()
    else:
        if FLAGS.if_valid:
            valid()
        if FLAGS.if_test:
            test()


if __name__ == '__main__':
    tf.app.run()
