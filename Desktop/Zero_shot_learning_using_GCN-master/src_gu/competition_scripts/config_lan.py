# Tianchi competition：zero-shot learning compet
# Team: AILAB-ZJU
# Code function：configuration of zero-shot-learning baseline using resnet
# Author: Lan Hong

import os


class FLAGS(object):
    # dictory configuration
    attrs_per_class_dir = \
        '../../data/DatasetA_train_20180813/attributes_per_class.txt'
    img_dir = \
        '../../data/DatasetA_train_20180813/train/'
    train_file = \
        '../../data/DatasetA_train_20180813/train.txt'
    label_list_file = \
        '../../data/DatasetA_train_20180813/label_list.txt'

    network_def = 'resnet_152'
    version = 'version_3'

    # model configuration
    weight_decay = 0.0002
    num_residual_blocks = 25

    # pattern configuration
    is_train = False
    use_gpu = True
    pretrained_model = False
    train_strategy = 'triplet'
    gpu_id = '0'

    # augment configuration
    if_size_augment = True
    min_compress_ratio = 0.7
    max_compress_ratio = 1.35
    if_rotate_augment = True
    max_rotation = 180
    if_color_augment = True
    color_augment_choices = [0.6, 0.8, 1.2, 1.4]

    # input configuration
    img_size = 64
    img_width = 64
    img_height = 64
    img_depth = 3

    # train configuration
    training_epoch = 50
    batch_size = 256
    attribute_label_cnt = 30

    # learning rate configuration
    dropout_keep_prob = 0.5
    learning_rate = 0.01
    lr_decay_rate = 0.5
    lr_decay_step = 10000
    # rms_decay = 0.9
    fixed_lr = 1e-5

    # triplet loss configuration
    triplet_strategy = 'batch_hard'
    squared = False
    margin = 1.0

    # subprocess configuration
    validation_interval = 20

    # test configuration
    test_img_dir = '../../data/DatasetA_test_20180813/DatasetA_test/test'
    test_file = '../../data/DatasetA_test_20180813/DatasetA_test/image.txt'
    if_test = True  # if is_train = False
    if_valid = False    # if is_train = False

    experiment_id = '1'
