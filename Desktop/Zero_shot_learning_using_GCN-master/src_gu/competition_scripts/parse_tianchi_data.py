# Tianchi competition：zero-shot learning compet
# Team: AILAB-ZJU
# Code function：parse data given by Tianchi zero-shot learning competition
# Author: Youzhi Gu


import numpy as np


def parse_train_image2represent_label_map(filepath):
    image2represent_label = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            image_name = line.split('	')[0]
            image_represent_label = line.split('	')[1].replace('\n', '')
            image2represent_label[image_name] = image_represent_label

    return image2represent_label


def parse_test_image_list(filepath):
    test_images = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            test_images.append(line.replace('\n', ''))

    return test_images


def parse_represent_label2true_label_map(filepath):
    represent_label2true_label = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            represent_label = line.split('	')[0]
            true_label = line.split('	')[1].replace('\n', '')
            represent_label2true_label[represent_label] = true_label

    return represent_label2true_label


def parse_attribute_list(filepath):
    attributes = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            attr_item = line.split('	')[1].replace('\n', '')
            attributes.append(attr_item)

    return attributes


def parse_attribute_per_class(filepath):
    attribute_per_class = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            class_represent = line.replace('\n', '').split('	')[0]
            attribute_vec = line.replace('\n', '').split('	')[1:]
            attribute_vec_float = []
            for item in attribute_vec:
                attribute_vec_float.append(float(item))
            attribute_per_class[class_represent] = attribute_vec_float

    return attribute_per_class


def parse_word_embedding_per_class(filepath):
    word_embedding_per_class = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            class_true_label = line.replace('\n', '').split(' ')[0]
            word_embedding_vec = line.replace('\n', '').split(' ')[1:]
            word_embedding_vec_float = []
            for item in word_embedding_vec:
                word_embedding_vec_float.append(float(item))
                word_embedding_per_class[class_true_label] = word_embedding_vec_float

    return word_embedding_per_class


def parse_repre_label2num_label_map(filepath):
    repre_label2num_label_map = {}
    label_list = parse_represent_label2true_label_map(filepath)
    i = 0
    for repre_label in label_list.keys():
        repre_label2num_label_map[repre_label] = i
        i += 1
    print('i:', i)
    return repre_label2num_label_map


def tmp_test():
    train_image2represent_label_map_filepath = '../../data/DatasetA_train_20180813/train.txt'
    train_image2represent_label_map = parse_train_image2represent_label_map(train_image2represent_label_map_filepath)
    print(train_image2represent_label_map['a6394b0f513290f4651cc46792e5ac86.jpeg'])
    print('Total number of train images:', len(train_image2represent_label_map))

    represent_label2true_label_map_filepath = \
        '../../data/DatasetA_train_20180813/label_list.txt'
    represent_label2true_label_map = parse_represent_label2true_label_map(represent_label2true_label_map_filepath)
    print(represent_label2true_label_map['ZJL1'])
    print('Total number of classes:', len(represent_label2true_label_map))

    represent_label2num_label_map = parse_repre_label2num_label_map(represent_label2true_label_map_filepath)
    print(represent_label2num_label_map)
    print('Total number of classes:', len(represent_label2num_label_map))

    attribute_list_filepath = '../../data/DatasetA_train_20180813/attribute_list.txt'
    attribute_list = parse_attribute_list(attribute_list_filepath)
    print(attribute_list[0])
    print('Total number of attributes:', len(attribute_list))

    represent_label2attribute_vec_map_filepath = \
        '../../data/DatasetA_train_20180813/attributes_per_class.txt'
    represent_label2attribute_vec_map = parse_attribute_per_class(represent_label2attribute_vec_map_filepath)
    print(represent_label2attribute_vec_map['ZJL1'])
    print('Total number of class and attributes per class:',
          len(represent_label2attribute_vec_map), len(represent_label2attribute_vec_map['ZJL1']))

    parse_word_embedding_per_class_filepath = \
        '../../data/DatasetA_train_20180813/class_wordembeddings.txt'
    true_label2word_embedding_vec_map = parse_word_embedding_per_class(parse_word_embedding_per_class_filepath)
    print(true_label2word_embedding_vec_map['book'])
    print('Total number of class and word embeddings per class:',
          len(true_label2word_embedding_vec_map), len(true_label2word_embedding_vec_map['book']))

    parse_test_image_list_filepath = '../../data/DatasetA_test_20180813/DatasetA_test/image.txt'
    test_images_list = parse_test_image_list(parse_test_image_list_filepath)
    print(test_images_list[0])
    print('Total number of test images:', len(test_images_list))


if __name__ == '__main__':
    tmp_test()