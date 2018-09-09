# -*- coding: utf-8 -*-
# Tianchi competition：zero-shot learning compet
# Team: AILAB-ZJU
# Code function：data generator of zero-shot-learning baseline using resnet
# Author: Lan Hong

import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm
from PIL import Image, ImageEnhance, ImageFilter
from parse_tianchi_data import *
from config_lan import FLAGS


class DataGenerator():
    """
    To process images and labels
    """
    def __init__(self, attrs_per_class_dir, img_dir, train_file):
        """Initializer
            Args:
            attrs_per_class_dir	: Attributes per class 
            img_dir				: Directory containing every images
            train_file		    : Text file with training set data

        """
        self.attrs_per_class_dir = attrs_per_class_dir
        self.img_dir = img_dir
        self.train_file = train_file

    # --------------------Generator Initialization Methods ---------------------

    def _read_train_data(self):
        """
        To read labels in csv
        """
        self.train_table = []     # The names of images being trained
        self.train_image2represent_label_map = {}
        self.represent_label2attribute_vec_map = {}
        self.data_dict = {}       # The labels of images
        
        with open(self.train_file, 'r') as f:
            for line in f.readlines():
                image_name = line.split('	')[0]
                self.train_table.append(image_name)
        # label_file = pd.read_csv(self.train_file)
        print('READING LABELS OF TRAIN DATA')
        print('Total num:', len(self.train_table))

        # obtain image name and class label in train.txt
        self.train_image2represent_label_map = \
        parse_train_image2represent_label_map(self.train_file)

        # print(self.train_image2represent_label_map)

        # obtain class label and attribute label in attrs_per_class.txt
        self.represent_label2attribute_vec_map = \
        parse_attribute_per_class(self.attrs_per_class_dir)

        self.repre_label2num_label_map = \
            parse_repre_label2num_label_map(FLAGS.label_list_file)

        # print(len(self.train_table))
        for i in range(len(self.train_table)):
            image_name = self.train_table[i]
            class_label = self.train_image2represent_label_map[image_name]
            attribute_label = self.represent_label2attribute_vec_map[class_label]
            num_label = self.repre_label2num_label_map[class_label]
            self.data_dict[image_name] = {'attribute_label': attribute_label, 'num_label': num_label}
        
        print('LABEL READING FINISHED')
        # print(self.data_dict[image_name]['attribute_label'])
        return self.train_table, self.data_dict

    def _randomize(self):
        """ Randomize the set
        """
        random.shuffle(self.train_table)

    def generate_set(self, rand=True, validationRate=0.1):
        """ Generate the training and validation set
        Args:
            rand : (bool) True to shuffle the set
        """
        self._read_train_data()
        if rand:
            self._randomize()
        self._create_sets(validation_rate=validationRate)

    def _create_sets(self, validation_rate=0.1):
        """ Select Elements to feed training and validation set
        Args:
            validation_rate		: Percentage of validation data (in ]0,1[, don't waste time use 0.1)
        """
        self.train_dict = {}
        self.valid_dict = {}
        sample = len(self.train_table)
        valid_sample = int(sample * validation_rate)
        self.train_set = self.train_table[:sample - valid_sample]
        self.valid_set = self.train_table[sample - valid_sample:]
        # preset = self.train_table[sample - valid_sample:]
        print('START SET CREATION')

        print('SET CREATED')
        # np.save('Dataset-Validation-Set', self.valid_set)
        # np.save('Dataset-Training-Set', self.train_set)
        print('--Training set :', len(self.train_set), ' samples.')
        print('--Validation set :', len(self.valid_set), ' samples.')

        for item in range(len(self.train_set)):
            self.train_dict[self.train_set[item]] = self.data_dict[self.train_set[item]]

        for item in range(len(self.valid_set)):
            self.valid_dict[self.valid_set[item]] = self.data_dict[self.valid_set[item]]

    def _give_batch_name(self, batch_size=16, set='train'):
        """ Returns a List of Samples
        Args:
            batch_size	: Number of sample wanted
            set			: Set to use (valid/train)
        """
        list_file = []
        for i in range(batch_size):
            if set == 'train':
                list_file.append(random.choice(self.train_set))
            elif set == 'valid':
                list_file.append(random.choice(self.valid_set))
            else:
                print('Set must be : train/valid')
                break
        return list_file

    # ---------------------------- Generating Methods --------------------------
    def _rotate_augment(self, img, max_rotation=FLAGS.max_rotation):
        """ # TODO : IMPLEMENT DATA AUGMENTATION
        """
        if random.choice([0, 1]):
            r_angle = np.random.randint(-1 * max_rotation, max_rotation)
            img = transform.rotate(img, r_angle, preserve_range=True)
            # hm = transform.rotate(hm, r_angle)
        return img

    # input image size=FLAGS.img_size,out image size=FLAGS.img_size,input joints size = FLAGS.img_size
    def _size_augment(self, img, min_compress_ratio=FLAGS.min_compress_ratio, max_compress_ratio=FLAGS.max_compress_ratio):
        compress_ratio = np.random.uniform(min_compress_ratio, max_compress_ratio)
        size = compress_ratio * img.shape[0]
        size = round(size)

        # img resize
        resized_img = cv2.resize(img, (size, size))
        resized_img_shape = resized_img.shape

        if compress_ratio <= 1.0:
            # resized img padding to 512
            img_x = resized_img_shape[0]
            img_y = resized_img_shape[1]
            img2 = np.zeros((FLAGS.img_size, FLAGS.img_size, 3), dtype=np.float32)
            img_x_padding = (FLAGS.img_size - img_x) // 2
            img_y_padding = (FLAGS.img_size - img_y) // 2
            img2[img_x_padding:img_x_padding + img_x, img_y_padding:img_y_padding + img_y, :] = resized_img[:, :, :]
            aug_img = img2

        else:
            img_x = resized_img_shape[0]
            img_y = resized_img_shape[1]
            img2 = np.zeros((FLAGS.img_size, FLAGS.img_size, 3), dtype=np.float32)
            img_x_padding = (img_x - FLAGS.img_size) // 2
            img_y_padding = (img_y - FLAGS.img_size) // 2
            img2[:, :, :] = resized_img[img_x_padding:img_x_padding + FLAGS.img_size, img_y_padding:img_y_padding + FLAGS.img_size, :]
            aug_img = img2

        return aug_img

    def _color_augment(self, img):
        image = Image.fromarray(img)
        # image.show()
        # 亮度增强
        if random.choice([0, 1]):
            enh_bri = ImageEnhance.Brightness(image)
            brightness = random.choice(FLAGS.color_augment_choices)
            image = enh_bri.enhance(brightness)
            # image.show()

        # 色度增强
        if random.choice([0, 1]):
            enh_col = ImageEnhance.Color(image)
            color = random.choice(FLAGS.color_augment_choices)
            image = enh_col.enhance(color)
            # image.show()

        # 对比度增强
        if random.choice([0, 1]):
            enh_con = ImageEnhance.Contrast(image)
            contrast = random.choice(FLAGS.color_augment_choices)
            image = enh_con.enhance(contrast)
            # image.show()

        # 锐度增强
        if random.choice([0, 1]):
            enh_sha = ImageEnhance.Sharpness(image)
            sharpness = random.choice(FLAGS.color_augment_choices)
            image = enh_sha.enhance(sharpness)
            # image.show()

        # mo hu
        if random.choice([0, 1]):
            image = image.filter(ImageFilter.BLUR)

        img = np.asarray(image)
        return img

        # ----------------------- Batch Random Generator ----------------------------------
    def _aux_generator(self, batch_size=16, normalize=True, sample_set='train'):
        """ Auxiliary Generator
        Args:
            See Args section in self._generator
        """
        while True:
            train_img = np.zeros((batch_size, FLAGS.img_size, FLAGS.img_size, 3), dtype=np.float32)
            attribute_labels = np.zeros((batch_size, 30), dtype=np.float32)
            num_labels = np.zeros((batch_size), dtype=np.int32)
            i = 0
            while i < batch_size:
                if sample_set == 'train':
                    name = random.choice(self.train_set)
                else:
                    name = random.choice(self.valid_set)

                # 读图片
                # print(name)
                img = self.open_img(name)

                # color aug
                if FLAGS.if_color_augment:
                    if random.choice([0, 1]):
                        img = self._color_augment(img)

                # augment size
                # TODO: size augment
                if FLAGS.if_size_augment:
                    if random.choice([0, 1]) == 1:
                        # new_j = self._relative_joints(cbox, padd, joints, to_size=FLAGS.img_size)
                        img = self._size_augment(img)
                        # hm = self._generate_hm(FLAGS.img_size, FLAGS.img_size, dress_type, new_j, FLAGS.img_size, train_weights[i])
                    else:
                        img = img

                # rotate augmentation
                if FLAGS.if_rotate_augment:
                    img = self._rotate_augment(img)

                attribute_labels[i] = self.data_dict[name]['attribute_label']
                num_labels[i] = self.data_dict[name]['num_label']
                # print('attribute_labels', attribute_labels[i])
                # print(num_labels[i])

                if normalize:
                    train_img[i] = img.astype(np.float32) / 255.0
                else:
                    train_img[i] = img.astype(np.float32)

                # cv2.imshow('train image', train_img[i])
                # cv2.waitKey(1000)

                i = i + 1
            yield train_img, attribute_labels, num_labels

    def generator(self, batchSize=16, norm=True, sample='train'):
        """ Create a Sample Generator
        Args:
            batchSize 	: Number of image per batch
            stacks 	 	: Stacks in HG model
            norm 	 	 	: (bool) True to normalize the batch
            sample 	 	: 'train'/'valid' Default: 'train'
        """
        return self._aux_generator(batch_size=batchSize, normalize=norm, sample_set=sample)

    # ---------------------------- Image Reader --------------------------------
    def open_img(self, name, color='RGB'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        img = cv2.imread(os.path.join(self.img_dir, name))
        # arr = np.asarray(img,dtype="float32")
        img = cv2.resize(img, (FLAGS.img_size, FLAGS.img_size))
        # print('img_shape: ', img.shape)

        if len(img.shape) == 2:
            temp = np.empty((FLAGS.img_size, FLAGS.img_size, 3))
            temp[:, :, 0] = img
            temp[:, :, 1] = img
            temp[:, :, 2] = img
            img = temp

        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

    def plot_img(self, name, plot='cv2'):
        """ Plot an image
        Args:
            name	: Name of the Sample
            plot	: Library to use (cv2: OpenCV, plt: matplotlib)
        """
        if plot == 'cv2':
            img = self.open_img(name, color='BGR')
            cv2.imshow('Image', img)
        elif plot == 'plt':
            img = self.open_img(name, color='RGB')
            plt.imshow(img)
            plt.show()

    def count_train(self):
        return len(self.train_set)


if __name__ == '__main__':
    dataset = DataGenerator(FLAGS.attrs_per_class_dir, FLAGS.img_dir, FLAGS.train_file)
    dataset.generate_set(rand=True, validationRate=0.0)
    
    generator = dataset.generator(batchSize=FLAGS.batch_size, norm=True, sample='train')
    while True:
        _, _, _ = next(generator)