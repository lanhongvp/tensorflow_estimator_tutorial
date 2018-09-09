# Tianchi competition：zero-shot learning compet
# Team: AILAB-ZJU
# Code function：run training process
# Author: Lan Hong

# from resnet import *
import tensorflow as tf
import os
import resnet_v0 as model
import time
import csv
import numpy as np
import cv2

from config_lan import FLAGS
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tianchi_data_preprocess_lan import DataGenerator
from parse_tianchi_data import *


def train_normal():
	'''
	Step 1: Create dirs for saving models and logs
	'''
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

	model_path_suffix = os.path.join(FLAGS.network_def + '_' + FLAGS.version)
	model_save_dir = os.path.join('../../data/results_normal/model_weights', model_path_suffix)
	train_log_save_dir = os.path.join('../../data/results_normal/logs', model_path_suffix, 'train')
	test_log_save_dir = os.path.join('../../data/results_normal/logs', model_path_suffix, 'val')

	os.system('mkdir -p {}'.format(model_save_dir))
	os.system('mkdir -p {}'.format(train_log_save_dir))
	os.system('mkdir -p {}'.format(test_log_save_dir))

	'''
	Step 2: Create dataset and data generator
	'''
	print('CREATE DIFFERENT DATASETS')

	dataset = DataGenerator(FLAGS.attrs_per_class_dir, FLAGS.img_dir, FLAGS.train_file)

	dataset.generate_set(rand=True, validationRate=0.0)

	# train setp configuration
	train_size = dataset.count_train()
	training_iters_per_epoch = int(train_size / FLAGS.batch_size)
	model_save_iters = int(training_iters_per_epoch / 4)
	print("train size: %d, training_iters_per_epoch: %d, model_save_iters: %d" % (train_size, training_iters_per_epoch, model_save_iters))

	generator = dataset.generator(batchSize=FLAGS.batch_size, norm=True, sample='train')
	generator_eval = dataset.generator(batchSize=FLAGS.batch_size, norm=True, sample='valid')

	image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])

	label_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.attribute_label_cnt])

	'''
	Step 3: Build network graph
	'''
	# logits = model.inference(image_placeholder, FLAGS.num_residual_blocks, reuse=False)
	logits, _ = resnet_v2.resnet_v2_152(image_placeholder, FLAGS.attribute_label_cnt, reuse=False)
	# square_loss, train = model.build_loss_3(logits, label_placeholder)
	# square_loss, train = model.build_loss(label_placeholder,logits)
	loss, train = model.build_loss_feng(logits, label_placeholder)

	'''
	Step 4: Training
	'''
	total_start_time = time.time()
	# Loop forever, alternating between training and validation.
	device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
	with tf.Session(config=tf.ConfigProto(device_count=device_count, allow_soft_placement=True)) as sess:
		# Create tensorboard
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(train_log_save_dir, sess.graph)
		validation_writer = tf.summary.FileWriter(test_log_save_dir, sess.graph)

		# Create model saver
		saver = tf.train.Saver()

		# Init all vars
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)

		# Restore pretrained weights
		if FLAGS.pretrained_model:
			pretrained_model = model_save_dir

			checkpoint = tf.train.get_checkpoint_state(pretrained_model)
			# 获取最新保存的模型检查点文件
			ckpt = checkpoint.model_checkpoint_path
			saver.restore(sess, ckpt)
			# check weights
			for variable in tf.trainable_variables():
				with tf.variable_scope('', reuse=True):
					var = tf.get_variable(variable.name.split(':0')[0])
					print(variable.name, np.mean(sess.run(var)))

		global_step = 0
		for epoch in range(FLAGS.training_epoch):

			for step in range(training_iters_per_epoch):
				# Train start
				image_data, attribute_labels, _= next(generator)
				batch_start_time = time.time()
				global_step = step + epoch * (training_iters_per_epoch)

				# if step % 20 == 0:
				summary, loss_result, _ = \
					sess.run([merged, loss, train], feed_dict={image_placeholder: image_data, label_placeholder: attribute_labels})

				train_writer.add_summary(summary, global_step)

				if step % 10 == 0:
					print('[%s][training][epoch %d, step %d / %d exec %.2f seconds]  loss : %3.10f' %
					      (time.strftime("%Y-%m-%d %H:%M:%S"), epoch + 1, step, training_iters_per_epoch, (time.time() - batch_start_time), loss_result))

			# Save models
			saver.save(sess=sess, save_path=model_save_dir + '/' + FLAGS.network_def.split('.py')[0], global_step=(global_step + 1))
			print('\nModel checkpoint saved for one epoch...\n')

		saver.save(sess=sess, save_path=model_save_dir + '/' + FLAGS.network_def.split('.py')[0], global_step=(global_step + 1))
		print('\nModel checkpoint saved for total train...\n')

	print('Training done.')
	print("[%s][total exec %s seconds" % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - total_start_time)))

	train_writer.close()


def train_triplet():
	'''
	Step 1: Create dirs for saving models and logs
	'''
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

	model_path_suffix = os.path.join(FLAGS.network_def + '_' + FLAGS.version)
	model_save_dir = os.path.join('../../data/results_triplet/model_weights', model_path_suffix)
	train_log_save_dir = os.path.join('../../data/results_triplet/logs', model_path_suffix, 'train')
	test_log_save_dir = os.path.join('../../data/results_triplet/logs', model_path_suffix, 'val')

	os.system('mkdir -p {}'.format(model_save_dir))
	os.system('mkdir -p {}'.format(train_log_save_dir))
	os.system('mkdir -p {}'.format(test_log_save_dir))

	'''
	Step 2: Create dataset and data generator
	'''
	print('CREATE DIFFERENT DATASETS')

	dataset = DataGenerator(FLAGS.attrs_per_class_dir, FLAGS.img_dir, FLAGS.train_file)

	dataset.generate_set(rand=True, validationRate=0.0)

	# train setp configuration
	train_size = dataset.count_train()
	training_iters_per_epoch = int(train_size / FLAGS.batch_size)
	model_save_iters = int(training_iters_per_epoch / 4)
	print("train size: %d, training_iters_per_epoch: %d, model_save_iters: %d" % (train_size, training_iters_per_epoch, model_save_iters))

	generator = dataset.generator(batchSize=FLAGS.batch_size, norm=True, sample='train')
	generator_eval = dataset.generator(batchSize=FLAGS.batch_size, norm=True, sample='valid')

	image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])

	num_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])

	'''
	Step 3: Build network graph
	'''
	# logits = model.inference(image_placeholder, FLAGS.num_residual_blocks, reuse=False)
	logits, _ = resnet_v2.resnet_v2_152(image_placeholder, FLAGS.attribute_label_cnt, reuse=False)
	# square_loss, train = model.build_loss_3(logits, label_placeholder)
	# square_loss, train = model.build_loss(label_placeholder,logits)
	# loss, train = model.build_loss_feng(logits, label_placeholder)
	loss, train = model.build_triplet_loss(logits, num_label_placeholder, FLAGS.margin, FLAGS.squared, FLAGS.triplet_strategy)

	'''
	Step 4: Training
	'''
	total_start_time = time.time()
	# Loop forever, alternating between training and validation.
	device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
	with tf.Session(config=tf.ConfigProto(device_count=device_count, allow_soft_placement=True)) as sess:
		# Create tensorboard
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(train_log_save_dir, sess.graph)
		validation_writer = tf.summary.FileWriter(test_log_save_dir, sess.graph)

		# Create model saver
		saver = tf.train.Saver()

		# Init all vars
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)

		# Restore pretrained weights
		if FLAGS.pretrained_model:
			pretrained_model = model_save_dir

			checkpoint = tf.train.get_checkpoint_state(pretrained_model)
			# 获取最新保存的模型检查点文件
			ckpt = checkpoint.model_checkpoint_path
			saver.restore(sess, ckpt)
			# check weights
			for variable in tf.trainable_variables():
				with tf.variable_scope('', reuse=True):
					var = tf.get_variable(variable.name.split(':0')[0])
					print(variable.name, np.mean(sess.run(var)))

		global_step = 0
		for epoch in range(FLAGS.training_epoch):

			for step in range(training_iters_per_epoch):
				# Train start
				image_data, _, num_labels = next(generator)
				batch_start_time = time.time()
				global_step = step + epoch * (training_iters_per_epoch)

				# if step % 20 == 0:
				summary, loss_result, _ = \
					sess.run([merged, loss, train], feed_dict={image_placeholder: image_data, num_label_placeholder: num_labels})

				train_writer.add_summary(summary, global_step)

				if step % 10 == 0:
					print('[%s][training][epoch %d, step %d / %d exec %.2f seconds]  loss : %3.10f' %
					      (time.strftime("%Y-%m-%d %H:%M:%S"), epoch + 1, step, training_iters_per_epoch, (time.time() - batch_start_time), loss_result))

			# Save models
			saver.save(sess=sess, save_path=model_save_dir + '/' + FLAGS.network_def.split('.py')[0], global_step=(global_step + 1))
			print('\nModel checkpoint saved for one epoch...\n')

		saver.save(sess=sess, save_path=model_save_dir + '/' + FLAGS.network_def.split('.py')[0], global_step=(global_step + 1))
		print('\nModel checkpoint saved for total train...\n')

	print('Training done.')
	print("[%s][total exec %s seconds" % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - total_start_time)))

	train_writer.close()


def valid():
	'''
	Step 1: Create dirs for saving models and logs
	'''
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

	model_path_suffix = os.path.join(FLAGS.network_def + '_' + FLAGS.version)
	test_save_dir = os.path.join('../../data/results/' + FLAGS.train_strategy + '/valid', model_path_suffix)
	# model_save_dir = os.path.join('../../data/results/' + FLAGS.train_strategy + '/model_weights', model_path_suffix)
	model_save_dir = os.path.join('../../data/results/' + '/model_weights', model_path_suffix)
	os.system('mkdir -p {}'.format(test_save_dir))

	'''
	Step 2: Create dataset and data generator
	'''
	# ####################### use train set to valid
	test_set = []
	with open(FLAGS.train_file, 'r') as f:
		for line in f.readlines():
			image_name = line.split('	')[0]
			test_set.append(image_name)

	print('READING LABELS OF TRAIN DATA')
	print('Total num:', len(test_set))
	test_set = test_set[:1000]

	# test setp configuration
	test_size = len(test_set)

	image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])

	'''
	Step 3: Build network graph
	'''
	# logits = model.inference(image_placeholder, FLAGS.num_residual_blocks, reuse=False)
	logits, _ = resnet_v2.resnet_v2_152(image_placeholder, FLAGS.attribute_label_cnt, reuse=False)

	'''
	Step 4: Testing
	'''
	total_start_time = time.time()

	represent_label2attribute_vec_map = parse_attribute_per_class(FLAGS.attrs_per_class_dir)
	print('represent_label2attribute_vec_map: ', len(represent_label2attribute_vec_map))
	
	repre_label_list = []
	attr_vec_list = []
	for repre_label in represent_label2attribute_vec_map.keys():
		print('REPER_LABEL',repre_label)
		repre_label_list.append(repre_label)
		attr_vec_list.append(represent_label2attribute_vec_map[repre_label])
	print('attribute_vec2represent_label_map: ', len(repre_label_list), len(attr_vec_list))
	print('REPRE_LABEL_LIST',repre_label_list)
	print('ATTR_VEC_LIST',attr_vec_list)

	# ####################### use train set to valid
	train_image2represent_label_map = parse_train_image2represent_label_map(FLAGS.train_file)
	print('train file', len(train_image2represent_label_map))

	device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
	with tf.Session(config=tf.ConfigProto(device_count=device_count, allow_soft_placement=True)) as sess:
		# Create model saver
		saver = tf.train.Saver()

		# Init all vars
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)

		
		# Restore pretrained weights
		pretrained_model = model_save_dir
		checkpoint = tf.train.get_checkpoint_state(pretrained_model)
		ckpt = checkpoint.model_checkpoint_path             # 获取最新保存的模型检查点文件
		saver.restore(sess, ckpt)
		for variable in tf.trainable_variables():           # check weights
			with tf.variable_scope('', reuse=True):
				var = tf.get_variable(variable.name.split(':0')[0])
				print(variable.name, np.mean(sess.run(var)))		


		# Test start
		total_num = 0
		accurate_num = 0
		step = 0
		while True:
			if step < test_size:
				gt_label = []
				image_name = test_set[step: step + 16]
				# print('IMAGE_NAME',image_name)
				step = step + 16
				image_num = len(image_name)
				print('image num', image_num)

				image_data = np.zeros((image_num, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth), dtype=np.float32)
				for i in range(image_num):
					image_data[i, :, :, :] = open_img(is_train=True, name=image_name[i], color='RGB')

					# ####################### use train set to valid
					gt_label.append(train_image2represent_label_map[image_name[i]])
				
				batch_start_time = time.time()

				pred_logits = sess.run([logits], feed_dict={image_placeholder: image_data})
				# print('PRED_LOGITS_0',pred_logits)
				pred_logits = np.array(pred_logits).squeeze()
				# print('PRED_LOGITS_1',pred_logits)
				print('pred logits: ', pred_logits.shape)

				pred_label = []
				for image_index in range(image_num):
					distance_list = []
					for attr_vec in attr_vec_list:
						distance = np.sum(np.square(pred_logits[image_index] - attr_vec))
						distance_list.append(distance)

					distance_np = np.array(distance_list)

					print('distance_np: ', distance_np.shape)
					min_index = np.argmin(distance_np)
					print('min index: ', min_index, repre_label_list[min_index], attr_vec_list[min_index])
					pred_label.append(repre_label_list[min_index])

				print('DISTANCE_LIST',len(distance_list))
				print('DISTANCE_NP',len(distance_np))
				print('pred_label:', pred_label, len(pred_label))
				print('GT_LABEL',gt_label)

				for i in range(image_num):
					total_num += 1
					if gt_label[i] == pred_label[i]:
						accurate_num += 1
				accuracy = accurate_num / total_num

				print('[%s][testing %d][step %d / %d exec %.2f seconds]  Accuracy : %3.10f' %
				      (time.strftime("%Y-%m-%d %H:%M:%S"), image_num, step, test_size, (time.time() - batch_start_time), accuracy))
			else:
				break

	print('Testing done.')
	print("[%s][total exec %s seconds" % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - total_start_time)))


def test():
	'''
	Step 1: Create dirs for saving models and logs
	'''
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

	model_path_suffix = os.path.join(FLAGS.network_def + '_' + FLAGS.version)
	test_save_dir = os.path.join('../../data/results/' + FLAGS.train_strategy + '/test', model_path_suffix)
	model_save_dir = os.path.join('../../data/results/' + FLAGS.train_strategy + '/model_weights', model_path_suffix)

	os.system('mkdir -p {}'.format(test_save_dir))

	'''
	Step 2: Create dataset and data generator
	'''
	# ####################### use train set to valid
	test_set = parse_test_image_list(FLAGS.test_file)

	# test setp configuration
	test_size = len(test_set)
	print('test size: ', test_size)

	image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])

	'''
	Step 3: Build network graph
	'''
	# logits = model.inference(image_placeholder, FLAGS.num_residual_blocks, reuse=False)
	logits, _ = resnet_v2.resnet_v2_152(image_placeholder, FLAGS.attribute_label_cnt, reuse=False)

	'''
	Step 4: Testing
	'''
	total_start_time = time.time()

	represent_label2attribute_vec_map = parse_attribute_per_class(FLAGS.attrs_per_class_dir)
	print('attribute_vec2represent_label_map: ', len(represent_label2attribute_vec_map))
	repre_label_list = []
	attr_vec_list = []
	for repre_label in represent_label2attribute_vec_map.keys():
		repre_label_list.append(repre_label)
		attr_vec_list.append(represent_label2attribute_vec_map[repre_label])
	print('attribute_vec2represent_label_map: ', len(repre_label_list), len(attr_vec_list))

	device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
	with tf.Session(config=tf.ConfigProto(device_count=device_count, allow_soft_placement=True)) as sess:
		# Create model saver
		saver = tf.train.Saver()

		# Init all vars
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)

		'''
		# Restore pretrained weights
		pretrained_model = model_save_dir
		checkpoint = tf.train.get_checkpoint_state(pretrained_model)
		ckpt = checkpoint.model_checkpoint_path             # 获取最新保存的模型检查点文件
		saver.restore(sess, ckpt)
		for variable in tf.trainable_variables():           # check weights
			with tf.variable_scope('', reuse=True):
				var = tf.get_variable(variable.name.split(':0')[0])
				print(variable.name, np.mean(sess.run(var)))
		'''

		# Test start
		step = 0
		while True:
			if step < test_size:
				gt_label = []
				image_name = test_set[step: step + 16]
				step = step + 16
				image_num = len(image_name)
				print('image num', image_num)

				image_data = np.zeros((image_num, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth), dtype=np.float32)
				for i in range(image_num):
					image_data[i, :, :, :] = open_img(is_train=False, name=image_name[i], color='RGB')

				batch_start_time = time.time()

				pred_logits = sess.run([logits], feed_dict={image_placeholder: image_data})
				pred_logits = np.array(pred_logits).squeeze()
				# print('pred logits: ', pred_logits.shape)

				pred_label = []
				for image_index in range(image_num):
					distance_list = []
					for attr_vec in attr_vec_list:
						distance = np.sum(np.square(pred_logits[image_index] - attr_vec))
						distance_list.append(distance)

					distance_np = np.array(distance_list)
					# print('distance_np: ', distance_np.shape)
					min_index = np.argmin(distance_np)
					# print('min index: ', min_index, repre_label_list[min_index], attr_vec_list[min_index])
					pred_label.append(repre_label_list[min_index])
				print('pred_label:', pred_label)

				print('[%s][testing %d][step %d / %d exec %.2f seconds]' %
				      (time.strftime("%Y-%m-%d %H:%M:%S"), image_num, step, test_size, (time.time() - batch_start_time)))
			else:
				break

	print('Testing done.')
	print("[%s][total exec %s seconds" % (time.strftime("%Y-%m-%d %H:%M:%S"), (time.time() - total_start_time)))


def open_img(is_train, name, color='RGB'):
	""" Open an image
	Args:
		name	: Name of the sample
		color	: Color Mode (RGB/BGR/GRAY)
	"""
	if is_train:
		img_dir = FLAGS.img_dir
	else:
		img_dir = FLAGS.test_img_dir
	img = cv2.imread(os.path.join(img_dir, name))
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