from consts import *
from decision_list import DecisionList, Conjunction

import tensorflow as tf
import pickle 
import random
import os
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np

shift_label = lambda x: 1 if x == POSITIVE else 0

def run_once(train_set_size, test_set_size, result_path):
	# create working dir
	run_title = 'DNN_N={0}_K={1}_D={2}'.format(N, K, D, train_set_size, test_set_size)
	dir_path = os.path.join(result_path, run_title)
	os.makedirs(dir_path)

	# Load all data
	train_set = pickle.load(open(os.path.join(result_path, TRAIN_SET_FILE_NAME), 'rb'))
	test_set = pickle.load(open(os.path.join(os.path.join(result_path, '..'), TEST_SET_FILE_NAME), 'rb'))
	
	# set layer size

	with tf.Graph().as_default():
		# init variables
		global_step = tf.Variable(0, trainable=False, name='global_step')
		X = tf.placeholder(tf.float32, name='X', shape=[BATCH_SIZE_DNN, N])
		Y = tf.placeholder(tf.float32, name='Y', shape=[BATCH_SIZE_DNN])
		W_1 = tf.get_variable('W_1', shape=[N, LAYER_SIZE], initializer=tf.contrib.layers.xavier_initializer())
		B_1 = tf.get_variable('B_1', shape=[LAYER_SIZE], initializer=tf.zeros_initializer())
		W_2 = tf.get_variable('W_2', shape=[LAYER_SIZE], initializer=tf.contrib.layers.xavier_initializer())
		B_2 = tf.get_variable('B_2', shape=[1], initializer=tf.zeros_initializer())
		
		out_1 = tf.nn.relu(tf.matmul(X, W_1) + B_1)
		logits = tf.tensordot(out_1, W_2, 1) + B_2

		# calc accuracy
		prediction = tf.round(tf.nn.sigmoid(logits))
		ones = tf.constant(np.ones([BATCH_SIZE_DNN]), dtype=tf.float32)
		zeros = tf.constant(np.zeros([BATCH_SIZE_DNN]), dtype=tf.float32)
		correct = tf.where(tf.equal(prediction, Y), ones, zeros)
		accuracy = tf.reduce_mean(correct)

		# calc loss
		loss_vec = tf.losses.hinge_loss(logits=logits, labels=Y)
		loss = tf.reduce_mean(loss_vec)
		
		# set optimizer
		#optimizer = tf.train.GradientDescentOptimizer(INITIAL_LEARNING_RATE)
		optimizer = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE_DNN)
		gradient = optimizer.compute_gradients(loss)
		train_op = optimizer.apply_gradients(gradient, global_step=global_step)
	
		with tf.Session() as sess:
			# init params
			init = tf.initialize_all_variables()
			sess.run(init)
	
			X_steps = []
			Y_train = []
			Y_test = []
			# train
			for step in range(MAX_STEP):
				idx = random.sample(range(train_set_size), BATCH_SIZE_DNN)
				x = [train_set[i][0] for i in range(train_set_size) if i in idx]
				y = [shift_label(train_set[i][1]) for i in range(train_set_size) if i in idx]
				_, train_loss, train_acc, val_W_1,  val_B_1, val_W_2, val_B_2 = sess.run([train_op, loss, accuracy, W_1,  B_1, W_2, B_2], {X:x, Y:y})
				if step % DISPLAY == 0:
					print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc))
					#print(val_W_1)
					#print(val_B_1)
					#print(val_W_2)
					#print(val_B_2)
				if step % TEST == 0:
					# test Train set
					X_steps.append(step)
					total_loss, total_accuracy = 0.0, 0.0
					train_sample_num = round(train_set_size / BATCH_SIZE_DNN)
					for i in range(0, train_set_size, BATCH_SIZE_DNN):
						x = [s[0] for s in train_set[i : i + BATCH_SIZE_DNN]]
						y = [shift_label(s[1]) for s in train_set[i : i + BATCH_SIZE_DNN]]
						train_loss, train_acc = sess.run([loss, accuracy], {X:x, Y:y})
						total_loss += train_loss
						total_accuracy += train_acc
					avg_loss = total_loss / train_sample_num
					avg_accuracy = total_accuracy / train_sample_num
					Y_train.append(avg_loss)
					print("Train loss: {0}, Trian accuracy: {1}".format(avg_loss, avg_accuracy))
					final_train_loss =  avg_loss
					final_train_accuracy = avg_accuracy

					# test Test set
					total_loss, total_accuracy = 0.0, 0.0
					test_sample_num = round(test_set_size / BATCH_SIZE_DNN)
					for i in range(0, test_set_size, BATCH_SIZE_DNN):
						x = [s[0] for s in test_set[i : i + BATCH_SIZE_DNN]]
						y = [shift_label(s[1]) for s in test_set[i : i + BATCH_SIZE_DNN]]
						test_loss, test_acc = sess.run([loss, accuracy], {X:x, Y:y})
						total_loss += test_loss
						total_accuracy += test_acc
					avg_loss = total_loss / test_sample_num
					avg_accuracy = total_accuracy / test_sample_num
					Y_test.append(avg_loss)
					print("Test loss: {0}, Test accuracy: {1}".format(avg_loss, avg_accuracy))
					final_test_loss = avg_loss
					final_test_accuracy = avg_accuracy

	plt.title(run_title)
	plt.plot(X_steps, Y_train, 'ro')
	plt.plot(X_steps, Y_test, 'bo')
	plt.savefig(os.path.join(dir_path, run_title + '.png'))
	plt.clf()
	
	return final_train_accuracy, final_test_accuracy	

