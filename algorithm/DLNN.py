from consts import *
from decision_list import DecisionList, Conjunction

import tensorflow as tf
import pickle 
import random
import os
import numpy as np

def run_once(train_set_size, test_set_size, result_path):
	# create working dir
	run_title = 'DLNN_N={0}_K={1}_D={2}'.format(N, K, D, train_set_size, test_set_size)
	dir_path = os.path.join(result_path, run_title)
	os.makedirs(dir_path)

	# Load all data
	train_set = pickle.load(open(os.path.join(result_path, TRAIN_SET_FILE_NAME), 'rb'))
	test_set = pickle.load(open(os.path.join(os.path.join(result_path, '..'), TEST_SET_FILE_NAME), 'rb'))
	
	with tf.Graph().as_default():
		# init variables
		global_step = tf.Variable(0, trainable=False, name='global_step')
		X = tf.placeholder(tf.float32, name='X', shape=[BATCH_SIZE_DLNN, N])
		Y = tf.placeholder(tf.float32, name='Y', shape=[BATCH_SIZE_DLNN])
		W = tf.get_variable('W', shape=[D - 1, N], initializer=tf.zeros_initializer())
		V = tf.get_variable('V', shape=[D], initializer=tf.zeros_initializer())
		#V = tf.constant([POSITIVE] * K - 1, [NEGATIVE], name='V', dtype=np.float32)
		#scale_vector = np.array([1 / (N * (i + 2)) for i in range(D)], dtype=np.float32)
		#O = tf.constant(scale_vector, name='O')
		O = tf.get_variable('O', shape=[D], initializer=tf.zeros_initializer())
		a = tf.get_variable('a', shape=[1], initializer=tf.ones_initializer())
		b = tf.get_variable('b', shape=[1], initializer=tf.ones_initializer())
		# clac logits
		logits = None
		for i in range(BATCH_SIZE_DLNN):
			L_1 = tf.norm(W, ord=1, axis=1)	
			conjection_result =  tf.reduce_sum(W * X[i], 1) / a
			list_restult =tf.concat([conjection_result, b], axis=0)
			list_choise = tf.nn.softmax((list_restult + O) * BETA)
			sample_logits = tf.reduce_sum(list_choise  * V)
			if logits is None:
				logits = [sample_logits]
			else:
				logits = tf.concat([logits, [sample_logits]], axis=0)
	
		# calc accuracy
		prediction = tf.sign(logits)
		ones = tf.constant(np.ones([BATCH_SIZE_DLNN]), dtype=tf.float32)
		zeros = tf.constant(np.zeros([BATCH_SIZE_DLNN]), dtype=tf.float32)
		correct = tf.where(tf.equal(prediction, Y), ones, zeros)
		accuracy = tf.reduce_mean(correct)
	
		# calc loss
		loss = tf.reduce_mean(tf.square(logits - Y))
		# set optimizer
		optimizer = tf.train.GradientDescentOptimizer(INITIAL_LEARNING_RATE_DLNN)
		#optimizer = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE_DLNN)
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
				idx = random.sample(range(train_set_size), BATCH_SIZE_DLNN)
				x = [train_set[i][0] for i in range(train_set_size) if i in idx]
				y = [train_set[i][1] for i in range(train_set_size) if i in idx]
				_, train_loss, train_acc, W_val, V_val, gradient_val = sess.run([train_op, loss, accuracy, W, V, gradient], {X:x, Y:y})
				if step % DISPLAY == 0:
					print('step: {0}, loss: {1}, accuracy: {2}'.format(step, train_loss, train_acc))
				if step % TEST == 0:
					# test Train set
					X_steps.append(step)
					total_loss, total_accuracy = 0.0, 0.0
					train_sample_num = round(train_set_size / BATCH_SIZE_DLNN)
					for i in range(0, train_set_size, BATCH_SIZE_DLNN):
						x = [s[0] for s in train_set[i : i + BATCH_SIZE_DLNN]]
						y = [s[1] for s in train_set[i : i + BATCH_SIZE_DLNN]]
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
					test_sample_num = round(test_set_size / BATCH_SIZE_DLNN)
					for i in range(0, test_set_size, BATCH_SIZE_DLNN):
						x = [s[0] for s in test_set[i : i + BATCH_SIZE_DLNN]]
						y = [s[1] for s in test_set[i : i + BATCH_SIZE_DLNN]]
						test_loss, test_acc = sess.run([loss, accuracy], {X:x, Y:y})
						total_loss += test_loss
						total_accuracy += test_acc
					avg_loss = total_loss / test_sample_num
					avg_accuracy = total_accuracy / test_sample_num
					Y_test.append(avg_loss)
					print("Test loss: {0}, Test accuracy: {1}".format(avg_loss, avg_accuracy))
					final_test_loss = avg_loss
					final_test_accuracy = avg_accuracy

	pickle.dump(W_val, open(os.path.join(dir_path, W_FILE_NAME), 'wb'))
	pickle.dump(V_val, open(os.path.join(dir_path, V_FILE_NAME), 'wb'))

	return final_train_accuracy, final_test_accuracy	

