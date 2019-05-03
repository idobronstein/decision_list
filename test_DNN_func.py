from consts import *
from decision_list import DecisionList, Conjunction

import tensorflow as tf
import pickle 
import random

import numpy as np

# Load all data
train_set = pickle.load(open(r"D:\checkouts\decision_lists\results\2019-04-24_09-43-35\trainSetSize=2000_testSetSize=1000\train_set.pkl", 'rb'))
test_set = pickle.load(open(r"D:\checkouts\decision_lists\results\2019-04-24_09-43-35\trainSetSize=2000_testSetSize=1000\test_set.pkl", 'rb'))
dl = pickle.load(open(r"D:\checkouts\decision_lists\results\2019-04-24_09-43-35\decision_list.pkl", 'rb'))
W_init, V_init = dl.get_matrix_representation()


with tf.Graph().as_default():
	# init variables
	global_step = tf.Variable(0, trainable=False, name='global_step')
	X = tf.placeholder(tf.float32, name='X', shape=[BATCH_SIZE_DLNN, N])
	Y = tf.placeholder(tf.float32, name='Y', shape=[BATCH_SIZE_DLNN])
	W = tf.get_variable('W', initializer=tf.convert_to_tensor(W_init))
	V = tf.get_variable('V', initializer=tf.convert_to_tensor(V_init))
	scale_vector = np.array([1 / (N * (i + 2)) for i in range(D)], dtype=np.float32)
	O = tf.constant(scale_vector, name='O')

	# clac logits
	logits = None
	for i in range(BATCH_SIZE_DLNN):	
		L_1 = tf.norm(W, ord=1, axis=1)	
		conjection_result =  tf.reduce_sum(W * X[i], 1) / (L_1 + EPSILON)
		list_restult =tf.concat([conjection_result, [1]], axis=0)
		list_choise = tf.nn.softmax((list_restult + O) * BETA)
		sample_logits = tf.reduce_mean(list_choise  * V)
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

	with tf.Session() as sess:
		# init params
		init = tf.initialize_all_variables()
		sess.run(init)

		# test 
		total_loss, total_accuracy = 0.0, 0.0
		test_sample_num = round(len(train_set) / BATCH_SIZE_DLNN)
		for i in range(0, len(train_set), BATCH_SIZE_DLNN):
			x = [s[0] for s in train_set[i : i + BATCH_SIZE_DLNN]]
			y = [s[1] for s in train_set[i : i + BATCH_SIZE_DLNN]]
			test_loss, test_acc = sess.run([loss, accuracy], {X:x, Y:y})
			total_loss += test_loss
			total_accuracy += test_acc
		avg_loss = total_loss / test_sample_num
		avg_accuracy = total_accuracy / test_sample_num
		print("Test loss: {0}, Test accuracy: {1}".format(avg_loss, avg_accuracy))