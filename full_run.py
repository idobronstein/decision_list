from consts import *
import algorithm.DLNN as DLNN
import algorithm.DNN as DNN
import algorithm.CNF3 as CNF3
import algorithm.classic as classic
import algorithm.randomforest as randomforest
import algorithm.gradientboosting as gradientboosting
from decision_list import *

import pickle
import os
import numpy as np
import random
import matplotlib.pyplot as plt

def generate_sampels(decision_list, size, name, result_path, p_positive):
	print("Generating: {0} sampels ".format(size))
	samples = []
	count_positive = 0
	count_negative = 0
	for i in range(size):
		x = np.array([np.random.choice([POSITIVE, NEGATIVE], p=[p_positive, 1 - p_positive]) for _ in range(N)])
		y = decision_list.get_value(x)
		if y == NEGATIVE:
			count_negative += 1
		else:
			count_positive += 1
		samples.append((x, y))
	print("number of positive samples: {}".format(count_positive))
	print("number of negative samples: {}".format(count_negative))
	pickle.dump(samples, open(os.path.join(result_path, name), 'wb'))


def calc_combinations(current, i, all_possible):
	if i + 1 > N:
		all_possible.append(np.array(current))
	else:
		current[i] = POSITIVE
		calc_combinations(current[:], i + 1, all_possible)
		current[i] = NEGATIVE
		calc_combinations(current[:], i + 1, all_possible)

def generate_all_sampels(decision_list, name, result_path):
	all_combinations = []
	init_state = [EMPTY] * N
	calc_combinations(init_state, 0, all_combinations)
	sampels_size = len(all_combinations)
	print("Generating: {0} sampels ".format(sampels_size))
	samples = []
	count_positive = 0
	count_negative = 0
	for x in all_combinations:
		y = decision_list.get_value(x)
		if y == NEGATIVE:
			count_negative += 1
		else:
			count_positive += 1
		samples.append((x, y))
	print("number of positive samples: {}".format(count_positive))
	print("number of negative samples: {}".format(count_negative))
	pickle.dump(samples, open(os.path.join(result_path, name), 'wb'))
	return sampels_size

def create_DNF():
	DNF_values = [(Conjunction(N, K), POSITIVE) for _ in range(D-1)] 
	DNF_values.append((Conjunction(N, K, is_last=True), NEGATIVE))
	DNF = DecisionList(N, K, D, init_values=DNF_values)
	return DNF


def run():
	result_path = os.path.join('results', strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
	os.makedirs(result_path)
	#decision_list = DecisionList(N, K, D)
	decision_list = create_DNF()
	#decision_list = CNF(N, 3, D, exact_size=True)
	print(decision_list)
	pickle.dump(decision_list, open(os.path.join(result_path, DECISIONLIST_FILE_NAME), 'wb'))
	all_CNF3_train_accuracy = []
	all_CNF3_test_accuracy = []	
	all_DNN_train_accuracy = []
	all_DNN_test_accuracy = []
	'''
	all_classic_train_accuracy = []
	all_classic_test_accuracy = []
	all_randomforest_train_accuracy = []
	all_randomforest_test_accuracy = []
	all_gradientboosting_train_accuracy = []
	all_gradientboosting_test_accuracy = []
	all_DLNN_train_accuracy = []
	all_DLNN_test_accuracy = []
	'''
	
	test_set_size = generate_all_sampels(decision_list, TEST_SET_FILE_NAME, result_path)
	#generate_sampels(decision_list, TEST_SET_SIZE, TEST_SET_FILE_NAME, result_path, P_POSITIVE)
	for train_set_size in ALL_TRAIN_SET_SIZE:
		result_run_path = os.path.join(result_path, 'trainSetSize={0},p_positive={1}'.format(train_set_size, P_POSITIVE))
		os.makedirs(result_run_path)
		generate_sampels(decision_list, train_set_size, TRAIN_SET_FILE_NAME, result_run_path, P_POSITIVE)

		print("==============================================================================================")
		print("Running CNF3 algorithem....")
		CNF3_train_accuracy , CNF3_test_accuracy = CNF3.run_once(train_set_size, test_set_size, result_run_path)
		print("classic got on train set: accuracy={}".format(CNF3_train_accuracy))
		print("classic got on test set: accuracy={}".format(CNF3_test_accuracy))
		all_CNF3_train_accuracy.append(CNF3_train_accuracy)
		all_CNF3_test_accuracy.append(CNF3_test_accuracy)

		print("==============================================================================================")
		print("Running DNN...")
		DNN_train_accuracy , DNN_test_accuracy = DNN.run_once(train_set_size, test_set_size, result_run_path)
		print("DNN got on train set: accuracy={}".format(DNN_train_accuracy))
		print("DNN got on test set: accuracy={}".format(DNN_test_accuracy))
		all_DNN_train_accuracy.append(DNN_train_accuracy)
		all_DNN_test_accuracy.append(DNN_test_accuracy)

		'''
		print("==============================================================================================")
		print("Running classic algorithem....")
		classic_train_accuracy , classic_test_accuracy = classic.run_once(train_set_size, test_set_size, result_run_path)
		print("classic got on train set: accuracy={}".format(classic_train_accuracy))
		print("classic got on test set: accuracy={}".format(classic_test_accuracy))
		all_classic_train_accuracy.append(classic_train_accuracy)
		all_classic_test_accuracy.append(classic_test_accuracy)

		print("==============================================================================================")
		print("Running DLNN...")
		DLNN_train_accuracy , DLNN_test_accuracy = DLNN.run_once(train_set_size, test_set_size, result_run_path)
		print("DLNN got on train set: accuracy={}".format(DLNN_train_accuracy))
		print("DLNN got on test set: accuracy={}".format(DLNN_test_accuracy))
		all_DLNN_train_accuracy.append(DLNN_train_accuracy)
		all_DLNN_test_accuracy.append(DLNN_test_accuracy)

		print("==============================================================================================")
		print("Running randomforest algorithem....")
		randomforest_train_accuracy , randomforest_test_accuracy = randomforest.run_once(train_set_size, test_set_size, result_run_path)
		print("randomforest got on train set: accuracy={}".format(randomforest_train_accuracy))
		print("randomforest got on test set: accuracy={}".format(randomforest_test_accuracy))
		all_randomforest_train_accuracy.append(randomforest_train_accuracy)
		all_randomforest_test_accuracy.append(randomforest_test_accuracy)
		
		print("==============================================================================================")
		print("Running gradientboosting algorithem....")
		gradientboosting_train_accuracy , gradientboosting_test_accuracy = gradientboosting.run_once(train_set_size, test_set_size, result_run_path)
		print("gradientboosting got on train set: accuracy={}".format(gradientboosting_train_accuracy))
		print("gradientboosting got on test set: accuracy={}".format(gradientboosting_test_accuracy))
		all_gradientboosting_train_accuracy.append(gradientboosting_train_accuracy)
		all_gradientboosting_test_accuracy.append(gradientboosting_test_accuracy)
		'''

	plt.title('train accuracy graph')
	plt.plot(ALL_TRAIN_SET_SIZE, all_CNF3_train_accuracy, 'mo')
	plt.plot(ALL_TRAIN_SET_SIZE, all_DNN_train_accuracy, 'ko')
	'''
	plt.plot(ALL_TRAIN_SET_SIZE, all_classic_train_accuracy, 'bo')
	plt.plot(ALL_TRAIN_SET_SIZE, all_DLNN_train_accuracy, 'ro')
	plt.plot(ALL_TRAIN_SET_SIZE, all_randomforest_train_accuracy, 'go')
	plt.plot(ALL_TRAIN_SET_SIZE, all_gradientboosting_train_accuracy, 'yo')
	'''
	plt.savefig(os.path.join(result_path, 'train_graph.png'))
	plt.clf()

	plt.title('test accuracy graph')
	plt.plot(ALL_TRAIN_SET_SIZE, all_CNF3_test_accuracy, 'mo')
	plt.plot(ALL_TRAIN_SET_SIZE, all_DNN_test_accuracy, 'ko')
	'''
	plt.plot(ALL_TRAIN_SET_SIZE, all_classic_test_accuracy, 'bo')
	plt.plot(ALL_TRAIN_SET_SIZE, all_randomforest_test_accuracy, 'go')
	plt.plot(ALL_TRAIN_SET_SIZE, all_gradientboosting_test_accuracy, 'yo')
	plt.plot(ALL_TRAIN_SET_SIZE, all_DLNN_test_accuracy, 'ro')
	'''
	plt.savefig(os.path.join(result_path, 'test_graph.png'))
	plt.clf()
	
	'''
	pickle.dump(all_classic_train_accuracy, open(os.path.join(result_path, 'classic_test_result.pkl'), 'wb'))
	pickle.dump(all_DNN_test_accuracy, open(os.path.join(result_path, 'DNN_test_result.pkl'), 'wb'))
	pickle.dump(all_randomforest_train_accuracy, open(os.path.join(result_path, 'randomforest_test_result.pkl'), 'wb'))
	pickle.dump(all_gradientboosting_train_accuracy, open(os.path.join(result_path, 'gradientboosting_test_result.pkl'), 'wb'))
	pickle.dump(all_DLNN_test_accuracy, open(os.path.join(result_path, 'DLNN_test_result.pkl'), 'wb'))

	'''
run()