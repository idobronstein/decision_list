from consts import *
from sklearn.ensemble import GradientBoostingClassifier

import os
import pickle

split_data = lambda data, index:  [i[index] for i in data]

def run_once(train_set_size, test_set_size, result_path):
	# create working dir
	run_title = 'GradientBoostingClassifier_N={0}_K={1}_D={2}'.format(N, K, D, train_set_size, test_set_size)
	dir_path = os.path.join(result_path, run_title)
	os.makedirs(dir_path)

	# Load all data
	train_set = pickle.load(open(os.path.join(result_path, TRAIN_SET_FILE_NAME), 'rb'))
	test_set = pickle.load(open(os.path.join(os.path.join(result_path, '..'), TEST_SET_FILE_NAME), 'rb'))

	# Split Data
	train_set_x = split_data(train_set, 0)
	train_set_y = split_data(train_set, 1)
	test_set_x = split_data(test_set, 0)
	test_set_y = split_data(test_set, 1)
	
	# Run random forest
	clf = GradientBoostingClassifier(n_estimators=100, max_depth=D * 2, random_state=0)
	clf.fit(train_set_x, train_set_y)

	# Test on Train 
	train_result = clf.predict(train_set_x)
	successes_num = 0
	for i in range(train_set_size):
		if train_result[i] == train_set_y[i]:
			successes_num += 1
	train_accuracy = float(successes_num) / train_set_size

	test_result = clf.predict(test_set_x)
	successes_num = 0
	for i in range(test_set_size):
		if test_result[i] == test_set_y[i]:
			successes_num += 1
	test_accuracy = float(successes_num) / test_set_size

	return train_accuracy, test_accuracy