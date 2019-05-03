from consts import *
from decision_list import *

import pickle 
import datetime
import os
import random

def calc_combinations(current, i, all_possible):
	if i + 1 > N:
		if POSITIVE in current or NEGATIVE in current:
			if sum([abs(j) for j in current]) == 3:
				all_possible.append(current)
	else:
		calc_combinations(current[:], i + 1, all_possible)
		if sum([abs(j) for j in current]) <= 3:
			current[i] = POSITIVE
			calc_combinations(current[:], i + 1, all_possible)
			current[i] = NEGATIVE
			calc_combinations(current[:], i + 1, all_possible)


def get_all_possible_disjunctions():
	current_conjunction = [EMPTY] * N
	all_possible_disjunctions = []
	print("Start calc all possible combinations")
	start_time = datetime.datetime.now()
	calc_combinations(current_conjunction, 0, all_possible_disjunctions)
	end_time = datetime.datetime.now()
	print("There is {0} combinations. It's took {1} to calc them".format(len(all_possible_disjunctions), end_time - start_time))
	return all_possible_disjunctions

def run_once(train_set_size, test_set_size, result_path):
	# create working dir
	run_title = 'CNF3_N={0}_K={1}_D={2}'.format(N, K, D, train_set_size, test_set_size)
	dir_path = os.path.join(result_path, run_title)
	os.makedirs(dir_path)

	# Load all data
	train_set = pickle.load(open(os.path.join(result_path, TRAIN_SET_FILE_NAME), 'rb'))
	test_set = pickle.load(open(os.path.join(os.path.join(result_path, '..'), TEST_SET_FILE_NAME), 'rb'))

	# create all "new literals"
	all_possible_disjunctions = get_all_possible_disjunctions()
	random.shuffle(all_possible_disjunctions)

	# Find 3-CNF from train set
	indexs_to_remove = []
	for x, y in train_set:
		if y == POSITIVE:
			for i in range(len(all_possible_disjunctions)):
				disjunction = Disjunction(N, 3, init_values=all_possible_disjunctions[i])
				if disjunction.get_value(x) == NEGATIVE:
					if i not in indexs_to_remove:
						indexs_to_remove.append(i)
	CNF_disjunctions_values = [all_possible_disjunctions[j] for j in range(len(all_possible_disjunctions)) if j not in indexs_to_remove]
	print("The CNF3 contains {0} conjunctions".format(len(CNF_disjunctions_values)))

	# Insert result to structed 3-CNF
	CNF_disjunctions = [Disjunction(N, 3, init_values=dis) for dis in CNF_disjunctions_values]
	CNF3 = CNF(N, 3, len(CNF_disjunctions_values), init_values=CNF_disjunctions)

	# save result
	pickle.dump(CNF3, open(os.path.join(dir_path, DECISIONLIST_FILE_NAME), 'wb'))

	# Check train set
	successes_num = 0
	for sample in train_set:
		x, y = sample
		if CNF3.get_value(x) == y:
			successes_num += 1
		else:
			print(y)
	train_accuracy = float(successes_num) / train_set_size
	print("Train accuracy of {}".format(train_accuracy))

	# Test this decision list 
	successes_num = 0
	for sample in test_set:
		x, y = sample
		if CNF3.get_value(x) == y:
			successes_num += 1
	test_accuracy = float(successes_num) / test_set_size
	print("Test accuracy of {}".format(test_accuracy))

	return train_accuracy, test_accuracy