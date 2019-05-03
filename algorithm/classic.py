from consts import *
from decision_list import DecisionList, Conjunction

import pickle 
import datetime
import os
import random

def calc_combinations(current, i, all_possible):
	if i + 1 > N:
		if POSITIVE in current or NEGATIVE in current:
			all_possible.append(current)
	else:
		calc_combinations(current[:], i + 1, all_possible)
		if sum([abs(j) for j in current]) < K:
			current[i] = POSITIVE
			calc_combinations(current[:], i + 1, all_possible)
			current[i] = NEGATIVE
			calc_combinations(current[:], i + 1, all_possible)

def get_all_possible_combinations():
	current_conjunction = [0] * N
	all_possible_combinations = []
	print("Start calc all possible combinations")
	start_time = datetime.datetime.now()
	calc_combinations(current_conjunction, 0, all_possible_combinations)
	end_time = datetime.datetime.now()
	print("There is {0} combinations. It's took {1} to calc them".format(len(all_possible_combinations), end_time - start_time))
	return all_possible_combinations

def generate_conjunction(all_possible_combinations):
	for combination in all_possible_combinations:
		yield Conjunction(N, K, init_values=combination)

def run_once(train_set_size, test_set_size, result_path):
	# create working dir
	run_title = 'classic_N={0}_K={1}_D={2}'.format(N, K, D, train_set_size, test_set_size)
	dir_path = os.path.join(result_path, run_title)
	os.makedirs(dir_path)

	# Load all data
	train_set = pickle.load(open(os.path.join(result_path, TRAIN_SET_FILE_NAME), 'rb'))
	test_set = pickle.load(open(os.path.join(os.path.join(result_path, '..'), TEST_SET_FILE_NAME), 'rb'))
	
	# Find decision list from train set
	decision_list = []
	all_possible_combinations = get_all_possible_combinations()
	random.shuffle(all_possible_combinations)
	while len(train_set) > 0:
		find_conjunction = False
		for conjunction in generate_conjunction(all_possible_combinations):
			coordinated_examples = []
			value = EMPTY
			finish_all_sampels = True
			for i in range(len(train_set)):
				c, v = train_set[i]
				if conjunction.get_value(c) == POSITIVE:
					if value == EMPTY:
						value = v
						coordinated_examples.append(i)
					elif value == v:
						coordinated_examples.append(i)
					elif value != v:
						finish_all_sampels = False
						break
			if finish_all_sampels and value != EMPTY:
				print("The conjunction {0} is fit to {1} sampels".format(conjunction, len(coordinated_examples)))
				find_conjunction = True
				decision_list.append((conjunction, value))
				train_set = [x for i, x in enumerate(train_set) if i not in coordinated_examples]
				break
		if not find_conjunction:
			print("The given samples is not consistent with ant k-DL function")
			exit()
	decision_list.append((Conjunction(N, K, True), POSITIVE))
	decision_list = DecisionList(N, K, len(decision_list), init_values=decision_list)
	pickle.dump(decision_list, open(os.path.join(dir_path, DECISIONLIST_FILE_NAME), 'wb'))
	
	# Check train set
	successes_num = 0
	train_set = pickle.load(open(os.path.join(result_path, TRAIN_SET_FILE_NAME), 'rb'))
	for sample in train_set:
		c, v = sample
		if decision_list.get_value(c) == v:
			successes_num += 1
	train_accuracy = float(successes_num) / train_set_size
	print("Train accuracy of {}".format(train_accuracy))

	# Test this decision list 
	successes_num = 0
	for sample in test_set:
		c, v = sample
		if decision_list.get_value(c) == v:
			successes_num += 1
	test_accuracy = float(successes_num) / test_set_size
	print("Test accuracy of {}".format(test_accuracy))

	return train_accuracy, test_accuracy

