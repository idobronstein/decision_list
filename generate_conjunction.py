from consts import *
from consts import *
from decision_list import *

import numpy as np
import random
import pickle


conjunction_cell = Conjunction(N, K)
last_cell = Conjunction(N, K, is_last=True)
init_values = [(conjunction_cell, POSITIVE), (last_cell, NEGATIVE)]
decision_list = DecisionList(N, 4, len(init_values), init_values=init_values)
print(decision_list)

samples = []
count_positive = 0
count_negative = 0
for _ in range(TRAIN_SET + TEST_SET):
	x = np.array([random.choice([POSITIVE, NEGATIVE]) for _ in range(N)])
	y = decision_list.get_value(x)
	if y == NEGATIVE:
		count_negative += 1
	else:
		count_positive += 1
	samples.append((x, y))

random.shuffle(samples)
print("number of positive samples: {}".format(count_positive))
print("number of negative samples: {}".format(count_negative))

pickle.dump(decision_list, open(DECISIONLIST_FILE_NAME, 'wb'))
pickle.dump(samples[ : TRAIN_SET], open(TRAIN_SET_FILE_NAME, 'wb'))
pickle.dump(samples[TRAIN_SET : TEST_SET + TRAIN_SET], open(TEST_SET_FILE_NAME, 'wb'))