from time import gmtime, strftime

# General
POSITIVE = 1 
NEGATIVE = -1 
EMPTY = 0

# Decision list params 
N = 10
K = 7
D = 4
DECISIONLIST_FILE_NAME = 'decision_list.pkl'

# Sampels params
ALL_TRAIN_SET_SIZE = [20, 100, 200, 500 ,1000, 2000]
TEST_SET_SIZE = 10000
TRAIN_SET_FILE_NAME = 'train_set.pkl'
TEST_SET_FILE_NAME = 'test_set.pkl'
P_POSITIVE = 0.5

# General training params
MAX_STEP = 200000
DISPLAY = 5000
TEST = MAX_STEP - 1

# DNN training params
BATCH_SIZE_DNN = 1
INITIAL_LEARNING_RATE_DNN = 0.001
LAYER_SIZE = 16

# DLNN training params
BATCH_SIZE_DLNN = 1
INITIAL_LEARNING_RATE_DLNN = 0.1
BETA = 40
EPSILON = 0.000001

# Script params
W_FILE_NAME = 'W.pkl'
V_FILE_NAME = 'V.pkl'

# CNF3
CNF_SIZE = 10