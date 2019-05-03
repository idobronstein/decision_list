from consts import *

import numpy as np
import random 

class Conjunction():

	def __init__(self, n, k, is_last=False, init_values=None, exact_size=False):
		self.n = n 
		self.k = k
		self.is_last = is_last
		
		if not self.is_last:
			# init random conjuction
			if not init_values:
				self.c = np.zeros(self.n, dtype=np.float32)
				n_index_list = list(range(self.n))
				if exact_size:
					conjunction_size = self.k
				else:
					conjunction_size = random.choice(list(range(1, self.k)))	
				chosen_indexes = random.sample(n_index_list, conjunction_size)
				for i in range(self.n):
					self.c[i] = random.choice([POSITIVE, NEGATIVE]) if i in chosen_indexes else EMPTY
			# init conjuction from init values
			else:
				assert len(init_values) == self.n, "The init values nust be {} long".format(self.n)
				for value in init_values:
					assert value in [POSITIVE, NEGATIVE, EMPTY], "The init values contains illegal value: {}".format(value)
				self.c = np.array(init_values, dtype=np.float32)

		
	def get_value(self, assignment):
		if self.is_last:
			return POSITIVE
		
		assert assignment.shape == self.c.shape, "The assignment must be {n} values long".format(n=self.n)
		variables_values = assignment * self.c
		return NEGATIVE if NEGATIVE in variables_values else POSITIVE

	def __str__(self):
		if self.is_last:
			return " True "
		string = ""
		for i in range(self.n):
			if self.c[i] == POSITIVE:
				string += " X_{} and ".format(i)
			elif self.c[i] == NEGATIVE:
				string += " !X_{} and ".format(i)
		if not string == "":
			string = string[:-5]
		return string

class Disjunction():

	def __init__(self, n, k, init_values=None, exact_size=False):
		self.n = n 
		self.k = k
		
		# init random conjuction
		if not init_values:
			self.c = np.zeros(self.n, dtype=np.float32)
			n_index_list = list(range(self.n))
			if exact_size:
				conjunction_size = self.k
			else:
				conjunction_size = random.choice(list(range(1, self.k)))	
			chosen_indexes = random.sample(n_index_list, conjunction_size)
			for i in range(self.n):
				self.c[i] = random.choice([POSITIVE, NEGATIVE]) if i in chosen_indexes else EMPTY
		# init conjuction from init values
		else:
			assert len(init_values) == self.n, "The init values nust be {} long".format(self.n)
			for value in init_values:
				assert value in [POSITIVE, NEGATIVE, EMPTY], "The init values contains illegal value: {}".format(value)
			self.c = np.array(init_values, dtype=np.float32)

		
	def get_value(self, assignment):
		assert assignment.shape == self.c.shape, "The assignment must be {n} values long".format(n=self.n)
		variables_values = assignment * self.c
		return POSITIVE if POSITIVE in variables_values else NEGATIVE

	def __str__(self):
		string = ""
		for i in range(self.n):
			if self.c[i] == POSITIVE:
				string += " X_{} or ".format(i)
			elif self.c[i] == NEGATIVE:
				string += " !X_{} or ".format(i)
		if not string == "":
			string = string[:-4]
		return string


class DecisionList():
	
	def __init__(self, n, k, d, init_values=None):
		self.n = n 
		self.k = k
		self.d = d

		# init random decision list 
		if not init_values:
			self.l = []
			for i in range(self.d - 1):
				c = Conjunction(self.n, self.k)
				v = random.choice([POSITIVE, NEGATIVE])
				self.l.append((c, v))
			# add last "True" conjunction
			c_last = Conjunction(self.n, self.k, True)
			v = random.choice([POSITIVE, NEGATIVE])
			self.l.append((c_last, v))

		# init decision list from init values
		else:
			assert len(init_values) == self.d,  "The init values nust be {} long".format(self.d)
			for c, v in init_values:
				assert type(c) == Conjunction and v in [POSITIVE, NEGATIVE], "Init values must contains pair of (Conjunction, Value)"
				assert c.k == self.k, "The Conjunction has wrong size"
				assert c.n == self.n, "The Conjunction has wrong number pf variables"
			assert c.is_last, "The last Conjunction is not is_last"
			self.l = init_values

	def get_value(self, assignment):
		for c, v in self.l:
			if c.get_value(assignment) == POSITIVE:
				return v

	def get_matrix_representation(self):
		W = np.zeros([self.d - 1, self.n], dtype=np.float32)
		V = np.zeros([self.d], dtype=np.float32)
		for idx, conjunction in enumerate(self.l[:-1]):
			c, v = conjunction
			W[idx] = c.c
			V[idx] = v
		V[self.d - 1] = self.l[-1][1] 
		return W, V

	def __str__(self):
		string = ""
		for c, v in self.l:
			string += str(c) + "--- True ---> {v}\n   |\n   |\n  False\n   |\n   |\n   V\n".format(v=v)
		return string


class CNF():
	def __init__(self, n, k, d, init_values=None, exact_size=False):
		self.n = n 
		self.k = k
		self.d = d

		# init random decision list 
		if not init_values:
			self.l = []
			for i in range(self.d):
				c = Disjunction(self.n, self.k, exact_size=exact_size)
				self.l.append(c)

		# init decision list from init values
		else:
			assert len(init_values) == self.d,  "The init values nust be {} long".format(self.d)
			for c  in init_values:
				assert type(c) == Disjunction, "Init values must contains Disjunction"
				assert c.k == self.k, "The Conjunction has wrong size"
				assert c.n == self.n, "The Conjunction has wrong number pf variables"
			self.l = init_values

	def get_value(self, assignment):
		for disjunction in self.l:
			if disjunction.get_value(assignment) == NEGATIVE:
				return NEGATIVE
		return POSITIVE


	def __str__(self):
		string = ""
		for i in self.l:
			string += "( " + str(i) +" ) and \n"
		if not string == "":
			string += string[:-8]	
		return string