import numpy as np

# handles storing of all variables in 1 spot. Can be encoded to be used as a genome
class VariableManager (object):
	def __init__ (self):
		# initialize variables to empty
		self.variables = np.ones ([0])

		# VM also stores a copy of all indices, for rebuilding
		self.indices = []

	# request new variables. returns indices which can be used to get actual values
	def request_vars (self, amount):
		# calc indices
		index_0 = self.variables.size
		index_1 = index_0 + amount
		self.indices += [(index_0, index_1)]

		# create vars
		new_vars = np.ones ([amount])
		self.variables = np.concatenate ((self.variables, new_vars), axis=0)

		# return indices for retrieval
		return self.indices [-1]

	# returns values inbetween indices
	def get_vars (self, indices):
		return self.variables [indices [0]: indices [1]]

	# iter over indices
	def __iter__ (self):
		self.iter_index = -1
		return self

	# iter next function
	def next (self):
		self.iter_index += 1

		if self.iter_index == len (self.indices):
			raise StopIteration ()

		return self.indices [self.iter_index]

	# wrapper function for better readability
	def pop_indices (self):
		return self.next ()


# wrapper class around variable manager to ease access of values
class Variable (object):
	def __init__ (self, indices, shape, vm):
		self.indices = indices
		self.shape = shape
		self.vm = vm

	def get (self):
		return self.vm.get_vars (self.indices).reshape (self.shape)
