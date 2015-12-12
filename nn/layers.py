import numpy as np

from variables import Variable

# base class for any layer in a neural network
class Layer (object):
	def __init__ (self, input_shape, output_shape, vm):
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.vm = vm
		self.next_layer = None

	# chain together multiple layers
	def connect (self, next_layer):
		# assert complience between input/output shapes
		assert (self.output_shape == next_layer.input_shape)

		# set next layer
		self.next_layer = next_layer
		return next_layer

	# initialize any variables using vm
	def init_variables (self):
		raise NotImplemented ()

	# eval the input, either return it or pass it to next layer
	def evaluate (self, input_):
		out = self.eval (input_)

		if self.next_layer:
			return self.next_layer.evaluate (out)
		else:
			return out

	# pass input_ through layer and spit out result
	def eval (self, input_):
		raise NotImplemented ()

	# get layer to request vars. Only used if no previous variables in variable manager
	# my clever trick to be able to build from scratch or from previous VM using only netlist
	def request_vars (self):
		raise NotImplemented ()


# fully connected layer
class FCLayer (Layer):
	def __init__ (self, input_shape, output_shape, vm):
		super (FCLayer, self).__init__ (input_shape, output_shape, vm)

		self.weights = None
		self.bias = None

	# initialize variables. can't happen in __init__ because request_vars might need to happen first
	def init_variables (self):
		w_shape = [self.input_shape [1], self.output_shape [1]]
		b_shape = [1, self.output_shape [1]]

		self.weights = Variable (self.vm.pop_indices (), w_shape, self.vm)
		self.bias = Variable (self.vm.pop_indices (), b_shape, self.vm)

	# input_ * weights + bias
	def eval (self, input_):
		return np.dot (input_, self.weights.get ()) + self.bias.get ()

	# [input, output], [output]
	def request_vars (self):
		self.vm.request_vars (self.input_shape [1] * self.output_shape [1])
		self.vm.request_vars (self.output_shape [1])

# Reshapes input_shape into output_shape
class ReshapeLayer (Layer):
	def __init__ (self, input_shape, output_shape, vm):
		super (ReshapeLayer, self).__init__ (input_shape, output_shape, vm)

	# no variables needed
	def init_variables (self):
		pass

	# reshape input shape to output shape
	def eval (self, input_):
		return input_.reshape (self.output_shape)

	# no variables
	def request_vars (self):
		pass

# runs inputs through the softmax function
class SoftmaxLayer (Layer):
	def __init__ (self, input_shape, output_shape, vm):
		super (SoftmaxLayer, self).__init__ (input_shape, output_shape, vm)

	# no variables needed
	def init_variables (self):
		pass

	# run input_ through softmax function
	def eval (self, input_):
		e_x = np.exp (input_)
		return e_x / np.sum (e_x)

	# no variables
	def request_vars (self):
		pass


# returns id of maximum arguement. shortcut layer because this is used a lot
class ArgmaxLayer (Layer):
	def __init__ (self, input_shape, output_shape, vm):
		super (ArgmaxLayer, self).__init__ (input_shape, output_shape, vm)

	# no variables needed
	def init_variables (self):
		pass

	# spit out index of maximum argument. Expects 1 dimensional vector as input
	def eval (self, input_):
		max_, id_ = input_ [0], 0
		for i in range (1, len (input_)):
			if input_ [i] > max_:
				id_ = i 
				max_ = input_ [i]
		return np.array ([id_])


	# no variables
	def request_vars (self):
		pass
