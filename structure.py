from variables import VariableManager

# object that describes a single layer of a neural network
class AbstractLayer (object):
	# takes params: 
	#	class_: Layer class to use
	#	input_shape: shape of input
	#	output_shape: shape of output
	def __init__ (self, class_, input_shape, output_shape):
		self.class_ = class_
		self.input_shape = input_shape
		self.output_shape = output_shape

# object that descrives the layout of a neural network
class Netlist (object):
	# netlist is an array of LayerAbstract objects, in order
	def __init__ (self, netlist):
		self.netlist = netlist

	# build full network from vm values, and return pointer to last layer for eval
	def build_graph (self, vm_=None):
		if vm_:
			vm = vm_
		else:
			vm = VariableManager ()

		# setup vm for iteration over indices
		iter (vm)

		# empty layers list
		layers = []

		# iter over netlist
		for l in self.netlist:
			# construct layer
			layer = l.class_ (l.input_shape, l.output_shape, vm)

			# request vars in vm if none were provided
			if not vm_:
				layer.request_vars ()

			# init variables for layer
			layer.init_variables ()

			# add layer to list of layers
			layers += [layer]

		# connect layers
		for i in range (len (layers) - 1):
			layers [i].connect (layers [i + 1])

		# return first layer for eval
		return layers [0]