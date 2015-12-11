if __name__ == '__main__':
	import numpy as np

	from layers import FCLayer, ReshapeLayer
	from structure import AbstractLayer, Netlist

	print '\nSmall NN demo. various outputs are:'

	in_ = np.ones ([1, 4])

	fc_1 = AbstractLayer (FCLayer, [1, 4], [1, 8])
	fc_2 = AbstractLayer (FCLayer, [1, 8], [1, 2])

	net = Netlist ([fc_1, fc_2])

	for a in range (5):
		graph, vm = net.build_graph ()
		print graph.evaluate (in_)

	print '\n'