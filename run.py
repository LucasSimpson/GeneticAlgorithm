import numpy as np

from nn.layers import FCLayer, ReshapeLayer, SoftmaxLayer

from nn.structure import AbstractLayer, Netlist



in_ = np.ones ([4, 4, 17])


rs_1 = AbstractLayer (ReshapeLayer, [4, 4, 17], [1, 4 * 4 * 17])
fc_1 = AbstractLayer (FCLayer, [1, 4 * 4 * 17], [1, 256])
fc_2 = AbstractLayer (FCLayer, [1, 256], [1, 4])
rs_2 = AbstractLayer (ReshapeLayer, [1, 4], [4])
sftm = AbstractLayer (SoftmaxLayer, [4], [4])


net = Netlist ([rs_1, fc_1, fc_2, rs_2, sftm])


for a in range (10):
	graph, vm = net.build_graph ()
	print graph.evaluate (in_)

# wrapper around VariableManager for genetic algorithm related operations
class Genotype (object):
	def __init__ (self, vm):
		self.vm = vm

# wrapper around netlist and genotype pair for genetic algorith related operations
class Phenotype (object):
	def __init__ (self, netlist, genotype):
		self.netlist = netlist
		self.genotype = genotype

	@property
	def genotype (self):
		return self.genotype


class GeneticAlgorithm (object):
	def __init__ (self, crossover_rate, mutation_rate, pop_size, netlist):
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		self.pop_size = pop_size
		self.netlist = netlist
		self.phenotypes = []

		for i in pop_size:
			graph, vm = netlist.build_graph ()
			self.phenotypes += [Phenotype (netlist, Genotype (vm))]

	def generation (self):
		pass
		# evaluate all phenotypes
		# 	play 2048

		# breed phenotypes
		# 	roulette wheel, crossover, mutate

		# reassign self.phenotypes
