import numpy as np

from nn.layers import FCLayer, ReshapeLayer

from nn.structure import AbstractLayer, Netlist



in_ = np.ones ([1, 4])


fc_1 = AbstractLayer (FCLayer, [1, 4], [1, 8])
fc_2 = AbstractLayer (FCLayer, [1, 8], [1, 2])


net = Netlist ([fc_1, fc_2])


num_models = 10
models = []
for a in range (num_models):
	models += [net.build_graph ()]
	print models [a] [0].evaluate (in_)

# wrapper around VariableManager for genetic algorithm related operations
class Genotype (object):
	def __init__ (self, vm):
		self.vm = vm

# wrapper around netlist and genotype pair for genetic algorith related operations
class Phenotype (object):
	def __init__ (self, netlist, genotype):
		self.netlist = netlist
		self.genotype = genotype


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
