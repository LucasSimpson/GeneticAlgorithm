import random

import numpy as np

from nn.layers import FCLayer, ReshapeLayer, SoftmaxLayer, ArgmaxLayer

from nn.structure import AbstractLayer, Netlist



# wrapper around VariableManager for genetic algorithm related operations
class Genotype (object):
	def __init__ (self, vm):
		self.vm = vm

	# modifies self and other such that their genotypes are a random crossover of eachother
	def crossover (self, other):
		# randomly select pivot index p
		p = random.randint (0, len (self.vm) - 1)

		# reassign for shorter line length following
		a = self.vm.variables
		b = other.vm.variables

		# make two new arrays, each from part a and part b, swapped around pivot point p
		c, d = np.concatenate ((a [:p], b [p:]), axis=0),  np.concatenate ((b [:p], a [p:]), axis=0)

		# reassign the variables
		self.vm.variables = c
		other.vm.variables = d

	# mutate each var with probability U
	def mutate (self, U):
		for v in self.vm.variables:
			if random.random () <= U:
				v = random.normalvariate (0, 1)

	# creates a deep copy
	def deep_copy (self):
		return Genotype (self.vm.deep_copy ())

	# returns length of genotype
	def __len__ (self):
		return len (self.vm)

# base wrapper around netlist and genotype pair for genetic algorith related operations
class PhenotypeBase (object):
	def __init__ (self, graph, genotype):
		self.graph = graph
		self.genotype = genotype
		self.fitness = -1

	# called to evaluate the phenotype. score is saved for quick lookup
	def evaluate (self):
		self.fitness = self.eval ()
		return self.fitness

	# evaluat the phenotype. implemented by child classes
	def eval (self):
		raise NotImplemented ('eval () must be implemented by child classes of ' + str (self.__class__))


# custom phenotype for various problems
class PhenotypeXOR (PhenotypeBase):
	def eval (self):
		scoring = [16, 8, 4, 2, 1]
		cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
		answers = [0, 1, 1, 0]
		score = 1
		for i in range (len (cases)):
			mistakes = 0
			in_ = np.array (cases [i])
			out = answers [i] - self.graph.evaluate (np.array (in_))
			mistakes += out * out
		return scoring [mistakes [0]]



class GeneticAlgorithm (object):
	def __init__ (self, crossover_rate, mutation_rate, pop_size, netlist, phenotype_class):
		# population size must be divisible by 2 for breeding
		assert (pop_size % 2 == 0)
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		self.pop_size = pop_size
		self.netlist = netlist
		self.phenotype_class = phenotype_class
		self.phenotypes = []

		for i in range (pop_size):
			graph, vm = self.netlist.build_graph ()
			self.phenotypes += [self.phenotype_class (graph, Genotype (vm))]

	def generation (self):
		for phenotype in self.phenotypes:
			phenotype.evaluate ()

		pairs = self.roulette_selection ()
		new_phenotypes = self.breed (pairs)
		self.phenotypes = new_phenotypes

	# returns an array of pairs of phenotypes, designating which pair to breed as selected by roulette_selection
	def roulette_selection (self):
		# helper function for biased selection
		def weighted_pick (reg_):
			u = random.random () * total

			# naive increment-until-there method
			# TODO optimize with binary search
			for b in range (len (reg_) - 1):
				if u >= reg_ [b] and u < reg_ [b + 1]:
					return b

		# calc cumulative fitness, and relative regions for weighted selection probability
		total = 0
		regions = [0]
		for phenotype in self.phenotypes:
			total += phenotype.fitness
			regions += [total]
		
		# build list of pairs
		pairs = []
		for i in range (self.pop_size / 2):
			# generate two random nums
			id_a = weighted_pick (regions)
			id_b = weighted_pick (regions)

			# add to list of pairs
			pairs += [(self.phenotypes [id_a], self.phenotypes [id_b])]

		# return pairs
		return pairs

	# takes in list of pairs of phenotypes, and breeds them together
	def breed (self, pairs):
		# empty list of new genotypes
		new_genotypes = []

		# iter over all pairs
		for pair in pairs:
			# create copy of each genotype
			new_pair = [p.genotype.deep_copy () for p in pair]

			u = random.random ()
			if u < self.crossover_rate:
				new_pair [0].crossover (new_pair [1])

			new_genotypes += new_pair


		new_phenotypes = []

		for genotype in new_genotypes:
			genotype.mutate (self.mutation_rate)

			graph, vm = self.netlist.build_graph (genotype.vm)
			new_phenotypes += [self.phenotype_class (graph, genotype)]


		return new_phenotypes

	def evaluate (self):
		print 'Current pop scores:'
		for phenotype in self.phenotypes:
			print phenotype.evaluate ()


rs_1 = AbstractLayer (ReshapeLayer, [2], [1, 2])
fc_1 = AbstractLayer (FCLayer, [1, 2], [1, 4])
fc_2 = AbstractLayer (FCLayer, [1, 4], [1, 2])
rs_2 = AbstractLayer (ReshapeLayer, [1, 2], [2])
sftm = AbstractLayer (SoftmaxLayer, [2], [2])
argmax = AbstractLayer (ArgmaxLayer, [2], [1])


net = Netlist ([rs_1, fc_1, fc_2, rs_2, sftm, argmax])
g, v = net.build_graph ()



ga = GeneticAlgorithm (0.7, 0.01, 50, net, PhenotypeXOR)
ga.evaluate ()

for a in range (100):
	ga.generation ()
	ga.evaluate ()

