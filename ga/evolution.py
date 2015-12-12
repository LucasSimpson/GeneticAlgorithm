import random
import numpy as np

from genotype import Genotype

class GeneticAlgorithm (object):
	def __init__ (self, crossover_rate, mutation_rate, pop_size, netlist, phenotype_class):
		# population size must be divisible by 2 for breeding
		assert (pop_size % 2 == 0)
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		self.pop_size = pop_size
		self.netlist = netlist
		self.phenotype_class = phenotype_class
		self.generation_num = 0
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
		self.generation_num += 1

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

			# crossover genes with probability self.crossover_rate
			u = random.random ()
			if u < self.crossover_rate:
				new_pair [0].crossover (new_pair [1])

			# add new_pair to new_genotypes
			new_genotypes += new_pair

		# new phenotypes to return
		new_phenotypes = []

		# iterate over genotypes
		for genotype in new_genotypes:
			# mutate genotype
			genotype.mutate (self.mutation_rate)

			# build graph with new genotype
			graph, vm = self.netlist.build_graph (genotype.vm)

			# create new phenotype from new genotype and graph
			new_phenotypes += [self.phenotype_class (graph, genotype)]

		# return new phenotypes
		return new_phenotypes

	# show some statistics about current generation
	def stats (self):
		scores = np.array ([p.evaluate () for p in self.phenotypes])
		print 'Generation #', self.generation_num
		print 'Highest:  ', scores [np.argmax (scores)]
		print 'Mean:     ', np.mean (scores)
		print 'variance: ', np.var (scores)
		print ''

