import random
import numpy as np

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
