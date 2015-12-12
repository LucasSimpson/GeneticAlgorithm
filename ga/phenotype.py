# base wrapper around netlist and genotype pair for genetic algorith related operations
class PhenotypeBase (object):
	def __init__ (self, graph, genotype):
		self.graph = graph
		self.genotype = genotype
		self.fitness = None

	# called to evaluate the phenotype. score is saved for quick lookup
	def evaluate (self):
		if not self.fitness:
			self.fitness = self.eval ()
		return self.fitness

	# evaluat the phenotype. implemented by child classes
	def eval (self):
		raise NotImplemented ('eval () must be implemented by child classes of ' + str (self.__class__))