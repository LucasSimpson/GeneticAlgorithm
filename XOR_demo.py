import numpy as np

from nn.layers import FCLayer, ReshapeLayer, SoftmaxLayer, ArgmaxLayer
from nn.structure import AbstractLayer, Netlist

from ga.phenotype import PhenotypeBase
from ga.evolution import GeneticAlgorithm


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



rs_1 = AbstractLayer (ReshapeLayer, [2], [1, 2])
fc_1 = AbstractLayer (FCLayer, [1, 2], [1, 4])
fc_2 = AbstractLayer (FCLayer, [1, 4], [1, 2])
rs_2 = AbstractLayer (ReshapeLayer, [1, 2], [2])
sftm = AbstractLayer (SoftmaxLayer, [2], [2])
argmax = AbstractLayer (ArgmaxLayer, [2], [1])


net = Netlist ([rs_1, fc_1, fc_2, rs_2, sftm, argmax])
g, v = net.build_graph ()


ga = GeneticAlgorithm (0.7, 0.01, 50, net, PhenotypeXOR)
ga.stats ()

for a in range (100):
	ga.generation ()

ga.stats ()

