import numpy as np

from nn.layers import FCLayer, ReshapeLayer, SoftmaxLayer, ArgmaxLayer, LogisticLayer
from nn.structure import AbstractLayer, Netlist

from ga.phenotype import PhenotypeBase
from ga.evolution import GeneticAlgorithm


# custom phenotype for various problems
class PhenotypeXOR (PhenotypeBase):
	def eval (self):
		scoring = [16, 8, 4, 2, 1]
		cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
		answers = [0, 1, 1, 0]
		mistakes = 0
		for i in range (len (cases)):
			in_ = np.array (cases [i])
			out = answers [i] - self.graph.evaluate (in_) [0]
			mistakes += np.absolute (out)

		return scoring [mistakes]


rs_1 = AbstractLayer (ReshapeLayer, [2], [1, 2])
fc_1 = AbstractLayer (FCLayer, [1, 2], [1, 4])
l_1 = AbstractLayer (LogisticLayer, [1, 4], [1, 4])
fc_2 = AbstractLayer (FCLayer, [1, 4], [1, 2])
rs_2 = AbstractLayer (ReshapeLayer, [1, 2], [2])
sftm = AbstractLayer (SoftmaxLayer, [2], [2])
argmax = AbstractLayer (ArgmaxLayer, [2], [1])


net = Netlist ([rs_1, fc_1, l_1, fc_2, rs_2, sftm, argmax])
g, v = net.build_graph ()


ga = GeneticAlgorithm (0.7, 0.05, 200, net, PhenotypeXOR)

a = 0
while (ga.get_best ().evaluate () != 16):
	if a % 20 == 0:
		ga.stats ()
	ga.generation ()
	a += 1

ga.stats ()

g = ga.get_best ().graph

print '[0, 0] -> ', g.evaluate (np.array ([0, 0]))
print '[0, 1] -> ', g.evaluate (np.array ([0, 1]))
print '[1, 0] -> ', g.evaluate (np.array ([1, 0]))
print '[1, 1] -> ', g.evaluate (np.array ([1, 1]))
