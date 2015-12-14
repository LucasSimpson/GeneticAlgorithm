from Game2048 import Game

import matplotlib.pyplot as plt


import numpy as np
import math

from nn.layers import FCLayer, ReshapeLayer, SoftmaxLayer, ArgmaxLayer, LogisticLayer
from nn.structure import AbstractLayer, Netlist

from ga.phenotype import PhenotypeBase
from ga.evolution import GeneticAlgorithm


# custom phenotype for various problems
class Phenotype2048 (PhenotypeBase):
	def eval (self):
		_map = ['w', 's', 'a', 'd']

		scores = []

		for a in range (10):
			g = Game ()
			while (not g.is_stale ()):
				state_ = g.get_state ()
				state = [0 for a in range (16)]
				for i in range (len (state_)):
					if state_ [i] != 0:
						state [i] = math.log (state_ [i], 2) / 8.0

				in_ = np.array (state)
				out = self.graph.evaluate (in_) [0]
				g.process_move (_map [out])
			scores += [g.get_score ()]

		return (sum (scores) / 10.0) ** 2


rs_1 = AbstractLayer (ReshapeLayer, [16], [1, 16])
fc_1 = AbstractLayer (FCLayer, [1, 16], [1, 256])
l_1 = AbstractLayer (LogisticLayer, [1, 256], [1, 256])
fc_2 = AbstractLayer (FCLayer, [1, 256], [1, 4])
rs_2 = AbstractLayer (ReshapeLayer, [1, 4], [4])
sftm = AbstractLayer (SoftmaxLayer, [4], [4])
argmax = AbstractLayer (ArgmaxLayer, [4], [1])


net = Netlist ([rs_1, fc_1, l_1, fc_2, rs_2, sftm, argmax])
g, v = net.build_graph ()


ga = GeneticAlgorithm (0.7, 0.05, 200, net, Phenotype2048)


scores = []
for a in range (100):
	scores += [math.sqrt (ga.get_best ().fitness)]
	if a % 2 == 0:
		ga.stats ()
	ga.generation ()


ga.stats ()



plt.plot (scores)
plt.xlabel ('Generation #')
plt.ylabel ('Fitness Score')
plt.show ()
