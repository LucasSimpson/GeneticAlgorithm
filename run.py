import numpy as np

from layers import FCLayer, ReshapeLayer

from structure import AbstractLayer, Netlist




fc_1 = AbstractLayer (FCLayer, [1, 4], [1, 8])
rs_1 = AbstractLayer (ReshapeLayer, [1, 8], [4, 2])
rs_2 = AbstractLayer (ReshapeLayer, [4, 2], [1, 8])
fc_2 = AbstractLayer (FCLayer, [1, 8], [1, 2])

net = Netlist ([fc_1, rs_1, rs_2, fc_2])

out = net.build_graph ()


in_ = np.ones ([1, 4])

print out.evaluate (in_)