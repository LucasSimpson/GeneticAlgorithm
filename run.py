import numpy as np

from layers import FCLayer, ReshapeLayer

from structure import AbstractLayer, Netlist



in_ = np.ones ([1, 4])


fc_1 = AbstractLayer (FCLayer, [1, 4], [1, 8])
rs_1 = AbstractLayer (ReshapeLayer, [1, 8], [4, 2])
rs_2 = AbstractLayer (ReshapeLayer, [4, 2], [1, 8])
fc_2 = AbstractLayer (FCLayer, [1, 8], [1, 2])


net = Netlist ([fc_1, rs_1, rs_2, fc_2])



out_1, vm = net.build_graph ()

out_2, vm = net.build_graph (vm)

print out_1.evaluate (in_)
print out_2.evaluate (in_)
