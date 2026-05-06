from flow_network import FlowNetwork
from contrastive_optimizer import ContrastiveOptimizer
import numpy as np

W = np.array([[0, 1, 1, 0], 
              [1, 0, 1, 1], 
              [1, 1, 0, 1], 
              [0, 1, 1, 0], 
])

Q_in = np.array([1., 0., 0., -1.])  # source and sink

net = FlowNetwork(W)
opt = ContrastiveOptimizer(net, Q_in, target_edge=(1,2), target_flow=0.3,
                           learning_rate=0.01, nudge_strength=0.01, loss_tol=1e-4)
opt.run(max_iter=100_000, log_every=1000)
opt.plot_results()