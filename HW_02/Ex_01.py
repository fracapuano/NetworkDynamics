import numpy as np
from markov_utils import *
from tqdm import tqdm
from scipy import linalg
from itertools import product

n_simulations = int(2)

# activation rate matrix
L = np.array([
    [0, 2/5, 1/5, 0, 0], 
    [0, 0, 3/4, 1/4, 0], 
    [1/2, 0, 0, 1/2, 0], 
    [0, 0, 1/3, 0, 2/3], 
    [0, 1/3, 0, 1/3, 0]
])

# nodes
nodes = ["o", "a", "b", "c", "d"]
# node-index dictionary
node_index = {
    node: idx for idx, node in enumerate(nodes)
}

# node degrees
omegas = L @ np.ones(L.shape[0])
# degree matrix
D = np.diag(omegas)
# jump probability matrix (row stochastic)
P = np.linalg.inv(D) @ L

# anything from starting
starting_node = "o"
# retrieving 'starting_node' rate from omegas
initial_rate = omegas[
    node_index[starting_node]
]

# maximal number of transitions
max_transitions = int(1e4)
# nodes visited, in order, during the random walk
visits = np.zeros(max_transitions)
# time spent in each node during random walk
visits_time = np.zeros_like(visits)

# creating node objects
node_objects = [Node(name=node_name, degree=degree) for node_name, degree in zip(nodes, omegas)]

# nodes
node_o, node_a, node_b, node_c, node_d = node_objects

# point (a)
print("Point (a): (empirical) Expected Return Time from node 'a'")

from_a = RandomWalk(start_node=node_a, nodes=node_objects, P=P)

durations = np.zeros(n_simulations)

for simulation in tqdm(range(n_simulations)): 
    _, duration = from_a.walk_until(target_node=node_a)
    durations[simulation] = duration

print("Average (empirical) Return Time from node '{}' : {:.4f} t.u.".format(
    from_a.input_start.name, durations.mean()
))
print("*"*50)

# point (b)
print("Point (b): (theoretical) Expected Return Time from node 'a'")

# average return time can also be obtained from a theorical standpoint
Lapl = D - L
# stationary probability is defined as kernel of Laplacian matrix (transposed)
Lapl_kernel = linalg.null_space(Lapl.T)
pi_bar =  (Lapl_kernel/ Lapl_kernel.sum()).reshape(-1,)

# return time according to theoretical results
return_times = 1/(omegas * pi_bar)

return_times = {node: return_times[idx] for idx, node in enumerate(nodes)}
print("Expected Return Time from node {}: {:.4f} t.u.".format(
    from_a.input_start.name, return_times[from_a.input_start.name]
))
print("*"*50)

print("Point (c): (empirical) Hitting Time from node 'o' to node 'd'")

from_o = RandomWalk(start_node=node_o, nodes=node_objects, P=P)

durations = np.zeros(n_simulations)

for simulation in tqdm(range(n_simulations)): 
    _, duration = from_o.walk_until(target_node=node_d, remove_last=True)
    durations[simulation] = duration

print("(empirical) Average Hitting Time" + f" from {from_o.input_start.name} to {node_d.name}" + ": {:.4f} t.u.".format(durations.mean()))
print("*"*50)

destination_node = "d"
target_node = node_objects[node_index[destination_node]]
print(f"Point (d): (theoretical) Expected Hitting Time from node '{from_o.input_start.name}' to node '{target_node.name}'")

# P_hat coincides with P when row and column related to node "o" are removed
P_hat = remove_i(arr=P, idx=node_index[destination_node])
omegas_hat = np.delete(arr=omegas, obj=node_index[destination_node])

A, b = np.eye(len(omegas_hat)) - P_hat, 1/omegas_hat

hitting_time = np.linalg.solve(A, b)

print(
    f"Expected Hitting Time from node '{from_o.input_start.name}' to node '{target_node.name}':" 
    + " {:.4f} t.u.".format(hitting_time[node_index[from_o.input_start.name]])
)