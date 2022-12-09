from HW_02.utils.markov_utils import *
from tqdm import tqdm
from scipy import linalg
from itertools import product
from statistics import variance
import matplotlib.pyplot as plt
import time
from pprint import pprint


n_simulations = int(5e3)

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
print("Question (a): (empirical) Expected Return Time from node 'a'")

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
print("Question (b): (theoretical) Expected Return Time from node 'a'")

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

print("Question (c): (empirical) Hitting Time from node 'o' to node 'd'")

from_o = RandomWalk(start_node=node_o, nodes=node_objects, P=P)

durations = np.zeros(n_simulations)

for simulation in tqdm(range(n_simulations)): 
    _, duration = from_o.walk_until(target_node=node_d, remove_last=True)
    durations[simulation] = duration

print("(empirical) Average Hitting Time" + f" from {from_o.input_start.name} to {node_d.name}" + ": {:.4f} t.u.".format(durations.mean()))
print("*"*50)

destination_node = "d"
target_node = node_objects[node_index[destination_node]]
print(f"Question (d): (theoretical) Expected Hitting Time from node '{from_o.input_start.name}' to node '{target_node.name}'")

# P_hat coincides with P when row and column related to node "o" are removed
P_hat = remove_i(arr=P, idx=node_index[destination_node])
omegas_hat = np.delete(arr=omegas, obj=node_index[destination_node])

A, b = np.eye(len(omegas_hat)) - P_hat, 1/omegas_hat

hitting_time = np.linalg.solve(A, b)

print(
    f"Expected Hitting Time from node '{from_o.input_start.name}' to node '{target_node.name}':" 
    + " {:.4f} t.u.".format(hitting_time[node_index[from_o.input_start.name]])
)
print("*"*50)

print("Question (e): French-DeGroot model using matrix Lambda as weight matrix'")

# simulating French-DeGroot in discrete time
n_opinions = 10
n_talks = int(1e2)
intalks_variance = np.zeros((n_simulations, n_talks))
# tensor storing opinion evolutions over all possible simulations (each slice corresponds to a single simulation. Each row corresponds
# to a single talk opinion).
opinions = np.zeros((n_simulations, n_talks, len(omegas)))
starts = np.zeros((n_simulations, len(omegas)))
save_talks = False

for simulation in tqdm(range(n_simulations)): 
    starting_condition = np.random.choice(np.arange(n_opinions), size=len(omegas))
    starts[simulation, :] = starting_condition
    talk_opinion = starting_condition
    
    if save_talks: 
        # talking
        for talk in range(n_talks):
            talk_opinion = P @ talk_opinion
            opinions[simulation, talk, :] = talk_opinion
            intalks_variance[simulation, talk] = variance(talk_opinion)
        final_opinion = talk_opinion
    else: 
        # shouting one to each others real fast
        final_opinion = np.linalg.matrix_power(a=P, n=n_talks) @ starting_condition

if save_talks:
    np.savetxt(fname="intalk_variance.txt", X=intalks_variance)
    np.savetxt(fname="initialcondition_variance.txt", X=starts)
    np.savetxt(fname="opinions.txt", X=opinions)

var_tolerance = 1e-9
consensus_reached = "did" if variance(final_opinion) <= var_tolerance else "did not"
print("Agents opinions: ")
for node, opinion in zip(nodes, final_opinion):
    print(
        "Node '{}' has opinion {:.4f}".format(node, opinion)
    )
print(f"Agents {consensus_reached} reach consensus!")
print("*"*50)

print("Question (f): Variance of final opinions")

# actual 'observation', in other labs indicated as "mu"
avg_opinion = 10.
# simulating with various level of variance, i.e. noise affecting agents' evaluation of mu
agents_error = np.array([1e-2, 1e-1, 1, 1e1, 1e2])
simulated_consensus_variance = np.zeros_like(agents_error)
sub_simulations = int(1e3)

start_time = time.time()
for simulation, variance_ in (enumerate(agents_error)): 
    # std_deviation to be passed to normal distribution
    agents_std = np.sqrt(variance_)
    subsimulation_variance = np.zeros(sub_simulations)
    # simulating sub_simulations different starting conditions
    for sub_simulation in range(sub_simulations):
        # iid samples with known variance
        starting_condition = np.random.normal(loc=avg_opinion, scale=agents_std, size=len(omegas))
        # talking
        final_opinion = np.linalg.matrix_power(a=P, n=n_talks) @ starting_condition
        # in each subsimulation, the final opinion (i.e. empirical consensus) variance with respect to initial opinion
        # is obtained according to square difference between mean of consensus and average opinion
        subsimulation_variance[sub_simulation] = (final_opinion.mean() - avg_opinion)**2
    # average variance between sub-simulation
    simulated_consensus_variance[simulation] = subsimulation_variance.mean()

sim_time = time.time() - start_time
print("Total simulation time: {:.4f} s\n".format(sim_time))

# final opinion variance can also be obtained from a more practical standpoint
# long-term probability distribution in discrete time
pi_hat = linalg.null_space(P.T - np.eye(len(omegas))).reshape(-1,); pi_hat = pi_hat / pi_hat.sum()
theoretical_variance = agents_error * (pi_hat**2).sum()

theoretical_simulated_pairs = zip(simulated_consensus_variance, theoretical_variance)

for idx, pair in enumerate(theoretical_simulated_pairs): 
    print(f"With average starting opinion {avg_opinion} and variance {agents_error[idx]}")
    print("\t(empirical) Consensus variance equals: {:.3e} (theoretical variance is {:.3e})".format(pair[0], pair[1]))

print("Average difference between simulations for various variance levels is {:.3e}".format(((theoretical_variance - simulated_consensus_variance)**2).mean()))
print("*"*50)

print("Question (g): Opinion dynamics with Graph modifications")

# new activation rate matrix
L_g = L.copy()

edge1, edge2 = "d,a", "d,c"
edges_to_remove = [edge1, edge2]
# removing edges
for e in edges_to_remove: 
    idx1, idx2 = node_index[e.split(",")[0]], node_index[e.split(",")[1]]
    L_g[idx1, idx2] = 0

# add self-loops for those nodes having a zero degree to be able to well define P
for row in range(L_g.shape[0]):
    L_g[row,row] = 1 if L_g[row,:].sum() <= 1e-9 else L_g[row, row]

D_g = np.diag(L_g.sum(axis=1))
P_g = np.linalg.inv(D_g) @ L_g

# simulating French-DeGroot in discrete time
n_opinions = 10
n_talks = int(1e2)

opinions = np.zeros((n_simulations, n_talks, len(omegas)))
starts = np.zeros((n_simulations, len(omegas)))
save_talks = False

for simulation in tqdm(range(n_simulations)): 
    starting_condition = np.random.choice(np.arange(n_opinions), size=len(omegas))
    starts[simulation, :] = starting_condition
    
    if save_talks: 
        talk_opinion = starting_condition
        # talking
        for talk in range(n_talks):
            talk_opinion = P_g @ talk_opinion
            opinions[simulation, talk, :] = talk_opinion
            intalks_variance[simulation, talk] = variance(talk_opinion)
        final_opinion = talk_opinion
    else: 
        # shouting one to each others real fast
        final_opinion = opinion_dynamics(matrix=P_g, initial_condition=starting_condition, n_talks=n_talks)

if save_talks:
    np.savetxt(fname="intalk_variance_(g).txt", X=intalks_variance)
    np.savetxt(fname="initialcondition_variance_(g).txt", X=starts)
    np.savetxt(fname="opinions_(g).txt", X=opinions.reshape(n_simulations * n_talks, -1))

# long-term probability distribution in discrete time
pi_hat_g = linalg.null_space(P_g.T - np.eye(len(omegas))).reshape(-1,); pi_hat_g = pi_hat_g / pi_hat_g.sum()

print("Invariant probability distribution:")
for idx, el in enumerate(pi_hat_g): 
    print(f"node '{nodes[idx]}':" + "{:.4f}".format(el))

print("*"*50)
print("Question (h): Opinion dynamics with Graph modifications")

# new activation rate matrix
L_h = L.copy()

edge1, edge2 = "c,b", "d,a"
edges_to_remove = [edge1, edge2]
# removing edges
for e in edges_to_remove: 
    idx1, idx2 = node_index[e.split(",")[0]], node_index[e.split(",")[1]]
    L_h[idx1, idx2] = 0

# add self-loops for those nodes having a zero degree to be able to well define P
for row in range(L_h.shape[0]):
    L_h[row,row] = 1 if L_h[row,:].sum() <= 1e-9 else L_h[row, row]

D_h = np.diag(L_h.sum(axis=1))
P_h = np.linalg.inv(D_h) @ L_h

# simulating French-DeGroot in discrete time
n_opinions = 10
n_talks = int(1e2)

opinions = np.zeros((n_simulations, n_talks, len(omegas)))
starts = np.zeros((n_simulations, len(omegas)))
save_talks = False

for simulation in tqdm(range(n_simulations)): 
    starting_condition = np.random.choice(np.arange(n_opinions), size=len(omegas))
    starts[simulation, :] = starting_condition
    
    if save_talks: 
        talk_opinion = starting_condition
        # talking
        for talk in range(n_talks):
            talk_opinion = P_h @ talk_opinion
            opinions[simulation, talk, :] = talk_opinion
            intalks_variance[simulation, talk] = variance(talk_opinion)
        final_opinion = talk_opinion
    else: 
        # shouting one to each others real fast
        final_opinion = opinion_dynamics(matrix=P_h, initial_condition=starting_condition, n_talks=n_talks)

if save_talks:
    np.savetxt(fname="intalk_variance_(h).txt", X=intalks_variance)
    np.savetxt(fname="initialcondition_variance_(h).txt", X=starts)
    np.savetxt(fname="opinions_(h).txt", X=opinions.reshape(n_simulations * n_talks, -1))

# long-term probability distribution in discrete time
pi_hat_h = linalg.null_space(P_h.T - np.eye(len(omegas))).reshape(-1,); pi_hat_h = pi_hat_h / pi_hat_h.sum()

print("Invariant probability distribution for point (h):")
for idx, el in enumerate(pi_hat_h): 
    print(f"node '{nodes[idx]}':" + "{:.4f}".format(np.abs(el)))
print("*"*50)

print(f"{bcolors.BOLD}{bcolors.WARNING}" + "\t"*6 + "Exercise 1: complete!"+f"{bcolors.ENDC}{bcolors.ENDC}")