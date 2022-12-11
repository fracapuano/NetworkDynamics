from utils.markov_utils import *
from tqdm import tqdm
from scipy import linalg
import time

def execute():
    L = np.array([
        [0, 2/5, 1/5, 0, 0],
        [0, 0, 3/4, 1/4, 0],
        [1/2, 0, 0, 1/2, 0],
        [0, 0, 1/3, 0, 2/3],
        [0, 1/3, 0, 1/3, 0]
    ])

    node_names = ["o", "a", "b", "c", "d"]
    # node degrees
    omegas = L @ np.ones(L.shape[0])
    # degree matrix
    D = np.diag(omegas)
    # jump probability matrix (row stochastic)
    P = np.linalg.inv(D) @ L
    # Laplacian matrix and invariant probability
    Lapl = D - L
    # stationary probability is defined as kernel of Laplacian matrix (transposed)
    Lapl_kernel = linalg.null_space(Lapl.T)
    pi_bar =  (Lapl_kernel/Lapl_kernel.sum()).reshape(-1,)
    # initial number of particles
    init_particles = [100, 0, 0, 0, 0]
    n_particles = sum(init_particles)
    # number of simulations
    n_simulations = int(5e3)

    print("Point (a): Particle perspective")
    # creating node objects
    node_objects = [Node(name=node_name, degree=degree) for node_name, degree in zip(node_names, omegas)]
    # nodes
    node_o, node_a, node_b, node_c, node_d = node_objects

    from_a = RandomWalk(start_node=node_a, nodes=node_objects, P=P)

    durations = np.zeros((n_simulations, n_particles))
    for simulation in tqdm(range(n_simulations)):
        for particle in range(n_particles):
            _, duration = from_a.walk_until(target_node=node_a)
            durations[simulation, particle] = duration

    # average duration in iteration *per particle*
    avg_perparticle = durations.mean(axis=1)
    # average over all particles
    total_avg = avg_perparticle.mean()

    print("Average (empirical) Return Time from node '{}' : {:.4f} t.u.".format(
        from_a.input_start.name, total_avg
    ))
    print("*"*50)

    print("Point (b): Node Perspective")
    n_simulations = int(5e2)
    node_objects = [
        CounterNode(name=node_name, degree=omegas[idx], nparticles_init=init_particles[idx]) for idx, node_name in enumerate(node_names)
    ]

    node_perspective = NodeRandomWalk(counting_nodes=node_objects, P=P, constant_particles=True)
    max_time = 60 # time units
    # uncomment following line to increase max_time to 240 and improve similarity of final state to stationary distribution
    # max_time = 240

    final_states = np.zeros((n_simulations, len(omegas)))
    trajectories, times, timelens, simulation_times  = [], [], [], []

    for simulation in tqdm(range(n_simulations)):
        # measuring how much time a single walk during 'max_time' takes
        start_walks = time.time()
        n_t, t = node_perspective.walk_until(max_time=max_time)
        end_walks = time.time() - start_walks
        # storing the system's state at last timestep considered
        final_states[simulation, :] = n_t[:,-1]
        times.append(t); timelens.append(len(t)), trajectories.append(n_t)
        simulation_times.append(end_walks)

    avg_final = final_states.mean(axis=0)
    print("Average Terminal State (number of particles in each node)")

    for idx, value in enumerate(avg_final):
        node = node_names[idx]
        print("\tNode '{}' - {:.4f} particles / Invariant Probability: {:.4f}".format(node, value, pi_bar[idx]))

    print("Average Number of Transitions in {} t.u.: {:.4f}".format(max_time, np.array(timelens).mean()))

    save_trajectories = False
    if save_trajectories:
        np.savetxt(fname="trajectories.txt", X=np.array(trajectories).reshape((n_simulations*len(omegas), -1)))
        np.savetxt(fname="times.txt", X=np.array(times))

    print("*"*50)
    print(f"{bcolors.BOLD}{bcolors.WARNING}" + "\t"*6 + "Exercise 2: complete!"+f"{bcolors.ENDC}{bcolors.ENDC}")