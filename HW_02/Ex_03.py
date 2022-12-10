import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

# this is useful to export plots with Latex fonts
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


# Plot the evolution of the number of particles in each node over time
def plot_simulation(img_name, evolution_of_particles, times, trans_matrix, name_nodes, force_one_node=None,
                    save_img=False):
    fig = plt.figure(dpi=200, figsize=(9, 3))
    ax = fig.add_subplot(111)
    ax.set_position([0.075, 0.15, 0.85, 0.8])

    if force_one_node is not None and isinstance(force_one_node, int) and 0 <= force_one_node < len(
            trans_matrix):
        evolution = evolution_of_particles.T[force_one_node]
        ax.plot(times, evolution, label='Node ' + name_nodes[force_one_node], linewidth=0.8)
    else:
        for node_index in range(len(trans_matrix)):
            evolution = evolution_of_particles.T[node_index]
            ax.plot(times, evolution, label='Node ' + name_nodes[node_index], linewidth=0.8)

    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of particles')

    # Save image
    if save_img:
        fig.savefig(f'./HW_02/simulation_imgs/{img_name}.eps')


def plot_all_nodes_simulation(img_name, evolution_of_particles, times, trans_matrix, name_nodes, save_img=False):
    fig = plt.figure(dpi=180, figsize=(11, 6))
    fig.subplots_adjust(hspace=0.3)

    colors = plt.cm.get_cmap('hsv', len(trans_matrix) + 1)

    for node_index in range(len(trans_matrix)):
        ax = fig.add_subplot(2, 3, node_index + 1)
        ax.plot(times, evolution_of_particles.T[node_index], label='Node ' + name_nodes[node_index],
                color=colors(node_index))
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
        ax.set_xlabel('Time')
        if node_index == 0 or node_index == 3:
            ax.set_ylabel('Number of particles')
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.25),
            ncol=3,
        )

    # Save image
    if save_img:
        fig.savefig(f'./HW_02/simulation_imgs/{img_name}.eps')


# n is N(t)
# it returns the next time and the index associated to the next node that has to move
def next_time(w, n):
    product = w * n
    times = [-np.log(np.random.rand()) / rate_i for rate_i in product]
    t_next = min(times)
    index = times.index(t_next)
    return t_next, index


def next_time_with_rate(rate=1):
    return -np.log(np.random.rand()) / rate


# n is N(t)
# it returns the next time and the index associated to the next node that has to move
def next_time_w_rate(w):
    times = [-np.log(np.random.rand()) / w_i for w_i in w]
    t_next = min(times)
    index = times.index(t_next)
    return t_next, index


# n is N(t)
# it returns the updated state of the system in which there is a new particle in node O
def introduce_new_particle(n):
    n[0] = n[0] + 1
    return n


# trans_matrix is
# index_node is
# it returns the index of the node that has to receive particles
def moves(trans_matrix, index_node):
    thrs = random.random()
    comulate = 0
    for index, value in enumerate(list(trans_matrix[index_node])):
        comulate += value
        if thrs <= comulate:
            return index
    return index


def execute():
    # define constants of the problem
    SAVE_SIMULATION_IMAGES = True

    Lambda = np.array([
        [0, 3 / 4, 3 / 8, 0, 0],
        [0, 0, 1 / 4, 1 / 4, 2 / 4],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]])

    TIME_UNITS_SIMULATION = 60
    # default Poisson rate
    ALPHA = 1
    # number of node in this system
    N_NODES = len(Lambda)

    w = Lambda.dot(np.ones(len(Lambda)))
    # Since node d does not have a node to send its particles to, we assume that ωd = 2.
    w[len(Lambda) - 1] = 2

    # Normalize the transition rate matrix
    D = np.diag(w)
    P = np.linalg.inv(D) @ Lambda

    # A dictiory of nodes just to print out the name of nodes
    name_nodes = {0: 'o', 1: 'a', 2: 'b', 3: 'c', 4: 'd'}

    # a) Proportional rate
    # Number of particles in each node
    n = np.zeros(len(Lambda))

    # Poisson clock
    clock = 0
    time_assix = [clock]  # it's needed to plot the evolution of the system over time
    evolution_of_particles = np.array(
        [np.zeros(len(Lambda))])  # it's needed to plot the evolution of the system over ti

    print('Starting simulation a): Propotional rate')
    while clock <= TIME_UNITS_SIMULATION:
        (t_next, index) = next_time(w, n)

        # I've to know wether introduce a new particle on the system or moves already exist particles
        if t_next < next_time_with_rate(ALPHA):
            # Move already exists particles
            if name_nodes[index] == "d":
                # When the Poisson clock ticks for node D, you could simply decrease the number of particles in the node by one
                # print('Move particle outside from the system')
                if n[index] > 0:
                    n[index] -= 1
            else:
                next_node = moves(P, index)  # it's the next node to receive particles
                n[next_node] += 1
                n[index] -= 1
        else:
            # Introduce new particle
            introduce_new_particle(n)

        clock += min(t_next, next_time_with_rate(ALPHA))

        time_assix.append(clock)
        evolution_of_particles = np.append(evolution_of_particles, [n], axis=0)
        print('t = %.2f' % clock + ":", n)

    print('Saving simulation plots...')
    plot_simulation('Proportional rate simulation', evolution_of_particles, time_assix, P, name_nodes
                    , save_img=SAVE_SIMULATION_IMAGES)
    plot_all_nodes_simulation('All nodes - Proportional rate simulation', evolution_of_particles, time_assix, P
                              , name_nodes, save_img=SAVE_SIMULATION_IMAGES)

    # b) Fixed rate
    # Number of particles in each node
    n = np.zeros(len(Lambda))

    # Poisson clock
    clock = 0
    time_assix = [clock]  # it's needed to plot the evolution of the system over time
    evolution_of_particles = np.array(
        [np.zeros(len(Lambda))])  # it's needed to plot the evolution of the system over ti

    print('Starting simulation b): Fixed rate')
    while clock <= TIME_UNITS_SIMULATION:
        (t_next, index) = next_time_w_rate(w)

        # I've to know wether introduce a new particle on the system or moves already exist particles
        if t_next < next_time_with_rate(ALPHA):
            # Move already exists particles
            if name_nodes[index] == "d":
                # When the Poisson clock ticks for node D, you could simply decrease the number of particles in the node by one
                # print('Move particle outside from the system')
                if n[index] > 0:
                    n[index] -= 1
            else:
                next_node = moves(P, index)  # it's the next node to receive particles
                # print(f'Move particle from node {name_nodes[index]} to node {name_nodes[next_node]}')
                if n[index] > 0:
                    n[next_node] += 1
                    n[index] -= 1
                # else:
                # print('There are no particles in node ' + str(name_nodes[index]))
        else:
            # Introduce new particle
            # print('Introduce new particle')
            introduce_new_particle(n)

        clock += min(t_next, next_time_with_rate(ALPHA))

        time_assix.append(clock)
        evolution_of_particles = np.append(evolution_of_particles, [n], axis=0)
        print('t = %.2f' % clock + ":", n)

    print('Saving simulation plots...')
    plot_simulation('Fixed rate simulation', evolution_of_particles, time_assix, P, name_nodes,
                    save_img=SAVE_SIMULATION_IMAGES)
    plot_all_nodes_simulation('All nodes - Fixed rate simulation', evolution_of_particles, time_assix, P,
                              name_nodes, save_img=SAVE_SIMULATION_IMAGES)