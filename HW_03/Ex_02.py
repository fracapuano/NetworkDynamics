import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from HW_03.coloring.SimpleLineProblem import SimpleLine
from HW_03.coloring.WifiProblem import WifiAssignment


def execute():
    # Exercise 2.1
    print('Exercise 2.1')
    # define the possible states
    simple_line = SimpleLine()
    # initialise the state of the system at t = 0, according to what prescribed in the text
    simple_line.init_nodes("r")
    # initialise an empty potential list -> to plot results
    potentials = []

    for t in tqdm(range(10_000), desc="Simulating"):
        # choose uniformly at random one node
        node = np.random.choice(simple_line.nodes)
        # for each colour, retrieve the probability that this node will be of that colour at the next timestamp
        p_next = simple_line.probability_next_colour(t, node.id)
        # we are only interested in the probability of a change
        other_colour = [colour for colour in simple_line.states if colour != node.state][0]
        probability_change = p_next[other_colour]

        # pick a number at random
        if np.random.rand() < probability_change:
            node.update_state(other_colour)

        colour_dict = simple_line.nodes_statistics
        potentials.append(simple_line.find_potential())

    # Plotting
    simple_line.draw()
    plt.clf()
    pd.Series(potentials).plot()\
        .get_figure().savefig('./HW_03/coloring_imgs/Ex2.1_potentials.png')

    # Exercise 2.1
    print('Exercise 2.2')
    wifi_problem = WifiAssignment()
    # initialise the state of the system at t = 0, according to what prescribed in the text
    wifi_problem.init_nodes('red')
    potentials = []
    node_states_history = []
    for t in tqdm(range(5_000), desc="Simulating"):
        # choose uniformly at random one node
        node = np.random.choice(wifi_problem.nodes)
        # for each colour, retrieve the probability that this node will be of that colour at the next timestamp
        p_next = wifi_problem.probability_next_colour(t, node.id)
        # find the next colour
        next_colour = np.random.choice(wifi_problem.states, p=list(p_next.values()))
        if node.state != next_colour:
            node.update_state(next_colour)

        potentials.append(wifi_problem.find_potential())
        node_states_history.append(wifi_problem.nodes)

    # Plotting
    plt.clf()
    pd.Series(potentials).plot()\
        .get_figure().savefig('./HW_03/coloring_imgs/Ex2.2_potentials.png')

    index = np.argmin(potentials)
    print(f"Potential becomes 0 after {index} iterations")
    color_map_graph = [node.state for node in node_states_history[index]]
    wifi_problem.draw(obj_param=color_map_graph)