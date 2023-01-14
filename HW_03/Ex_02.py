import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import statistics
from HW_03.coloring.SimpleLineProblem import SimpleLine
import random
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
        p_next = simple_line.probability_next_colour(t, node.id, 0)
        # we are only interested in the probability of a change
        other_colour = [colour for colour in simple_line.states if colour != node.state][0]
        probability_change = p_next[other_colour]

        # pick a number at random
        if np.random.rand() < probability_change:
            node.update_state(other_colour)

        colour_dict = simple_line.nodes_statistics
        potentials.append(simple_line.find_potential())
    
    print("Final Potential: {:.4f}".format(potentials[-1]))

    # Exercise 2.2
    print('Exercise 2.2')

    wifi_problem = WifiAssignment()
    wifi_problem2 = WifiAssignment()
    # initialise the state of the system at t = 0, according to what prescribed in the text

    colorlist_all = ["red"] * 100
    colorlist = random.choices(["red", "green", "blue", "yellow", "magenta", "cyan", "white", "black"], k=100)
    # We initialized two wifi assignment instances in order to study the evolution of the dynamics from two different starting configurations
    wifi_problem.init_nodes(colorlist)
    wifi_problem2.init_nodes(colorlist_all)

    potentials = []
    potentials2 = []
    node_states_history = []

    for t in tqdm(range(5000), desc="Simulating (from all red)"):
        # choose uniformly at random one node
        node = np.random.choice(wifi_problem.nodes)
        # for each colour, retrieve the probability that this node will be of that colour at the next timestamp
        p_next = wifi_problem.probability_next_colour(t, node.id, 0)
        # find the next colour
        next_colour = np.random.choice(wifi_problem.states, p=list(p_next.values()))
        if node.state != next_colour:
            node.update_state(next_colour)

        potentials.append(wifi_problem.find_potential())
        node_states_history.append(wifi_problem.nodes)   
    print("Final potential (from all red): {:.4f}".format(potentials[-1]))

    potentials2 = []
    node_states_history = []
    for t in tqdm(range(5000), desc="Simulating (from random config)"):
        # choose uniformly at random one node
        node = np.random.choice(wifi_problem2.nodes)
        # for each colour, retrieve the probability that this node will be of that colour at the next timestamp
        p_next = wifi_problem2.probability_next_colour(t, node.id, 0)
        # find the next colour
        next_colour = np.random.choice(wifi_problem2.states, p=list(p_next.values()))
        if node.state != next_colour:
            node.update_state(next_colour)

        potentials2.append(wifi_problem2.find_potential())
        node_states_history.append(wifi_problem2.nodes)
    print("Final potential (from random config): {:.4f}".format(potentials[-1]))

    # Exercise 2.3
    print('Exercise 2.3')
    mean_potentials = []
    all_potentials = []
    print("Simulating with various etachoices...")

    expressions = {
        1: "eta(t) = 0, for all t >= 0",
        2: "eta(t) = 3, for all t >= 0",
        3: "eta(t) = 100, for all t >= 0",
        4: "eta(t) = t^4/100, for all t >= 0"
    }

    for etachoice in [1,2,3,4]: # Check the report for etachoice functional expression
        wifi_problem = WifiAssignment()
        # initialise the state of the system at t = 0, according to what prescribed in the text
        wifi_problem.init_nodes(['red'] * 100)
        potentials = []
        node_states_history = []
        for t in tqdm(range(5_000), desc="Simulating"):
            # choose uniformly at random one node
            node = np.random.choice(wifi_problem.nodes)
            # for each colour, retrieve the probability that this node will be of that colour at the next timestamp
            p_next = wifi_problem.probability_next_colour(t, node.id, etachoice)
            # find the next colour
            if np.nansum(list(p_next.values())) == 0:
                continue
            else:
                next_colour = np.random.choice(wifi_problem.states, p=list(p_next.values()))
                
                if node.state != next_colour:
                    node.update_state(next_colour)

            potentials.append(wifi_problem.find_potential())
            node_states_history.append(wifi_problem.nodes)

        print(f"With etachoice: {expressions[etachoice]}:")
        print(f"\tAverage Potential: {statistics.mean(potentials)}")
            
        all_potentials.append(potentials)
        mean_potentials.append(statistics.mean(potentials))
