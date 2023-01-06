import networkx as nx
from typing import Dict, Any

import numpy as np
from matplotlib import pyplot as plt

from HW_03.coloring.BaseProblem import BaseProblem


class SimpleLine(BaseProblem):

    def __init__(self) -> None:
        g = nx.DiGraph()
        g.add_edges_from([(i, i + 1) for i in range(9)])
        possible_states = ['r', 'y']
        colour_dict = {"y": [], "r": []}

        super().__init__(g, possible_states, colour_dict)

    def cost_function(self, state1: Any, state2: Any):
        """Given two states, compute the cost function.
            The cost is 1 the two states have the same colour, 0 otherwise.
            Args:
                state1, state2 (str): Two in `possible_states`. Can be the same
            Returns:
                int: 1 if they have matching colour, 0 otherwise.
            """
        return int(state1 == state2)

    def probability_next_colour(self, t: int, node_id: int):
        """Given a discrete timestamp t and a node previously randomly chosen, return the two conditioned probabilities describing the colour
            of `node` at time `t+1`
            Args:
                t (int): discrete timestamp.
                node_id (int): node previously randomly chosen
            Returns:
                dict: dictionary. For each possible colour (key), it gives the probabilities for the node to be of that colour at time t+1.
            """
        nu = t / 100
        # this dictionary will contain the probabilities, following the formula given in the text
        probabilities = {}
        for s in self.states:
            c = []
            for node in self.nodes:
                c_sj = self.cost_function(s, node.state)
                c.append(c_sj)
            probabilities[s] = np.exp(-1 * nu * (self.weigths[node_id, :] @ c))

        # normalise the entries so that they sum up to 1
        probabilities = {key: value / sum(probabilities.values()) for key, value in probabilities.items()}

        return probabilities

    def draw(self, name: str = "Ex2.1_line_graph.png", color_mapping: Dict = None, obj_param: Any = None) -> None:
        if color_mapping is not None and list(color_mapping.keys()) != self.states:
            raise ValueError('Color mapping must have the same keys as in possible states')

        node_colors = []
        if color_mapping is None:
            color_mapping = {'r': 'red', 'y': 'green'}
        for node in self.nodes:
            node_colors.append(color_mapping[node.state])

        plt.clf()
        nx.draw(self.graph, node_color=node_colors, with_labels=True)
        plt.savefig('./coloring_imgs/' + name, dpi=300, bbox_inches='tight')