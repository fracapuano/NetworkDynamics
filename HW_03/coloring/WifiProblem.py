from typing import Dict, Any

import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from HW_03.coloring.BaseProblem import BaseProblem


class WifiAssignment(BaseProblem):

    def __init__(self, w: np.array = None) -> None:
        if w is None:
            w = np.loadtxt('./HW_03/coloring/dataset/wifi.mat')
        colour_map = {
            "red": 1, "green": 2, "blue": 3, "yellow": 4,
            "magenta": 5, "cyan": 6, "white": 7, "black": 8
        }
        possible_states = list(colour_map.keys())
        g = nx.from_numpy_matrix(w, parallel_edges=False, create_using=nx.Graph())

        super().__init__(g, possible_states, colour_map, w)
        self.routers_coords = np.loadtxt('./HW_03/coloring/dataset/coords.mat')

    def cost_function(self, state1: Any, state2: Any) -> int:
        """Given two states, compute the cost function.
            The cost is 2 if the two states have the same colour,
            1 if the colours are 1 integer apart according to the dictionary given in the text,
            0 otherwise.
            Args:
                state1, state2 (str): Two in `possible_states`. Can be the same
            Returns:
                int: 2 if they have matching colour, 1 if they are adjacent, 0 otherwise.
            """
        colour_int_1 = self.colours[state1]
        colour_int_2 = self.colours[state2]
        if state1 == state2:
            return 2
        elif np.abs(colour_int_1 - colour_int_2) == 1:
            return 1
        else:
            return 0

    def probability_next_colour(self, t: int, node_id: int) -> Dict:
        """Given a discrete timestamp t and a node previously randomly chosen, return the conditioned probabilities describing the colour
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
                # compute the cost function c(s, X_j(t)) = 1 if X_j(t) = s, 0 otherwise
                c_sj = self.cost_function(s, node.state)
                c.append(c_sj)
            probabilities[s] = np.exp(-1 * nu * (self.weigths[node_id, :] @ c))

        # normalise the entries so that they sum up to 1
        probabilities = {key: value / sum(probabilities.values()) for key, value in probabilities.items()}

        return probabilities

    def draw(self, name: str = "Ex2.2_wifi_assignment.png", color_mapping: Dict = None, obj_param: Any = None) -> None:
        if obj_param is None or not isinstance(obj_param, list):
            raise ValueError('Colors map array is needed')

        cmap = matplotlib.colors.ListedColormap(obj_param)
        node_size = [300 for i in range(len(self.graph.nodes))]
        edge_colors = ['gray' for i in range(len(self.graph.nodes))]
        coords_dict = {i: coords for i, coords in enumerate(self.routers_coords)}

        plt.clf()
        nx.draw(self.graph, node_color=obj_param, with_labels=True, cmap=cmap, pos=coords_dict,
                node_size=node_size, edge_color=edge_colors, edgecolors='gray', font_size=8)
        plt.savefig(self.final_path + name, dpi=300, bbox_inches='tight')