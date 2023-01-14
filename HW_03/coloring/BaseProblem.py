import os
from abc import ABC, abstractmethod
from collections import Counter

import networkx as nx
from typing import List, Dict, Any

import numpy as np

from HW_03.coloring.Node import Node


class BaseProblem(ABC):

    def __init__(self, g: nx.Graph, possible_states: List, colour_map: Dict, w: np.array = None) -> None:
        if len(colour_map.keys()) != len(possible_states):
            raise ValueError('Length of possibile states and keys of color_map must be the same')
        self._graph = g
        self._states = possible_states
        self._colors = colour_map

        if w is not None:
            self._w = w
        else:
            self._w = nx.to_numpy_array(g.to_undirected())

        self._nodes = []
        for node in g.nodes:
            self._nodes.append(Node(node, ""))

        # create output folder for simulation
        self.final_path = "./HW_03/coloring_imgs/"
        if not os.path.exists(self.final_path):
            os.makedirs(self.final_path)

    @property
    def states(self) -> List:
        return self._states

    @property
    def weigths(self) -> np.array:
        return self._w

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    @property
    def colours(self) -> Dict:
        return self._colors

    @property
    def nodes_statistics(self) -> Dict:
        """Returns the number of nodes in each considered state"""
        return Counter([i.state for i in self.nodes])

    def init_nodes(self, colorlist: list) -> None:
        for node, new_state in zip(self._nodes, colorlist):
            node.update_state(new_state)

    @abstractmethod
    def cost_function(self, state1: Any, state2: Any) -> int:
        pass

    @abstractmethod
    def probability_next_colour(self, t: int, node_id: int, etachoice: int) -> Dict:
        pass

    @abstractmethod
    def draw(self, name: str = "", color_mapping: Dict = None, obj_param: Any = None) -> None:
        pass

    def find_potential(self) -> float:
        """Compute the potential in any given moment of the evolution, as per problem specification.
        This is used to study how close to a solution the learning algorithm is.
        Returns:
            float: potential according to the formula given in the exercise
        """
        sum_W = 0
        for node_i in self._nodes:
            for node_j in self._nodes:
                sum_W += self._w[node_i.id, node_j.id] * self.cost_function(node_i.state, node_j.state)
        U = 0.5 * sum_W
        return U
