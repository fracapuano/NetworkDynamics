import math

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List, Dict

from HW_03.epidemics.Individual import Individual
from HW_03.epidemics.Model import EpidemicsModel
from HW_03.epidemics.Population import Population


class EpidemicsUtils:

    @staticmethod
    def symmetric_regular_graph(n: int = 8, k: int = 4) -> nx.Graph:
        """Generate a symmetric k-regular undirected graph with N nodes where every node is directly connected
          to the K nodes whose index is closest to their own modulo N.
          Args:
              n (int, optional): number of nodes. Defaults to 8.
              k (int, optional): number of links to every node. Defaults to 4.
          Returns:
              graph: the symmetric k-regular undirected graph
          """
        if (k % 2) != 0:
            raise ValueError("The parameter K must be an even number")
        elif k >= n:
            raise ValueError("The parameter K must less than the total number of nodes")
        G = nx.Graph()
        G.add_nodes_from(range(n))
        half_k = int(k / 2)
        for node in range(n):
            for x in range(1, half_k + 1):
                index_to_add = (node + x) % n
                G.add_edge(node, index_to_add)
                G.add_edge(index_to_add, node)
            for x in range(1, half_k + 1):
                index_to_add = (node - x) % n
                G.add_edge(node, index_to_add)
                G.add_edge(index_to_add, node)
        return G

    @staticmethod
    def generate_random_graph(G: nx.Graph, k: int = 10, target: int = 900, seed: int = None) -> nx.Graph:
        """Given a graph, a target degree k, and a target number of nodes, return a random Graph following the preferential attachment rule
        Args:
            G (nx.Graph): graph at state 0.
            k (int, optional): average degree k. Defaults to 10.
            target (int, optional): desired number of nodes at the end of the simulation. Defaults to 900.
            seed (int, optional): seed to get reproducible results. Defaults to None.
        Returns:
            nx.Graph : the final graph with `target` nodes.
        """
        if seed is not None:
            np.random.seed(seed)
        # target should take into account that the graph is non-empty at first.
        target = target - len(G.nodes)
        for i in tqdm(range(target), desc="Generating random graph"):
            new_node = k + 1 + (i + 1)
            # decide how many edges to add alternate between ceiling function and floor function, according to what
            # prescribed in the text of the exercise
            c = (math.floor if i % 2 else math.ceil)(k / 2)

            # create a list with new_node repeated c times. This will come in handy when generating the edged
            new_node = [new_node] * c

            # work out the probabilities
            degrees = dict(G.degree()).values()
            probabilities = [degree / sum(degrees) for degree in degrees]

            # choose the nodes to connect to
            linked_nodes = np.random.choice(G.nodes(), size=c, p=probabilities, replace=False)
            # create the links
            links_n_to_l = list(zip(new_node, linked_nodes))  # links from node to linked nodes
            # make them undirected
            links_l_to_n = list(zip(linked_nodes, new_node))  # links from linked nodes to node
            links = links_n_to_l + links_l_to_n

            # create the new links
            G.add_edges_from(links)
        return G

    @staticmethod
    def from_graph_to_individuals(G: nx.Graph, epidemics_model: EpidemicsModel) -> List[Individual]:
        individuals = []
        for i in list(G.nodes):
            individuals.append(Individual(i, epidemics_model))
        return individuals

    @staticmethod
    def simulate_epidemics_n_times(p: Population, num_simulations: int, n_weeks: int, n_infected_t0: int = 10,
                                   beta: float = 0.3, rho: float = 0.7,
                                   seed: int = None) -> Dict:
        final_results = {}
        for _ in tqdm(range(num_simulations),
                      desc="Simulating contagion with " + str(n_weeks) + " weeks for " + str(
                          num_simulations) + " times"):
            recap_per_week = p.simulate_epidemic(n_weeks, n_infected_t0=n_infected_t0, beta=beta, rho=rho, seed=seed)
            for state, recap in recap_per_week.items():
                if state in final_results:
                    final_results[state].append(recap)
                else:
                    final_results[state] = [recap]
        for state, recap in final_results.items():
            final_results[state] = np.mean(np.array(recap), axis=0)
        return final_results

    @staticmethod
    def draw_newly_infected(final_results: dict, n_weeks: int, name: str = 'newly_infected.png') -> None:
        # The average number of newly infected individuals each week
        fig = plt.figure(dpi=100, figsize=(9, 3))
        ax = fig.add_subplot(111)
        ax.set_position([0.075, 0.15, 0.85, 0.8])
        ax.plot([i + 1 for i in range(n_weeks)], final_results['new_infected'], linewidth=0.8)
        ax.set_xlabel('Week')
        ax.set_ylabel('Number of newly infected individuals')
        plt.savefig('./simulation_imgs/' + name)
        print('Saved ' + name + ' images into /simulation_imgs folder')

    @staticmethod
    def draw_simulation(final_results: dict, n_weeks: int, mapping_name: dict = None,
                        name: str = 'average_totals.png') -> None:
        # The average total number of susceptible, infected, and recovered individuals at each week
        if mapping_name is None:
            mapping_name = {}
        fig = plt.figure(dpi=100, figsize=(9, 3))
        ax = fig.add_subplot(111)
        ax.set_position([0.075, 0.15, 0.85, 0.8])
        for key in final_results.keys():
            if key in mapping_name and isinstance(mapping_name[key], str):
                ax.plot([i + 1 for i in range(n_weeks)], final_results[key], label=mapping_name[key], linewidth=0.8)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.set_xlabel('Week')
        ax.set_ylabel('Number of individuals')
        plt.savefig('./simulation_imgs/' + name)
        print('Saved ' + name + ' images into /simulation_imgs folder')
