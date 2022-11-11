import numpy as np
import networkx as nx
from typing import Tuple
from cvxpy import Variable, Problem, Maximize, hstack


def embed_capacity(G: nx.DiGraph, capacity: np.array) -> nx.DiGraph:
    """
    This function returns a graph object with capacity vector "capacity".
    Args:
        G (nx.DiGraph): Object representing a graph in networkx API.
        capacity (np.array): np.array containing the capacities. MUST BE IN THE SAME ORDER OF G.edges

    Returns:
        nx.DiGraph: Object representing a graph in networkx API.
    """
    G_ = G.copy()
    for edge, c in zip(G.edges, capacity):
        G_[edge[0]][edge[-1]]["capacity"] = c
    return G_


def evaluate_cut(G: nx.DiGraph, nodes: list, pos: dict, visualize: bool = True) -> Tuple[float, list]:
    """
    This function evaluates the cut considering the nodes in "nodes" and returns cut's capacity and cross-edges.
    Args:
        G (nx.DiGraph): Graph object whose cut is to evaluate.
        nodes (list): Nodes in the cut considered.
        visualize (bool, optional): Whether or not to visualize the results.

    Returns:
        Tuple[float, list]: Cut's capacity value and list of edges outside the cut, respectively.
    """
    cut_capacity = 0
    capacities = []
    edges = []

    for edge in G.edges:
        if edge[0] in nodes and edge[-1] not in nodes:  # tail of edge in cut and head of edge not in cut
            cut_capacity += G[edge[0]][edge[-1]]["capacity"]  # incrementing capacity
            edges.append(edge)  # storing edge
            if visualize:
                capacities.append(G[edge[0]][edge[-1]]["capacity"])

    if visualize:
        nx.draw(G, pos, with_labels=True)
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels={key: value for key, value in zip(edges, capacities)}, font_color='red')
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=["red" if n in nodes else "#1f78b4" for n in G.nodes])

    return (cut_capacity, edges)


def obtain_flow_vector(flow_dict: dict, edges: list) -> np.array:
    """
    This function converts the flow represented in the dictionary returned by flow maximization in a numpy array.
    Args:
        flow_dict (dict): Dictionary representing the flow.
        edges (list): List of edges on which the flow is to be reconstructed.

    Returns:
        np.array: flow vector
    """
    flow_vector = np.zeros(len(edges))
    for idx, edge in enumerate(edges):
        flow_vector[idx] = flow_dict[edge[0]][edge[1]]

    return flow_vector


def increment_capacity(capacity: np.array, increment_idx: int, increment_value: float = 1.0) -> np.array:
    """
    This function performs a capacity increment of value increment_value on the capacity stored in capacity at
    index increment_idx.
    Args:
        capacity (np.array): The capacity array to be modified.
        increment_idx (int): Where to perform the capacity increment.
        increment_value (float, optional): The value of the increment. Defaults to float(1).

    Returns:
        np.array: The incremented capacity.
    """
    new_capacity = capacity.copy().astype(np.float64)  # to possibly store all increments.
    new_capacity[increment_idx] += increment_value  # performing increment
    return new_capacity


def increment_index(multipliers: np.array, target: float = 1.) -> list:
    """
    This function retrieves the value at which to perform a (possibly continous) marginal constraint relaxation
    considering the lagrangian multipliers associated with such constraints.
    Args:
        multipliers (np.array): Lagrangian multipliers associated to the constraints considered.
        target (float): Value to look for in the multipliers vector.

    Returns:
        list: Index/Indices associated to the constraint to marginally relax.
    """
    return np.isclose(multipliers, target)


def obtain_throughput(G: nx.DiGraph, capacity: np.array) -> int:
    """
    This function returns the throughput associated to a capacited graph.
    Args:
        G (nx.DiGraph): Object representing the graph of interest. MUST HAVE EDGES ORDERED AS THOSE IN CAPACITY.
        capacity (np.array): Array of capacities. MUST BE ORDERED AS THE EDGES OF G ARE ORDERED THEMSELVES.

    Returns:
        int: Throughput of the graph.
    """
    G_capacity = embed_capacity(G=G, capacity=capacity)
    # this function assumes the source node is "o" and the destination node is "d"
    if "o" not in G.nodes or "d" not in G.nodes:
        raise ValueError(f"This function only works for flows from node 'o' to node 'd', which are not in {G.nodes}")

    return nx.algorithms.flow.maximum_flow(G_capacity, "o", "d")[0]


def get_flow(G: nx.DiGraph, capacity: np.array) -> int:
    """
    This function returns the optimal flow vector associated to a capacited graph.
    Args:
        G (nx.DiGraph): Object representing the graph of interest. MUST HAVE EDGES ORDERED AS THOSE IN CAPACITY.
        capacity (np.array): Array of capacities. MUST BE ORDERED AS THE EDGES OF G ARE ORDERED THEMSELVES.

    Returns:
        np.array: Array in which each element represents the flow on any given edge of the graph.
    """
    G_capacity = embed_capacity(G=G, capacity=capacity)
    # this function assumes the source node is "o" and the destination node is "d"
    if "o" not in G.nodes or "d" not in G.nodes:
        raise ValueError(f"This function only works for flows from node 'o' to node 'd', which are not in {G.nodes}")
    flow_dict = nx.algorithms.flow.maximum_flow(G_capacity, "o", "d")[1]

    return obtain_flow_vector(flow_dict=flow_dict, G=G_capacity)


def obtain_multipliers(B: np.array, edges: list, G: nx.DiGraph, capacity: np.array) -> Tuple[list, np.array]:
    """
    This function defines and solves the Max-Flow problem for a given capacity vector (which parametrizes the
    constraints), returning the associated Lagrangian multipliers.
    Args:
        capacity (np.array): Capacity vector that defines the constraints.

    Returns:
        Tuple[list, np.array]: Result such as (edge, Lagrangian multiplier) associated to the optimal solution
                               of the primal of the Max-Flow problem.
    """
    # this function relies on cvxpy API.
    flows = Variable(shape=(len(edges),), nonneg=True)  # the flow vector is in (R^E)-plus.
    # out-flow source = in-flow sink = throughput
    source = Variable()
    sink = Variable()
    # nu
    mass_conservation = hstack((source, np.zeros(len(G.nodes) - 2), sink))
    # actual problem
    maxflow = Problem(
        # objective function: maximize out-flow from source
        Maximize(source),
        # constraints
        [
            # conservation of mass: Bf = v
            B @ flows == mass_conservation,
            # capacity-wise feasibility (non-negativity of flow in variable definition)
            flows <= capacity
        ]
    )
    # solving the problem
    maxflow.solve()
    # returning (edge, multiplier values)
    return edges, maxflow.constraints[-1].dual_value
