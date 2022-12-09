import numpy as np
from typing import Union, Iterable, Tuple

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def opinion_dynamics(matrix:np.ndarray, initial_condition:np.ndarray=None, n_talks:int=int(1e2))->np.ndarray:
    """This function simulates the opinion dynamics mechanism considering as system matrix 'matrix' from 'initial_condition' for 'n_talks'.

    Args: 
        matrix (np.ndarray): Opinions system matrix
        initial_condition (np.array, optional): Starting condition for agents' opinion. When None, it is randomly generated. Defaults to None.
        n_talks (int, optional): Number of 'talks' in opinion mechanism.
    
    Returns: 
        np.ndarray: Agents' opinions after n_talks. If n_talks is large enough, it converges to a consensus vector (if any).
    """
    if initial_condition.shape[0] != matrix.shape[0]: 
        raise ValueError(f"Initial condition must be an array of shape ({matrix.shape[0]},)!")
    # tolerance on final variance to decleare consensus
    consensus_tol = 1e-9
    # after n_talks...
    return np.linalg.matrix_power(a=matrix, n=n_talks) @ initial_condition

def remove_i(arr:np.ndarray, idx:int)->np.ndarray:
    """This function remove rows and columns associated to index "idx"

    Args:
        arr (np.ndarray): input nd-array
        idx (int): index corresponding at row and column to delete

    Returns:
        np.array: new array after deletion
    """
    # first removes idx-th row and then idx-th column
    return np.delete(np.delete(arr=arr, obj=idx, axis=0), obj=idx, axis=1)

class Node: 
    def __init__(self, name:str, degree:float): 
        self.name = name
        self.degree = degree
    
    @property
    def rate(self): 
        return self.degree
    
    @rate.setter
    def set_rate(self, rate:float): 
        if rate > 0 and isinstance(rate, float): 
            self.degree = rate
        else: 
            raise ValueError(f"Rate must be a positive number whereas you provided {rate}")
    
    @rate.getter
    def get_rate(self): 
        return self.degree
    
    @rate.deleter
    def delete_rate(self): 
        self.degree = None
    
    @rate.getter
    def get_rate(self): 
        return self.rate

    def t_next(self)->float:
        """This function draws the next transition instant from an exponentially-distributed random variable
        parametrized by a given rate.

        Returns: 
            float: t_next, i.e. the moment in which transition will take place.
        """
        # sometimes rate might be 0, in that case t_next should simply be a very large number (such as 1e6)
        next_transition = -np.log(np.random.rand()) / self.get_rate if self.get_rate != 0 else 1e6
        return next_transition
    
class Walk: 
    def __init__(self, start_node:object, start_transition:float=None): 
        
        self.input_start = start_node

        self.walk = [start_node]
        self._visited_nodes = list(dict.fromkeys([start_node]))
        self.transition_times = [
            start_transition if start_transition is not None else start_node.t_next()
        ]
    @property
    def walk_nodes(self): 
        return [node.name for node in self.walk]

    @property
    def visited_nodes(self):
        return [node.name for node in self._visited_nodes]
    
    @property
    def times_array(self):
        """Converts self.transition_times to array to use numpy useful tools (such as masking etc)"""
        return np.array(self.transition_times)

    @property
    def walk_array(self):
        """Converts self.walk to array to use numpy useful tools (such as masking etc)"""
        return np.array(self.walk)
    
    def reset_history(self, node:object=None):
        """This function resets the walk"""
        # reset the whole walk history to either to arbitrary node or original input node
        self.walk = [node if node is not None else self.input_start]
        self._visited_nodes = list(dict.fromkeys([node])) if node is not None else list(dict.fromkeys([self.input_start]))
        self.transition_times = [node.t_next() if node is not None else self.input_start.t_next()]
    
    def add_node(self, node:object): 
        """This function adds a node to the considered path

        Raises: 
            ValueError: when node is not an instance of the Node class.
        Args:
            node (object): Node object according to Node API
        """
        if not isinstance(node, Node): 
            raise ValueError("Can't add non Node object to walk! Use Node API to code up a node")
        
        self.walk.append(node)
        self.transition_times.append(node.t_next())

        if node not in self._visited_nodes: 
            self._visited_nodes.append(node)
    
    def invariant_distribution(self)->list: 
        """This function estimates the stationary probability distribution using visited nodes and time spent
        in each node for a given walk.

        Returns: 
            list: list in which each element is
        """
        # walk array (to perform masking)
        walk_array = self.walk_array
        # times array (to perform masking)
        times_array = self.times_array

        # fraction of time in each node
        pi_bar = []
        total_time = times_array.sum()
        
        # the stationary probability distribution of node i is defined as the fraction of total time spent in node i 
        for visited_node in self._visited_nodes: 
            # when the path is in node 'visited_node'
            in_node_mask = walk_array == visited_node
            # along the whole walk, time spent in node
            time_spent_in_node = sum(times_array[in_node_mask])
            # append to pi_bar fraction of time in node 'visited_node'
            pi_bar.append(time_spent_in_node / total_time)
        
        return pi_bar

class RandomWalk(Walk): 
    def __init__(
        self, 
        start_node:object, 
        nodes:Iterable,
        P:np.ndarray):

        super().__init__(start_node=start_node)
        
        self.nodes = nodes
        self.P = P

        self.nodes_index = {node:idx for idx, node in enumerate(nodes)}

    def perform_walk(self, max_steps:int=None, max_time:float=None, return_walk:bool=False)->Union[None, list]:
        """This function performs the actual random walk starting from node self.start_node updating the inner
        state of the RandomWalk.
        
        Args: 
            max_steps (int, optional): number of transitions allowed. Defaults to None. 
            max_time (float, optional): maximal duratio of random walk. Defaults to None.
            return_walk (bool, optional): whether to return or not a list representing the actual walk

        Raises: 
            ValueError: max_steps and max_time cannot be both defined or None at the same time.

        Returns:
            Union[None, list]: either None if the walk has not to be returned or the list of Node objects
                               visited during the walk"""
        
        # checking input sanity with logical "xor"
        if not((max_steps is None) ^ (max_time is None)): 
            print(f"Provided inputs - max_steps={max_steps}, max_time={max_time}")
            raise ValueError("max_steps and max_time cannot be defined or none at the same time!")
        
        # stopping condition
        if max_steps is not None: 
            condition = lambda times_array: len(times_array) <= max_steps
        else:
            condition = lambda times_array: times_array.sum() <= max_time
        
        current_node = self.walk[-1]
        
        while condition(self.times_array):
            # retrieve index of current node (to access matrix P)
            current_index = self.nodes_index[current_node]
            # probabilities of jumping to all other nodes
            jump_probabilities = self.P[current_index, :]
            # choosing next node according to jump probability
            next_node = np.random.choice(self.nodes, p=jump_probabilities)
            # storing the visit to next node in the walk
            self.add_node(next_node)
            # overwrite current node
            current_node = next_node

            if max_steps is not None: 
                print(f"Step {len(self.times_array)} of {max_steps}", end="\r")
            else: 
                print(f"Elapsed time: {self.times_array.sum()} out of {max_time}", end="\r")
                
        if return_walk:
            return self.walk

    def walk_until(self, target_node:object, forget:bool=True, remove_last:bool=True)->Tuple[list, float]:
        """This function performs a random walk until a target node 'target_node' is met.

        Args:
            target_node (object): Target node to be used
            forget (bool, optional): Whether or not to completely forget past walk(s). Defaults to True.
            remove_last (bool, optional): Whether or not to remove last element in the temporal array (useful to differentiate 
                                          between return time and hitting time). Defaults to True.
        
        Returns: 
            Tuple[list, float]: Respectively, the actual walk from starting node to target node and the time it took to
                                walk the walk.
        """
        condition = lambda walk: len(walk) > 1 and walk[-1] == target_node
        
        if forget:
            self.reset_history()

        current_node = self.walk[-1]
        while not condition(self.walk):
            # retrieve index of current node (to access matrix P)
            current_index = self.nodes_index[current_node]
            # probabilities of jumping to all other nodes
            jump_probabilities = self.P[current_index, :]
            # choosing next node according to jump probability
            next_node = np.random.choice(self.nodes, p=jump_probabilities)
            # storing the visit to next node in the walk
            self.add_node(next_node)
            # overwrite current node with next node
            current_node = next_node
        
        if remove_last:
            # remove last t_next
            self.transition_times.pop(-1)
        
        return self.walk, self.times_array.sum()
    
class CounterNode(Node): 
    """Overrides rate method of node to make the rate dependant on the number of particles inside too."""
    def __init__(self, name:str, degree:float, nparticles_init:int=0):
        super().__init__(name=name, degree=degree) 
        self.nparticles = nparticles_init

    def increment_particles(self, increment:int=1): 
        """This function increments internal number of particles by a quantity equal to increment"""
        self.nparticles += increment
    
    def decrement_particles(self, decrement:int=1): 
        """This function decrements internal number of particles by a quantity equal to increment"""
        self.nparticles = max(0, self.nparticles - decrement)
    
    @property
    def rate(self):
        """Overrides parent rate with a rate proportional to number of particles"""
        return self.degree * self.nparticles

class NodeRandomWalk:
    def __init__(self, counting_nodes:Iterable[CounterNode], P:np.ndarray, constant_particles:bool=True):
        self.nodes = counting_nodes
        self.matrix = P
        # whether or not the number of particles is to be considered constant (no input from outside)
        self.constant_particles = constant_particles
        self.total_particles = sum([node.nparticles for node in self.nodes])
        self.nodes_index = {node:idx for idx, node in enumerate(self.nodes)}

        # making every node tick
        self.ticks = np.array([node.t_next() for node in self.nodes])
        # storing earliest tick and quickest node to tick
        self.clocks = [np.min(self.ticks)]
        self.quick_nodes = [self.nodes[np.argmin(self.ticks)]]
        # list of list having as many elements as the number of nodes (one sublist for each node, each storing number of particles evolution)
        self.nparticles_history = [[n.nparticles] for n in self.nodes]
    
    @property
    def times_array(self): 
        return np.array(self.clocks)
    
    def append_newtick(self): 
        # storing the value of the node that ticked first
        earliest_tick = np.min(self.ticks)
        self.clocks.append(earliest_tick)

    def append_newticker(self):
        # storing the quickest node to tick
        quickest_node = np.argmin(self.ticks)
        self.quick_nodes.append(self.nodes[quickest_node])
    
    def update_ticks(self): 
        # making every node tick
        self.ticks = np.array([node.t_next() for node in self.nodes])
    
    def walk_until(self, max_transitions:int=None, max_time:float=None)->Union[np.ndarray, np.array]:
        # checking input sanity with logical "xor"
        if not((max_transitions is None) ^ (max_time is None)): 
            print(f"Provided inputs - max_transitions={max_transitions}, max_time={max_time}")
            raise ValueError("max_steps and max_time cannot be defined or none at the same time!")
        
        # stopping condition
        if max_transitions is not None: 
            condition = lambda arr: len(arr) <= max_transitions
        else:
            condition = lambda arr: arr.sum() <= max_time
        
        while condition(self.times_array):
            current_node = self.quick_nodes[-1]
            jump_probabilities = self.matrix[self.nodes_index[current_node], :]
            next_node = np.random.choice(self.nodes, p=jump_probabilities)
            # move one particle from one node to another
            if next_node == current_node: 
                pass
            else: 
                current_node.decrement_particles()
                next_node.increment_particles(increment = 0 if self.constant_particles and next_node.nparticles == self.total_particles else 1)

            # record changes in number of particles in nodes
            for n in self.nodes:
                self.nparticles_history[self.nodes_index[n]].append(n.nparticles)
            
            # recompute ticks
            self.update_ticks()
            # update times array
            self.append_newtick()
            # update list of nodes that ticked
            self.append_newticker()
        
        # return evolution of number of particles and time axis
        return np.array(self.nparticles_history), self.times_array