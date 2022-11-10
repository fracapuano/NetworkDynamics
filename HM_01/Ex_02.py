from utils.utils import *

#Graph initialization and representation
G = nx.DiGraph()
G.add_edges_from([('p1','b1'),('p1','b2'),
                  ('p2','b2'),('p2','b3'),
                  ('p3','b1'),('p3','b4'),
                  ('p4','b1'),('p4','b2'),('p4','b4')])

pos = {'p1':[0,4], 'p2':[0,3], 'p3':[0,2], 'p4':[0,1],
       'b1':[1,4], 'b2':[1,3], 'b3':[1,2], 'b4':[1,1]}

nx.draw(G, pos, with_labels=True)


# --------------- Question 2.1 -------------------------------------
#Auxiliary graph initialization and representation
G = nx.DiGraph()

G.add_edge("o", "p1", capacity=1.0)
G.add_edge("o", "p2", capacity=1.0)
G.add_edge("o", "p3", capacity=1.0)
G.add_edge("o", "p4", capacity=1.0)

G.add_edge("p1", "b1", capacity=1.0)
G.add_edge("p1", "b2", capacity=1.0)
G.add_edge("p2", "b2", capacity=1.0)
G.add_edge("p2", "b3", capacity=1.0)
G.add_edge("p3", "b1", capacity=1.0)
G.add_edge("p3", "b4", capacity=1.0)
G.add_edge("p4", "b1", capacity=1.0)
G.add_edge("p4", "b2", capacity=1.0)
G.add_edge("p4", "b4", capacity=1.0)

G.add_edge("b1", "d", capacity=1.0)
G.add_edge("b2", "d", capacity=1.0)
G.add_edge("b3", "d", capacity=1.0)
G.add_edge("b4", "d", capacity=1.0)

pos = {'o':[-1, 2.5],
       'p1':[0,4], 'p2':[0,3], 'p3':[0,2], 'p4':[0,1],
       'b1':[1,4], 'b2':[1,3], 'b3':[1,2], 'b4':[1,1],
       'd':[2,2.5]}

nx.draw(G,pos,with_labels=True)


#Maximum-flow using Ford-Fulkerson algorithm
max_flow, max_flow_dict = nx.maximum_flow(G, "o", "d")

print("The maximum flow:", max_flow, "\n")
print("The flow distribution: \n", max_flow_dict)


# --------------- Question 2.2 -------------------------------------
#Graph initialization using the updated capacities
G = nx.DiGraph()

G.add_edge("o", "p1", capacity=4.0)
G.add_edge("o", "p2", capacity=4.0)
G.add_edge("o", "p3", capacity=4.0)
G.add_edge("o", "p4", capacity=4.0)

G.add_edge("p1", "b1", capacity=1.0)
G.add_edge("p1", "b2", capacity=1.0)
G.add_edge("p2", "b2", capacity=1.0)
G.add_edge("p2", "b3", capacity=1.0)
G.add_edge("p3", "b1", capacity=1.0)
G.add_edge("p3", "b4", capacity=1.0)
G.add_edge("p4", "b1", capacity=1.0)
G.add_edge("p4", "b2", capacity=1.0)
G.add_edge("p4", "b4", capacity=1.0)

G.add_edge("b1", "d", capacity=2.0)
G.add_edge("b2", "d", capacity=3.0)
G.add_edge("b3", "d", capacity=2.0)
G.add_edge("b4", "d", capacity=2.0)

#Maximum-flow computation and flow distribution
max_flow, max_flow_dict = nx.maximum_flow(G, "o", "d")

print("The maximum flow:", max_flow, "\n")
print("The flow distribution: \n", max_flow_dict)


# --------------- Question 2.3 -------------------------------------
#Graph initialization using the updated capacities
G = nx.DiGraph()

G.add_edge("o", "p1", capacity=np.Inf)
G.add_edge("o", "p2", capacity=np.Inf)
G.add_edge("o", "p3", capacity=np.Inf)
G.add_edge("o", "p4", capacity=np.Inf)

G.add_edge("p1", "b1", capacity=1.0)
G.add_edge("p1", "b2", capacity=1.0)
G.add_edge("p2", "b2", capacity=1.0)
G.add_edge("p2", "b3", capacity=1.0)
G.add_edge("p3", "b1", capacity=1.0)
G.add_edge("p3", "b4", capacity=1.0)
G.add_edge("p4", "b1", capacity=1.0)
G.add_edge("p4", "b2", capacity=1.0)
G.add_edge("p4", "b4", capacity=1.0)

G.add_edge("b1", "d", capacity=2)
G.add_edge("b2", "d", capacity=3)
G.add_edge("b3", "d", capacity=2)
G.add_edge("b4", "d", capacity=2)


global B
B = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #o
              [-1,0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #p1
              [0,-1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #p2
              [0, 0,-1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  #p3
              [0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  #p4
              [0, 0, 0, 0,-1, 0, 0, 0,-1, 0,-1, 0, 0, 1, 0, 0, 0],  #b1
              [0, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0,-1, 0, 0, 1, 0, 0],  #b2
              [0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  #b3
              [0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0,-1, 0, 0, 0, 1],  #b4
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1,-1,-1],  #d
             ])

edges_list = [["o", "p1"], ["o", "p2"], ["o", "p3"], ["o", "p4"],
              ["p1", "b1"], ["p1", "b2"], ["p2", "b2"], ["p2", "b3"],
              ["p3", "b1"], ["p3", "b4"], ["p4", "b1"], ["p4", "b2"], ["p4", "b4"],
              ["b1", "d"], ["b2", "d"], ["b3", "d"], ["b4", "d"]]
capacity_array=np.array([1e4,1e4,1e4,1e4,1,1,1,1,1,1,1,1,1,2,3,2,2])   #1e8 is the biggest integer power that does
                                                                       #cause errors

obtain_multipliers(B=B, edges=edges_list, G=G, capacity=capacity_array)
