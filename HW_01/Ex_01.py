from pprint import pprint
import pandas as pd
from HW_01.utils.utils import *


def execute():
    edges = [("o","a"),("a","d"),("o","b"),("b","d"),("b","c"),("c","d")]
    G = nx.DiGraph()
    # addind edges to the graph
    G.add_edges_from(edges)
    # fixing position of each vertex - for visualization purpose only
    pos = {"o":(0,0), "a":(1,1), "b":(1,0), "c":(1,-1), "d":(2,0)}

    visualize = False
    if visualize:
        nx.draw(G, pos, with_labels=True)

    # --------------- Question 1.1 -------------------------------------
    print("Question 1")
    # giving capacity to the graph G
    G = embed_capacity(G, capacity = np.array([2,1,2,1,2,1]))
    # obtaining maximal flow for this capacited graph
    results_flow = nx.algorithms.flow.maximum_flow(G,"o","d")
    # obtaining minimal cut for this capacited graph
    results_cut = nx.algorithms.flow.minimum_cut(G,"o","d")

    print("The maximal throughput is:", results_cut[0])
    print("The optimal flow vector is: \n")
    pprint(list(zip(G.edges, obtain_flow_vector(results_flow[-1], edges))))
    print("*"*50)

    # --------------- Question 1.2 -------------------------------------
    capacities = np.array([
        [1,1,2,1,1,1], # removing exactly c_left units of capacity from *non-saturated* edges
        [2,1,1,1,2,1], # removing less than c_left units of capacity from *saturated* edges
        [1,1,1,1,1,1]  # removing more than c_left (thus impacting both non saturated *and* saturated edges)
    ])

    descriptions = [
        "Removing c_left units of capacity from *non-saturated* edges only",
        "Removing less than c_left units of capacity from *saturated* edges only",
        "Removing more than c_left units of capacity (thus impacting both non saturated *and* saturated edges)"
    ]
    print("Question 2")
    for c, description in zip(capacities, descriptions):
        print()
        print(description)
        print()
        # changing to c the capacity of G
        new_G = embed_capacity(G, capacity = c)
        # obtaining maximal flow for this capacited graph
        results_flow = nx.algorithms.flow.maximum_flow(new_G,"o","d")

        print("The maximal throughput is:", results_flow[0])
        print("The optimal flow vector is: \n")
        pprint(list(zip(G.edges, obtain_flow_vector(results_flow[-1], edges))))

    # --------------- Question 1.3 -------------------------------------
    print("*"*50)
    print("Question 3")
    # node-link incidence matrix
    B = np.array([
        # e1 e2 e3  e4 e5 e6
        [+1, 0, +1, 0, 0, 0], # o
        [-1,+1,  0, 0, 0, 0], # a
        [0,  0, -1,+1,+1, 0], # b
        [0,  0,  0, 0,-1,+1], # c
        [0, -1,  0,-1, 0,-1]  # d
    ])
    # original capacity vector
    original_capacity = np.array([2,1,2,1,2,1])


    flows = Variable(shape = (len(edges),)) # the flow vector is in R^E.
    # out-flow source = in-flow sink = throughput
    source = Variable()
    sink = Variable()

    mass_conservation = hstack((source, np.zeros(len(G.nodes)-2), sink))

    maxflow = Problem(
        # objective function: maximize out-flow from source
        Maximize(source),
        # constraints
        [
            # conservation of mass: Bf = v
            B @ flows == mass_conservation,
            # non-negativity of flow
            0 <= flows,
            # capacity-wise feasibility
            flows <= original_capacity
        ]
    )
    maxflow.solve()

    cuts = [
        list("o"),
        list("oa"),
        list("ob"),
        list("oc"),
        list("oab"),
        list("oac"),
        list("obc"),
        list("oabc")
    ]

    df = pd.DataFrame(zip(
        edges,
        maxflow.variables()[1].value,
        original_capacity,
        np.array([
            "Active" if np.isclose(maxflow.variables()[1].value[i], original_capacity[i])
            else "Non Active" for i in range(len(maxflow.variables()[1].value))]
        ),
        # lagrangian multiplier associated to each capacity-feasibility constraint
        maxflow.constraints[-1].dual_value)
                     )

    df.columns = ["Edge", "Optimal Flow", "Capacity", "Constraint Activity", "Shadow Price"]
    df["Shadow Price"] = df["Shadow Price"].apply(lambda price: 0 if np.isclose(price, 0) else price)
    df.set_index(keys = ["Edge"], drop="True", inplace = True)
    print(df)
