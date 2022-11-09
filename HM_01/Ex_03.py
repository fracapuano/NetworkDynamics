from scipy import io as sc
import cvxpy as cp
from pprint import pprint
from utils.utils import *


#Graph initialization and representation
G = nx.DiGraph()
G.add_edges_from([("1", "2"), ("1", "6"), ("2", "3"), ("2", "7"),
                ("3", "4"), ("3", "8"), ("3", "9"), ("4", "5"),
                ("4", "9"), ("5", "14"), ("6", "7"), ("6", "10"),
                ("7", "8"), ("7", "10"), ("8", "9"), ("8", "11"),
                ("9", "12"), ("9", "13"), ("10", "11"), ("10", "15"),
                ("11", "12"), ("11", "15"), ("12", "13"), ("13", "14"),
                ("13", "17"), ("14", "17"), ("16", "17"), ("15", "16"), ])

pos = nx.spring_layout(G)

# nx.draw_networkx_edge_labels(G,pos,edge_labels={"o","a"):'2',
# (0,2):'2',(2,3):'2'},font_color='red')

nx.draw(G, pos, with_labels = True)


#Reshape
f = sc.loadmat('utils/flow.mat')["flow"].reshape(28, )
C = sc.loadmat('utils/capacities.mat')["capacities"].reshape(28, )
B = sc.loadmat('utils/traffic.mat')["traffic"]
l = sc.loadmat('utils/traveltime.mat')["traveltime"].reshape(28, )

#Dictionary to convert edges to capacities
conversion_edges_to_capacity = {('1', '2'): 0, ('2', '3'): 1, ('3', '4'): 2,
                                ('4', '5'): 3, ('1', '6'): 4, ('6', '7'): 5,
                                ('7', '8'): 6, ('8', '9'): 7, ('9', '13'): 8,
                                ('2', '7'): 9, ('3', '8'): 10, ('3', '9'): 11,
                                ('4', '9'): 12, ('5', '14'): 13, ('6', '10'): 14,
                                ('10', '11'): 15, ('10', '15'): 16,
                                ('7', '10'): 17, ('8', '11'): 18,
                                ('9', '12'): 19, ('11', '12'): 20,
                                ('12', '13'): 21, ('13', '14'): 22,
                                ('11', '15') : 23, ('13', '17'): 24,
                                ('14', '17'): 25, ('15', '16'): 26,
                                ('16', '17') : 27
                               }

for edge in G.edges:
    edge_1, edge_2 = edge
    position_capacity = conversion_edges_to_capacity[edge]
    G[edge_1][edge_2]["capacity"] = C[position_capacity]
    G[edge_1][edge_2]["weight"] = l[position_capacity]


# --------------- Question 3.1 -------------------------------------
#Shortest path, found as the shortest path in an empty network
print(nx.shortest_path(G, source="1", target="17", weight = "weight"))


# --------------- Question 3.2 -------------------------------------
#Max-flow between nodes 1 and 17
flow_value, _ = nx.maximum_flow(G, "1", "17")
print(flow_value)


# --------------- Question 3.3 -------------------------------------
#Given the flow vector in flow.mat, compute the external inflow ν satisfying Bf = ν.
v = B @ f
pprint(v)


# --------------- Question 3.4 -------------------------------------
n_edges = len(C)

f_ = cp.Variable(n_edges)
pi_ = []
cost_function_ = []

for el in range(n_edges):
    cost_function_.append(
        cp.multiply(l[el] * C[el] , cp.inv_pos(1 - (f_[el] / C[el]))) - l[el] * C[el]
    )

objective = cp.Minimize(cp.sum(cost_function_))

nu = np.zeros(len(v))
nu[0] = v[0]
nu[-1] = -v[0]

constraints = [B @ f_ == nu, f_ >= 0, f_ <= C]

prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# result = prob.solve(solver = "SCS")#, verbose = True)
# The optimal value for f is stored in `f.value`.
print("Optimal f:", f_.value)
opt_flow = f_.value


# --------------- Question 3.5 -------------------------------------
# Construct the problem.
f_w = cp.Variable(n_edges)
cost_function = 0
for el in range(n_edges):
    integral_e = - cp.multiply( C[el] * l[el] , (cp.log(1 - f_w[el] / C[el])))
    cost_function += integral_e
objective_w = cp.Minimize(cost_function)
constraints_w = [B @ f_w == nu, f_w >=0, f_w <= C]
prob_w = cp.Problem(objective_w, constraints_w)

# The optimal objective value is returned by `prob.solve()`.
result_w = prob_w.solve()
# The optimal value for f is stored in `f.value`.
print("Wardrop equilibrium:", f_w.value)


# --------------- Question 3.6 -------------------------------------
# find marginal tolls: for each edge the toll equals f*_e d_e'(f_e*)
# for affine delays l_e + b_e f_e, this reads b_e f*_e
# in this example b = [1,0]
omega = opt_flow * C * l / (opt_flow - C) ** 2


# Construct the problem.
f_t = cp.Variable(n_edges)
cost_function_toll = 0

for el in range(n_edges):
    integral_e = - cp.multiply( C[el] * l[el] , (cp.log(1 - f_t[el] / C[el])) )
    cost_function_toll += integral_e + omega[el] * f_t[el]

objective_t = cp.Minimize(cost_function_toll)
constraints_t = [B @ f_t == nu, f_t >=0, f_t <= C]
prob_t = cp.Problem(objective_t, constraints_t)

# The optimal objective value is returned by `prob.solve()`.
result_t = prob_t.solve()
# The optimal value for f is stored in `f.value`.
print("f at Wardrop equilibrium with tolls:", f_t.value)



# --------------- Question 3.7 ------------------------------------
n_edges = len(C)

f_ = cp.Variable(n_edges)
pi_ = []
cost_function_ = []

for el in range(n_edges):
    cost_function_.append(
        cp.multiply(l[el] * C[el] , cp.inv_pos(1 - (f_[el] / C[el]))) - l[el] * C[el] - cp.multiply(f_[el], l[el])
    )


objective = cp.Minimize(cp.sum(cost_function_))

nu = np.zeros(len(v))
nu[0] = v[0]
nu[-1] = -v[0]

constraints = [B @ f_ == nu, f_ >=0, f_ <= C]

prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# result = prob.solve(solver = "SCS")#, verbose = True)
# The optimal value for f is stored in `f.value`.
print("Optimal f:", f_.value)
opt_flow = f_.value


# Construct the problem.
f_w = cp.Variable(n_edges)
cost_function = 0
for el in range(n_edges):
    integral_e = C[el] * l[el] * (1 - cp.log(1 - f_w[el] / C[el]))
    cost_function += integral_e
objective_w = cp.Minimize(cost_function)
constraints_w = [B @ f_w == nu, f_w >=0, f_w <= C]
prob_w = cp.Problem(objective_w, constraints_w)

# The optimal objective value is returned by `prob.solve()`.
result_w = prob_w.solve()
# The optimal value for f is stored in `f.value`.
print("Wardrop equilibrium:", f_w.value)


# find marginal tolls: for each edge the toll equals f*_e d_e'(f_e*)
# for affine delays l_e + b_e f_e, this reads b_e f*_e
# in this example b = [1,0]
omega = opt_flow * C * l / (opt_flow - C) ** 2 - l

# Construct the problem.
f_t = cp.Variable(n_edges)
cost_function_toll = 0

for el in range(n_edges):
    integral_e = - cp.multiply( C[el] * l[el] , (cp.log(1 - f_t[el] / C[el])) )
    cost_function_toll += integral_e + omega[el] * f_t[el]

objective_t = cp.Minimize(cost_function_toll)
constraints_t = [B @ f_t == nu, f_t >=0, f_t <= C]
prob_t = cp.Problem(objective_t, constraints_t)

# The optimal objective value is returned by `prob.solve()`.
result_t = prob_t.solve()
# The optimal value for f is stored in `f.value`.
print("Wardrop equilibrium with tolls:", f_t.value)