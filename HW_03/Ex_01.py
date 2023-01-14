import networkx as nx
import numpy as np
import pandas as pd
from IPython.core.display_functions import display

from HW_03.SwedenEpidemics import SwedenEpidemics
from HW_03.epidemics.Model import EpidemicsModel
from HW_03.epidemics.Population import Population, PopulationVax
from HW_03.epidemics.Utils import EpidemicsUtils


def execute():
    # Excercise 1.1.1
    print('Excercise 1.1.1')
    N = 500
    K = 4
    sir_model = EpidemicsModel('sir', ['s', 'i', 'r'])

    G = EpidemicsUtils.symmetric_regular_graph(N, K)
    individuals = EpidemicsUtils.from_graph_to_individuals(G, sir_model)
    p = Population(individuals, G, sir_model)
    print('Individuals inside the population:', p.amount_population)
    NUM_SIMULATION = 100
    n_weeks = 15
    final_results = EpidemicsUtils.simulate_epidemics_n_times(p, NUM_SIMULATION, n_weeks)

    EpidemicsUtils.draw_newly_infected(final_results, n_weeks, name='Ex1.1.1_newly_infected.png')
    EpidemicsUtils.draw_simulation(final_results, n_weeks, mapping_name={'i': 'infected', 's': 'susceptible', 'r': 'recovered'}, name='Ex1.1.1_average_totals.png')

    # Exericse 1.1.2
    print('Excercise 1.1.2')
    k = 17
    final_number_of_nodes = 900
    # generate the initial complete graph G0
    G = nx.complete_graph(k + 1)
    G = EpidemicsUtils.generate_random_graph(G, k, final_number_of_nodes)
    print(f"\nThe degree of the random graph is: {sum(dict(G.degree).values()) / final_number_of_nodes}")

    # Exericse 1.2
    print('Excercise 1.2')
    k = 6
    final_number_of_nodes = 500
    # generate the initial complete graph G0
    G = nx.complete_graph(k + 1)
    G = EpidemicsUtils.generate_random_graph(G, k, final_number_of_nodes)
    print(f"\nThe degree of the random graph is: {sum(dict(G.degree).values()) / final_number_of_nodes}")
    print(f'Number of nodes: {len(G.nodes)}')
    # Given the graph, we need to simulate what has been done in the previous exercise.
    individuals = EpidemicsUtils.from_graph_to_individuals(G, sir_model)
    p = Population(individuals, G, sir_model)
    print('Individuals inside the population:', p.amount_population)
    # parameters for the loop
    NUM_SIMULATION = 100
    n_weeks = 15
    final_results = EpidemicsUtils.simulate_epidemics_n_times(p, NUM_SIMULATION, n_weeks)
    EpidemicsUtils.draw_newly_infected(final_results, n_weeks, name='Ex1.2_newly_infected.png')
    EpidemicsUtils.draw_simulation(final_results, n_weeks, mapping_name={'i': 'infected', 's': 'susceptible', 'r': 'recovered'}, name='Ex1.2_average_totals.png')

    # Exercise 1.3
    print('Excercise 1.3')
    k = 6
    final_number_of_nodes = 500
    #  generate the initial complete graph G0
    G = nx.complete_graph(k + 1)
    G = EpidemicsUtils.generate_random_graph(G, k, final_number_of_nodes)
    print(f"\nThe degree of the random graph is: {sum(dict(G.degree).values()) / final_number_of_nodes}")
    print(f'Number of nodes: {len(G.nodes)}')
    NUM_SIMULATION = 100
    n_weeks = 15
    vacc = np.ones(n_weeks) * 0.60
    vacc[[0, 1, 2, 3, 4, 5, 6]] = [0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55]

    sir_model_with_vax = EpidemicsModel('sir_with_vax', ['s', 'i', 'r', 'v'])
    individuals = EpidemicsUtils.from_graph_to_individuals(G, sir_model_with_vax)
    p_vax = PopulationVax(individuals, G, sir_model_with_vax, vacc)
    print('Individuals inside the population:', p_vax.amount_population)
    final_results = EpidemicsUtils.simulate_epidemics_n_times(p_vax, NUM_SIMULATION, n_weeks)
    EpidemicsUtils.draw_simulation(final_results, n_weeks, mapping_name={'new_infected': 'Newly infected', 'new_vaccinated': 'Newly vaccinated'}, name='Ex1.3_newly_infected_vaccianted.png')
    EpidemicsUtils.draw_simulation(final_results, n_weeks, mapping_name={'i': 'infected', 's': 'susceptible', 'r': 'recovered', 'v': 'vaccinated'}, name='Ex1.3_average_totals.png')

    # Excercise 1.4
    print('Excercise 1.4')
    # how many different gridsearches are tested
    n_gridsearch = 5
    # how many times the same configuration in a gridsearch is tested
    NUM_SIMULATION = 10
    # for how many weeks
    n_weeks = 16
    final_number_of_nodes = 934
    # define Vacc(t)
    vacc = np.array([5, 9, 16, 24, 32, 40, 47, 54, 59, 60, 60, 60, 60, 60, 60, 60])
    vacc = (vacc / vacc.sum()).tolist()
    # set the ground truth, as per problem specifications
    I_0 = [1, 1, 3, 5, 9, 17, 32, 32, 17, 5, 2, 1, 0, 0, 0, 0]

    sweden_model = SwedenEpidemics(vacc, I_0)
    # Choose here if you wanna train the entire model or just evaluate it
    # sweden_model.train()
    sweden_model.eval()

    final_results, df_results = sweden_model.simulate(n_gridsearch, final_number_of_nodes, NUM_SIMULATION, n_weeks)
    print(f"Tested configuration:")
    display(df_results.iloc[[0]])
    print(f"Predicted new number of infected per week: {final_results['new_infected']}")
    print(f"True new number of infected per week: {I_0}")
    EpidemicsUtils.draw_simulation(final_results, n_weeks, mapping_name={'i': 'infected', 's': 'susceptible', 'r': 'recovered', 'v': 'vaccinated'}, name='Ex1.4_average_totals.png')
    pd.DataFrame({"Truth": I_0, "Estimation": final_results['new_infected']}).plot()\
        .get_figure().savefig('./simulation_imgs/Ex1.4_newly_infected_compared_with_truth.png')
