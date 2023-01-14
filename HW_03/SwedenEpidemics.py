import itertools
import random
from typing import Iterable, Tuple, Dict
from IPython.display import clear_output

import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame

from HW_03.epidemics.Model import EpidemicsModel
from HW_03.epidemics.Population import PopulationVax
from HW_03.epidemics.Utils import EpidemicsUtils


class SwedenEpidemics:
    def __init__(self, vaccination_scheme: Iterable[int], ground_truth: Iterable[int]) -> None:
        self.vax_scheme = vaccination_scheme
        self.truth = ground_truth
        self._training = False

        self.DELTA_BETA_THRESHOLD = 0.0001

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    def simulate(self, n_grisearch: int, final_number_of_nodes: int, n_simulations: int, n_weeks: int,
                 k: int = 3, beta: float = 0.3871, rho: float = 0.5992) -> Tuple[Dict, DataFrame]:
        """Function to estimate k, beta, rho for the Swedish H1N1 epidemic
            Args:
                final_number_of_nodes (int, optional): number of people in the Swedish network. Defaults to 934.
                If training == True, set the parameters for the simulation:
                    n_simulations (int, optional): how many times the same configuration is simulated. Defaults to 10.
                    n_weeks (int, optional): number of weeks through which a given configuration is simulated. Defaults to 16.
                    n_grisearch (int, optional): how many different gridsearches are simulated. Each gridsearch takes ~3 min. Defaults to 1.
                    So in total there are n_simulations * N * n_weeks iterations.
                    The gridsearch works as follow:
                        - k_0 is a random integer between 5 and 14. delta_k is set to 6.
                        - beta_0 and rho_0 are random float between 0 and 1. The corresponding deltas are both set to 0.5.
                        - the gridsearch is built as
                            {
                                "K" : [k_0 - delta_k, k_0, k_0 + delta_k],
                                "Beta" : [beta_0 - delta_beta, beta_0 + delta_beta),
                                "Rho" : [rho_0 - delta_rho, rho_0 + delta_rho]
                            }
                        Some attention is paid to make sure the values are always valid and the same hyperparameter cannot have the same value twice.
                If training == False, the parameters are set as follows:
                    k (int, optional): average degree k. Defaults to 13.
                    beta (float, optional): probability that the infection is spread from an infected individual to a susceptible one. Defaults to 0.11.
                    rho (float, optional):  probability that an infected individual will recover during one time step. Defaults to 0.437.

                    These values are the results of a pre-ran tested configuration, whose yielded results can be found in the dataframe which comes with this repo.
                    To work that out, we ran a simulation with N = 100 and n_simulations = 5.
            Returns:
                dict: recap per week, returning the results to answer the exercise question.
                pd.DataFrame: DataFrame containing the results of the simulations.
            """
        df_results = None
        if self._training:
            for i in range(n_grisearch):
                # define initial parameters
                # they are all randomly chosen
                k_0 = random.randint(5, 15)
                delta_k = 6
                beta_0 = random.random()
                delta_beta = 0.5
                rho_0 = random.random()
                delta_rho = 0.5
                # keep track of the current minimum
                current_min = {"k": None, "beta": None, "rho": None, "RMSE": None}
                while True:
                    # define parameter space
                    # if the deltas are too small, break the loop
                    if delta_beta < self.DELTA_BETA_THRESHOLD:
                        break

                    # create a gridsearch
                    parameter_space = {
                        "K": ([max(2, k_0 - delta_k), max(3, k_0), k_0 + delta_k] if delta_k > 0 else [k_0]),
                        "Beta": [max(0.001, beta_0 - delta_beta), min(0.999, beta_0 + delta_beta)],
                        "Rho": [max(0.001, rho_0 - delta_rho), min(0.999, rho_0 + delta_rho)]
                    }

                    # store results for this configuration
                    keys, values = zip(*parameter_space.items())
                    df_results_this_sim = pd.DataFrame([dict(zip(keys, v)) for v in itertools.product(*values)])
                    # set initial RMSE to a really big number
                    df_results_this_sim["RMSE"] = 100000000
                    # hide Pandas annoying warning
                    pd.options.mode.chained_assignment = None

                    # iterate through the different configurations
                    for k in parameter_space["K"]:
                        # cast k to an integer, to make sure it is not interpreted as a float
                        k = int(k)
                        #  generate the initial complete graph G0
                        G = nx.complete_graph(k + 1)
                        G = EpidemicsUtils.generate_random_graph(G, k, final_number_of_nodes, seed=0)

                        # choose parameters
                        for beta in parameter_space["Beta"]:
                            for rho in parameter_space["Rho"]:
                                print(f"[{i + 1}/{n_simulations}]: ")
                                print(f"Current argmin = {current_min}")
                                print(f"k : {k_0} ± {delta_k} = {k}", end="\t")
                                print(f"beta : {beta_0} ± {delta_beta} = {beta}", end="\t")
                                print(f"rho : {rho_0} ± {delta_rho} = {rho}")
                                # simulate contagion
                                sir_model_with_vax = EpidemicsModel('sir_with_vax', ['s', 'i', 'r', 'v'])
                                individuals = EpidemicsUtils.from_graph_to_individuals(G, sir_model_with_vax)
                                p_vax = PopulationVax(individuals, G, sir_model_with_vax, self.vax_scheme)
                                recap_per_week = EpidemicsUtils.simulate_epidemics_n_times(p_vax, n_simulations,
                                                                                           n_weeks, n_infected_t0=1,
                                                                                           beta=beta, rho=rho, seed=0)
                                # recap_per_week = simulate_contagion(G, beta, rho, N, n_weeks, n_infected_t0=1, vaccination_scheme=vaccination_scheme, seed=0)
                                # compute I_t
                                I_t = recap_per_week["new_infected"]
                                # compute RMSE
                                RMSE = np.sqrt(np.sum((I_t - self.truth) ** 2) / 15)
                                # add RMSE to the dataframe for this configuration
                                df_results_this_sim.loc[
                                    (df_results_this_sim["K"] == k) & (df_results_this_sim["Beta"] == beta) & (
                                            df_results_this_sim["Rho"] == rho), ["RMSE"]] = RMSE

                    # find the best row, i.e. the argminimiser of RMSE
                    min_row = df_results_this_sim["RMSE"].argmin()
                    new_min = {
                        "k": df_results_this_sim.iloc[min_row]["K"],
                        "beta": df_results_this_sim.iloc[min_row]["Beta"],
                        "rho": df_results_this_sim.iloc[min_row]["Rho"],
                        "RMSE": df_results_this_sim.iloc[min_row]["RMSE"]
                    }

                    # check if current_min is empty:
                    if current_min["k"] is None:
                        # if empty, it means we are at the first iterations.
                        current_min = new_min.copy()
                        # update values
                        k_0 = current_min["k"]
                        beta_0 = current_min["beta"]
                        rho_0 = current_min["rho"]
                        # half all deltas
                        delta_k = max(delta_k - 1, 0)
                        delta_beta /= 2
                        delta_rho /= 2
                    # if non-empty, update.
                    else:
                        # if the new minimum is smaller (with a margin) than the old minimum,
                        # update parameters and current minimum
                        if df_results_this_sim.iloc[min_row]["RMSE"] < (current_min["RMSE"] - 0.1):
                            k_0 = new_min["k"]
                            beta_0 = new_min["beta"]
                            rho_0 = new_min["rho"]
                            current_min = new_min.copy()
                        else:
                            # the minimum has not changed. Repeat everything, halving all deltas.
                            # k cannot become negative.
                            delta_k = max(delta_k - 1, 0)
                            delta_beta /= 2
                            delta_rho /= 2

                    # clear output
                    # This make sure the user is constantly given feedback, but old information is removed
                    clear_output(wait=True)
                    # at the end of the simulation, store the dataframe of this simulation to the global dataframe
                    if df_results is None:
                        df_results = df_results_this_sim.copy()
                    else:
                        df_results = pd.concat([df_results, df_results_this_sim])
            # find the best amongst the best configurations
            best_configuration_idx = df_results["RMSE"].argmin()
            k, beta, rho, _ = df_results.iloc[best_configuration_idx]
            k = int(k)

        #  generate the initial complete graph G0
        G = nx.complete_graph(k + 1)
        G = EpidemicsUtils.generate_random_graph(G, k, final_number_of_nodes)
        sir_model_with_vax = EpidemicsModel('sir_with_vax', ['s', 'i', 'r', 'v'])
        individuals = EpidemicsUtils.from_graph_to_individuals(G, sir_model_with_vax)
        p_vax = PopulationVax(individuals, G, sir_model_with_vax, self.vax_scheme)
        recap_per_week = EpidemicsUtils.simulate_epidemics_n_times(p_vax, n_simulations, n_weeks, n_infected_t0=1,
                                                                   beta=beta, rho=rho)
        #recap_per_week = simulate_contagion(G, beta, rho, N, n_weeks, vaccination_scheme=vaccination_scheme, n_infected_t0=1, seed=0)
        I_t = recap_per_week["new_infected"]
        RMSE = np.sqrt(np.sum((I_t - self.truth) ** 2) / len(I_t))

        if self._training:
            return recap_per_week, df_results.sort_values(by="RMSE", ascending=True)
        else:
            return recap_per_week, pd.DataFrame({"k": k, "beta": beta, "rho": rho, "RMSE": RMSE}, index=[0])
