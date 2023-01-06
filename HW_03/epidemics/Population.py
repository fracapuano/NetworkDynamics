from typing import Iterable, Dict, List
from collections import Counter
import networkx as nx
import numpy as np

from HW_03.epidemics.Individual import Individual
from HW_03.epidemics.Model import EpidemicsModel


class Population:
    def __init__(self, population: Iterable[Individual], topology: nx.Graph, epidemic_model: EpidemicsModel) -> None:
        # sanity check on individuals
        if not all([isinstance(i, Individual) for i in population]):
            raise ValueError("Individuals in population must be instances of 'Individual' class!")
        # sanity check on population
        if not all([i.model == epidemic_model for i in population]):
            print(f"Epidemic model {epidemic_model} differs from epidemic model of individuals in population")
            raise ValueError(f"{population[-1].model} differs from {epidemic_model}")

        self._population = population
        self._topology = topology
        self._model = epidemic_model

    @property
    def population(self) -> Iterable[Individual]:
        return self._population

    @property
    def amount_population(self) -> int:
        return len([i.state for i in self.population])

    def update_individual(self, individual_index: int, individual_newstate: str) -> None:
        """Updates the state of an individual, identified with their index, to `individual_newstate`"""
        # self._population[individual_index].update_state(new_state=individual_newstate)
        self.get_individuals_by_id(individual_index).update_state(individual_newstate)

    @property
    def population_statistics(self) -> Dict:
        """Returns the number of individuals in each considered state"""
        return Counter([i.state for i in self.population])

    def get_individuals_by_state(self, state: str) -> List[Individual]:
        """Returns all the individuals that match with the state parameter"""
        return [i for i in self.population if i.state == state]

    def get_individuals_not_in_state(self, state: str) -> List[Individual]:
        """Returns all the individuals that NOT match with the state parameter"""
        return [i for i in self.population if i.state != state]

    def get_individuals_by_id(self, id: int) -> Individual:
        """Returns a single individual"""
        for individual in self.population:
            if individual.id == id:
                return individual
        raise ValueError(f'No individual found with id {id}')

    def simulate_epidemic(self, n_weeks: int, n_infected_t0: int = 10, beta: float = 0.3, rho: float = 0.7,
                          seed: int = None) -> Dict:
        """Actually simulates the epidemics, simulating new infections and recovers"""
        # set the seed, to reproduce results
        if seed is not None:
            np.random.seed(seed)
        # initialise variables to store results to answer the questions
        recap_per_week = {
            state: np.zeros(n_weeks) for state in self._model.states
        }
        recap_per_week["new_infected"] = np.zeros(n_weeks)
        # initialise the status of the system at t = 0. We start with all S and 10 random I.
        for idx in self.population:
            self.update_individual(idx.id, 's')
        infected_nodes = np.random.choice(range(self.amount_population), size=n_infected_t0, replace=False)
        for infected in infected_nodes:
            self.update_individual(self._population[infected].id, 'i')

        for week in range(n_weeks):
            # variable to keep track of the number of new infected
            new_infected = []
            # isolate the susceptibles
            susceptibles = self.get_individuals_by_state('s')
            if len(susceptibles) > 0:
                # retrieve how many infected in the neighbourhood of each susceptible
                for susceptible in susceptibles:
                    # m is the number of infected neighbours, as per the problem specifications
                    m = len([neighbour for neighbour in self._topology.neighbors(susceptible.id) if
                             self.get_individuals_by_id(neighbour).state == "i"])
                    # compute the probability of infection, according to the formula given in the text
                    p_infection = 1 - (1 - beta) ** m
                    # susceptible becomes infected with probability p_infection
                    if np.random.rand() < p_infection:
                        new_infected.append(susceptible)

            # isolate the infected
            infected_people = self.get_individuals_by_state('i')
            if (n_infected := len(infected_people)) > 0:
                # for each one of them, pick a random number if this number < rho, infected becomes removed.
                probs_recovery = np.random.rand(n_infected, 1).reshape(n_infected, )
                # recovery or not
                recovery = np.array([prob_recovery < rho for prob_recovery in probs_recovery])
                # isolate the recovered
                recovered = np.array(infected_people)[recovery]
                # make them R
                for r in recovered:
                    self.update_individual(r.id, 'r')

            # turn the susceptibles into infected
            for new_infected_individual in new_infected:
                self.update_individual(new_infected_individual.id, 'i')

            # how many individuals in total are susceptible/infected/recovered at each week (to answer question 1.1.2)
            for state, number in self.population_statistics.items():
                recap_per_week[state][week] = number
            recap_per_week["new_infected"][week] = len(new_infected)

        return recap_per_week


class PopulationVax(Population):
    def __init__(self, population: Iterable[Individual], topology: nx.Graph, epidemic_model: EpidemicsModel,
                 vaccination_scheme: Iterable[int]) -> None:
        super().__init__(population=population, topology=topology, epidemic_model=epidemic_model)
        self.vax_scheme = vaccination_scheme

    def _length_vax_scheme(self) -> int:
        return sum(1 for e in self.vax_scheme)

    def simulate_epidemic(self, n_weeks: int, n_infected_t0: int = 10, beta: float = 0.3, rho: float = 0.7,
                          seed: int = None) -> Dict:
        """Actually simulates the epidemics, simulating new infections and recovers, as well as vaccinations"""
        if self._length_vax_scheme() != n_weeks:
            raise ValueError("Vacc(t) should be defined for every week.")
        # set the seed, to reproduce results
        if seed is not None:
            np.random.seed(seed)
        # initialise variables to store results to answer the questions
        recap_per_week = {
            state: np.zeros(n_weeks) for state in self._model.states
        }
        recap_per_week["new_infected"] = np.zeros(n_weeks)
        recap_per_week["new_vaccinated"] = np.zeros(n_weeks)
        # initialise the status of the system at t = 0. We start with all S and 10 random I.
        for idx in self.population:
            self.update_individual(idx.id, 's')
        infected_nodes = np.random.choice(range(self.amount_population), size=n_infected_t0, replace=False)
        for infected in infected_nodes:
            self.update_individual(self._population[infected].id, 'i')

        for week in range(n_weeks):
            # isolate non_vaccinated
            non_vaccinated = self.get_individuals_not_in_state('v')
            # compute how many people are already vaccinated
            n_vaccinated = self.amount_population - len(non_vaccinated)
            # compute how many there are
            n_nonvacc = len(non_vaccinated)
            # vacc(t) refers to the global population. We need to convert the proportion of population in proportion of nonvaccinated
            proportion_population_week = self.vax_scheme[week]
            if proportion_population_week > 0:
                # the number of vaccinated + the number of new vaccinated should be proportion_population_week% of
                # the total population convert this proportion into number of people
                target_n_vaccinations = self.amount_population * proportion_population_week
                # work out the number of new vaccinations
                n_people_to_jab = target_n_vaccinations - n_vaccinated
                new_vaccinated = np.random.choice(range(n_nonvacc), int(n_people_to_jab), replace=False)
                # vaccination is immediately effective.
                for v in new_vaccinated:
                    self.update_individual(self._population[v].id, 'v')
            else:
                new_vaccinated = []

            new_infected = []
            # isolate the susceptibles
            susceptibles = self.get_individuals_by_state('s')
            if len(susceptibles) > 0:
                # retrieve how many infected in the neighbourhood of each susceptible
                for susceptible in susceptibles:
                    # m is the number of infected neighbours, as per the problem specifications
                    m = len([neighbour for neighbour in self._topology.neighbors(susceptible.id) if
                             self.get_individuals_by_id(neighbour).state == "i"])
                    # compute the probability of infection, according to the formula given in the text
                    p_infection = 1 - (1 - beta) ** m
                    # susceptible becomes infected with probability p_infection
                    if np.random.rand() < p_infection:
                        new_infected.append(susceptible)

            # isolate the infected
            infected_people = self.get_individuals_by_state('i')
            if (n_infected := len(infected_people)) > 0:
                # for each one of them, pick a random number if this number < rho, infected becomes removed.
                probs_recovery = np.random.rand(n_infected, 1).reshape(n_infected, )
                # recovery or not
                recovery = np.array([prob_recovery < rho for prob_recovery in probs_recovery])
                # isolate the recovered
                recovered = np.array(infected_people)[recovery]
                # make them R
                for r in recovered:
                    self.update_individual(r.id, 'r')

            # turn the susceptibles into infected
            for new_infected_individual in new_infected:
                self.update_individual(new_infected_individual.id, 'i')

            # how many individuals in total are susceptible/infected/recovered at each week (to answer question 1.1.2)
            for state, number in self.population_statistics.items():
                recap_per_week[state][week] = number
            recap_per_week["new_infected"][week] = len(new_infected)
            recap_per_week["new_vaccinated"][week] = len(new_vaccinated)

        return recap_per_week
