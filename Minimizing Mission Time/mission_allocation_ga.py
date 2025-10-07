# ==============================================================================
#      Mission Allocation via Genetic Algorithm (Final Paper-Faithful Version)
#
# File Objective:
# This final version implements the interpretation that the GA solves for both
# allocation and routing simultaneously. It uses a permutation-based chromosome
# with dividers, which is a standard representation for MTSP, and employs
# compatible evolutionary operators (Order Crossover and Swap Mutation) to evolve
# complete, ordered tours for all UAVs.
# ==============================================================================

import numpy as np
import random
from typing import List, Dict, Tuple

import models

class MissionAllocationGA:
    """
    Manages the GA for mission assignment and routing, faithful to the MTSP model.
    """

    def __init__(self, gns: np.ndarray, num_uavs: int, data_center_pos: np.ndarray, transmission_radius_d: float, params: Dict):
        """Initializes the GA solver for the combined allocation and routing problem."""
        # Problem definition
        self.gns = gns
        self.num_gns = len(gns)
        self.num_uavs = num_uavs
        self.data_center_pos = data_center_pos
        self.transmission_radius_d = transmission_radius_d

        # GA parameters from config
        self.population_size = params['GA_POPULATION_SIZE']
        self.num_iterations = params['GA_NUM_ITERATIONS']
        self.crossover_rate = params['GA_CROSSOVER_RATE']
        self.mutation_rate = params['GA_MUTATION_RATE']
        self.tournament_size = params.get('GA_TOURNAMENT_SIZE', 3) 

        # Mission cost model parameters
        self.scaling_a = params['SCALING_FACTOR_A']
        self.scaling_b = params['SCALING_FACTOR_B']

        # GA state
        self.population = []
        self.best_solution_chromosome = None
        self.best_solution_fitness = float('inf')

    def _create_initial_population(self):
        """
        Generates the first generation. A chromosome is a permutation of all
        GN indices and M-1 dividers, representing a complete ordered solution.
        """
        self.population = []
        # Edge case: If there's only 1 UAV, there are no dividers.
        if self.num_uavs <= 1:
            base_chromosome = list(range(self.num_gns))
        else:
            # GN indices are 0 to N-1. Dividers are -1 to -(M-1).
            base_chromosome = list(range(self.num_gns)) + list(range(-1, -self.num_uavs, -1))
        
        for _ in range(self.population_size):
            shuffled_chromosome = random.sample(base_chromosome, len(base_chromosome))
            self.population.append(shuffled_chromosome)

    def _decode_chromosome(self, chromosome: List[int]) -> Dict[int, List[int]]:
        """
        Parses a chromosome with dividers into a dictionary of ordered routes for each UAV.
        """
        routes = {i: [] for i in range(self.num_uavs)}
        if self.num_uavs <= 1:
            routes[0] = [gene for gene in chromosome if gene >= 0]
            return routes

        current_uav_idx = 0
        current_route = []
        
        for gene in chromosome:
            if gene < 0: # It's a divider
                if current_uav_idx < self.num_uavs:
                    routes[current_uav_idx] = current_route
                current_uav_idx += 1
                current_route = []
            else: # It's a GN index
                current_route.append(gene)
        
        # Assign the last route after the last divider
        if current_uav_idx < self.num_uavs:
            routes[current_uav_idx] = current_route
        
        return routes

    def _calculate_fitness(self, chromosome: List[int]) -> float:
        """
        Evaluates the fitness of a single chromosome (lower is better).
        The order is taken directly from the chromosome.
        """
        routes = self._decode_chromosome(chromosome)
        all_tour_costs = []

        for uav_idx in range(self.num_uavs):
            tour_cost = 0.0
            route = routes.get(uav_idx, [])
            
            if not route:
                all_tour_costs.append(0.0)
                continue

            # Cost from data center to the first GN in the evolved order
            tour_cost += models.calculate_initial_mission_cost(
                self.data_center_pos, self.gns[route[0]], self.transmission_radius_d, self.scaling_a, self.scaling_b
            )
            
            # Cost between GNs according to the evolved order
            for i in range(len(route) - 1):
                tour_cost += models.calculate_initial_mission_cost(
                    self.gns[route[i]], self.gns[route[i+1]], self.transmission_radius_d, self.scaling_a, self.scaling_b
                )

            # Cost from the last GN back to the data center
            tour_cost += models.calculate_initial_mission_cost(
                self.gns[route[-1]], self.data_center_pos, self.transmission_radius_d, self.scaling_a, self.scaling_b
            )
            
            all_tour_costs.append(tour_cost)

        return max(all_tour_costs) if all_tour_costs else 0.0

    def _selection(self) -> List[int]:
        """Selects a parent using tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        tournament_fitness = [self._calculate_fitness(ind) for ind in tournament]
        winner_index = np.argmin(tournament_fitness)
        return tournament[winner_index]

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        Performs Order Crossover (OX1), which is suitable for permutation-based encodings.
        It preserves the relative order of genes from the parents.
        """
        size = len(parent1)
        child1, child2 = [None] * size, [None] * size
        
        start, end = sorted(random.sample(range(size), 2))
        
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        genes1_in_child = set(child1[start:end])
        genes2_in_child = set(child2[start:end])

        p2_idx, c1_idx = end, end
        while None in child1:
            gene = parent2[p2_idx % size]
            if gene not in genes1_in_child:
                child1[c1_idx % size] = gene
                c1_idx += 1
            p2_idx += 1

        p1_idx, c2_idx = end, end
        while None in child2:
            gene = parent1[p1_idx % size]
            if gene not in genes2_in_child:
                child2[c2_idx % size] = gene
                c2_idx += 1
            p1_idx += 1
            
        return child1, child2

    def _mutation(self, chromosome: List[int]) -> List[int]:
        """Performs swap mutation, which can change order or allocation."""
        idx1, idx2 = random.sample(range(len(chromosome)), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome

    def solve(self) -> Dict:
        """Runs the entire GA process to evolve both allocation and routes."""
        print("Starting Genetic Algorithm for mission allocation and routing...")
        self._create_initial_population()

        for gen in range(self.num_iterations):
            fitness_scores = [self._calculate_fitness(ind) for ind in self.population]

            min_fitness_in_gen = min(fitness_scores)
            if min_fitness_in_gen < self.best_solution_fitness:
                self.best_solution_fitness = min_fitness_in_gen
                best_idx = np.argmin(fitness_scores)
                self.best_solution_chromosome = self.population[best_idx][:]

            if (gen + 1) % 10 == 0:
                print(f"Generation {gen+1}/{self.num_iterations}, Best Fitness: {self.best_solution_fitness:.2f}")

            next_generation = []
            best_current_idx = np.argmin(fitness_scores)
            next_generation.append(self.population[best_current_idx]) # Elitism
            
            while len(next_generation) < self.population_size:
                parent1 = self._selection()
                parent2 = self._selection()

                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1[:], parent2[:]

                if random.random() < self.mutation_rate:
                    offspring1 = self._mutation(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = self._mutation(offspring2)
                
                next_generation.append(offspring1)
                if len(next_generation) < self.population_size:
                    next_generation.append(offspring2)

            self.population = next_generation

        final_routes_decoded = self._decode_chromosome(self.best_solution_chromosome)
        final_assignment = {f"UAV_{i}": final_routes_decoded.get(i, []) for i in range(self.num_uavs)}
        
        result = {
            "assignment": final_assignment,
            "min_max_cost": self.best_solution_fitness
        }
        
        print("Genetic Algorithm finished.")
        print(f"Final Min-Max Mission Cost: {result['min_max_cost']:.2f}")
        for uav, route in result['assignment'].items():
            print(f"  {uav}: Route {route}")
        return result
    